from __future__ import annotations

import random
from random import randrange, choice

from math import sqrt
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import torch
from torch import cat, stack, nn, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter
from torch.optim import Adam
from torch.func import functional_call

import torch.distributed as dist
torch.set_float32_matmul_precision('high')

from torch.nn.utils.parametrizations import weight_norm

import einx
from einops import reduce, repeat, rearrange, einsum, pack, unpack
from einops.layers.torch import Rearrange

from ema_pytorch import EMA

from tqdm import tqdm as orig_tqdm

from accelerate import Accelerator

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def identity(t, *args, **kwargs):
    return t

def first(seq):
    return seq[0]

def divisible_by(num, den):
    return (num % den) == 0

def xnor(x, y):
    return not (x ^ y)

def join(arr, delimiter):
    return delimiter.join(arr)

def is_empty(t):
    return t.numel() == 0

def item(t):
    if t.numel() == 0:
        out = t.item()
    else:
        out = t.tolist()

    return out

def arange_like(t, *, dim = None, length = None):
    assert exists(dim) or exists(length)

    if not exists(length):
        length = t.shape[dim]

    return torch.arange(length, device = t.device)

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(t, temp = 1.):
    is_greedy = temp <= 0.

    if not is_greedy:
        t = (t / temp) + gumbel_noise(t)

    return t.argmax(dim = -1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def orthogonal_(t):
    nn.init.orthogonal_(t.t())
    return t * sqrt(t.shape[-1])

def from_numpy(t):
    if isinstance(t, np.float64):
        t = np.array(t)

    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)

    return t.float()

# distributed

def maybe_all_reduce_mean(t):
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return t

    dist.all_reduce(t)
    return t / dist.get_world_size()

# networks

class StateNorm(Module):
    def __init__(
        self,
        dim_state,
        eps = 1e-5,
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754
        super().__init__()
        self.dim = dim_state
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim_state))
        self.register_buffer('running_variance', torch.ones(dim_state))

    def forward(
        self,
        state
    ):
        assert state.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {state.shape[-1]}'

        time = self.step.item()
        mean = self.running_mean
        variance = self.running_variance

        normed = (state - mean) / variance.sqrt().clamp(min = self.eps)

        if not self.training:
            return normed

        # update running mean and variance

        new_obs_mean = reduce(state, '... d -> d', 'mean')
        new_obs_mean = maybe_all_reduce_mean(new_obs_mean)

        delta = new_obs_mean - mean

        new_mean = mean + delta / time
        new_variance = (time - 1) / time * (variance + (delta ** 2) / time)

        self.step.add_(1)
        self.running_mean.copy_(new_mean)
        self.running_variance.copy_(new_variance)

        return normed

class Actor(Module):
    def __init__(
        self,
        dim_state,
        *,
        num_actions,
        continuous = False,
        hidden_dim = 32,
        accepts_latent = False,
        dim_latent = None,
        sample = False,
        weight_norm_linears = True,
        eps = 1e-5
    ):
        super().__init__()
        maybe_weight_norm = weight_norm if weight_norm_linears else identity
        self.weight_norm_linears = weight_norm_linears

        self.mem_norm = nn.RMSNorm(hidden_dim)

        self.proj_in = nn.Linear(dim_state, hidden_dim + 1, bias = False)
        self.proj_in = maybe_weight_norm(self.proj_in, name = 'weight', dim = None)

        self.to_embed = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.to_embed = maybe_weight_norm(self.to_embed, name = 'weight', dim = None)

        self.final_norm = nn.RMSNorm(hidden_dim)

        if continuous:
            self.to_mean = nn.Linear(hidden_dim, num_actions, bias = False)
            self.to_log_var = nn.Linear(hidden_dim, num_actions, bias = False)

            self.to_mean = maybe_weight_norm(self.to_mean, name = 'weight', dim = None)
            self.to_log_var = maybe_weight_norm(self.to_log_var, name = 'weight', dim = None)
        else:
            self.to_logits = nn.Linear(hidden_dim, num_actions, bias = False)
            self.to_logits = maybe_weight_norm(self.to_logits, name = 'weight', dim = None)

        self.continuous = continuous

        self.norm_weights_()

        # whether to sample from the output discrete logits

        self.sample = sample

        # for genes -> expression network (the analogy is growing on me)

        self.accepts_latent = accepts_latent
        if accepts_latent:
            assert exists(dim_latent)

            self.encode_latent = nn.Linear(dim_latent, hidden_dim)
            self.encode_latent = maybe_weight_norm(self.encode_latent, name = 'weight', dim = None)
            self.post_norm_latent_added = nn.RMSNorm(hidden_dim)

        self.register_buffer('init_hiddens', torch.zeros(hidden_dim))

    def norm_weights_(self):
        if not self.weight_norm_linears:
            return

        for param in self.parameters():
            if not isinstance(param, nn.Linear):
                continue

            param.parametrization.weight.original.copy_(param.weight)

    def forward(
        self,
        x,
        hiddens = None,
        latent = None,
        sample_temperature = 1.
    ):
        assert xnor(exists(latent), self.accepts_latent)

        x = self.proj_in(x)
        x, forget = x[:-1], x[-1]

        x = F.silu(x)

        if exists(hiddens):
            past_mem = self.mem_norm(hiddens) * forget.sigmoid()
            x = x + past_mem

        if self.accepts_latent:
            latent = l2norm(latent) # could be noised
            x = x + self.encode_latent(latent)
            x = self.post_norm_latent_added(x)

        x = self.to_embed(x)
        hiddens = F.silu(x)

        embed = self.final_norm(hiddens)

        if not self.continuous:
            raw_actions = self.to_logits(embed)
        else:
            mean, log_var = self.to_mean(embed), self.to_log_var(embed)
            raw_actions = stack((mean, log_var))

        if not self.sample:
            return raw_actions, hiddens

        # actor can return sampled action(s) for the simulation / environment

        if not self.continuous:
            action_logits = raw_actions
            actions = gumbel_sample(action_logits, temp = sample_temperature)
        else:
            mean, raw_std = raw_actions
            std = raw_std.sigmoid() * 3.
            actions = torch.normal(mean, std * sample_temperature).tanh() # todo - accept action range and do scale and shift
            actions = actions.tanh()

        return actions, hiddens

# an actor wrapper that contains the state normalizer and latent gene pool, defaults to calling the fittest gene

class ActorWrapper(Module):
    def __init__(
        self,
        actor: Module,
        *,
        state_norm: StateNorm | None = None,
        latent_gene_pool: LatentGenePool | None = None,
        default_latent_gene_id = 0
    ):
        super().__init__()
        self.actor = actor
        self.state_norm = state_norm
        self.latents = latent_gene_pool

        self.default_latent_gene_id = default_latent_gene_id

    def forward(
        self,
        state,
        hiddens = None,
        latent_gene_id = None
    ):
        latent_gene_id = default(latent_gene_id, self.default_latent_gene_id)

        if exists(self.state_norm):
            self.state_norm.eval()

            with torch.no_grad():
                state = self.state_norm(state)

        latent = None

        if exists(self.latents):
            latent = self.latents[latent_gene_id]

        out = self.actor(
            state,
            hiddens = hiddens,
            latent = latent
        )

        return out

# latent gene pool

# proposed by Wang et al. evolutionary policy optimization (EPO)
# https://arxiv.org/abs/2503.19037

class LatentGenePool(Module):
    def __init__(
        self,
        dim,
        num_genes_per_island,
        num_selected,
        tournament_size,
        num_elites = 1,             # exempt from genetic mutation and migration
        mutation_std_dev = 0.1,
        num_islands = 1,
        migrate_genes_every = 10,   # every number of evolution step to do a migration between islands, if using multi-islands for increasing diversity
        num_frac_migrate = 0.1      # migrate 10 percent of the bottom population
    ):
        super().__init__()
        assert num_islands >= 1
        assert num_genes_per_island > 2

        self.num_islands = num_islands

        num_genes = num_genes_per_island * num_islands
        self.num_genes = num_genes
        self.num_genes_per_island = num_genes_per_island

        assert 2 <= num_selected < num_genes_per_island, f'must select at least 2 genes for mating'

        self.num_selected = num_selected
        self.num_children = num_genes_per_island - num_selected
        self.tournament_size = tournament_size

        self.dim_gene = dim
        self.genes = nn.Parameter(l2norm(torch.randn(num_genes, dim)))

        self.split_islands = Rearrange('(i g) ... -> i g ...', i = num_islands)
        self.merge_islands = Rearrange('i g ... -> (i g) ...')

        self.num_elites = num_elites # todo - redo with affinity maturation algorithm from artificial immune system field
        self.mutation_std_dev = mutation_std_dev

        assert 0. <= num_frac_migrate <= 1.

        self.num_frac_migrate = num_frac_migrate
        self.migrate_genes_every = migrate_genes_every

        self.register_buffer('step', tensor(0))

    def __getitem__(self, idx):
        return l2norm(self.genes[idx])

    @torch.inference_mode()
    def evolve(
        self,
        fitnesses,
        temperature = 1.5
    ):
        device, num_selected = fitnesses.device, self.num_selected
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.num_genes

        # split out the islands

        genes = self.genes
        num_islands = self.num_islands
        has_elites = self.num_elites > 0

        fitnesses = self.split_islands(fitnesses)
        genes = self.split_islands(genes)

        # local competition within each island

        sorted_fitness, sorted_gene_ids = fitnesses.sort(dim = -1, descending = True)

        selected_gene_ids = sorted_gene_ids[:, :num_selected]
        selected_fitness = sorted_fitness[:, :num_selected]

        selected_gene_ids_for_gather = repeat(selected_gene_ids, '... -> ... d', d = self.dim_gene)

        selected_genes = genes.gather(1, selected_gene_ids_for_gather)

        # tournament

        num_children = self.num_children

        batch_randperm = torch.randn((num_islands, num_children, num_selected), device = device).argsort(dim = -1)
        tourn_ids = batch_randperm[..., :self.tournament_size]

        sorted_fitness = repeat(sorted_fitness, '... -> ... d', d = tourn_ids.shape[-1])

        tourn_fitness_ids = sorted_fitness.gather(1, tourn_ids)

        parent_ids = tourn_fitness_ids.topk(2, dim = -1).indices

        parent_ids = rearrange(parent_ids, 'i g parents -> i (g parents)')

        parent_ids = repeat(parent_ids, '... -> ... d', d = self.dim_gene)

        parents = selected_genes.gather(1, parent_ids)
        parents = rearrange(parents, 'i (g parents) d -> parents i g d', parents = 2)

        # cross over

        parent1, parent2 = parents

        children = parent1.lerp(parent2, (torch.randn_like(parent1) / temperature).sigmoid())

        # maybe migration

        if (
            divisible_by(self.step.item() + 1, self.migrate_genes_every) and
            self.num_islands > 1 and
            self.num_frac_migrate > 0.
        ):

            if has_elites:
                elites, selected_genes = selected_genes[:, :1], selected_genes[:, 1:]

            num_can_migrate = selected_genes.shape[1]

            num_migrate = max(1, num_can_migrate * self.num_frac_migrate)

            # fixed migration pattern - what i observe to work best, for now
            # todo - option to make it randomly selected with a mask

            selected_genes, migrants = selected_genes[:, -num_migrate:], selected_genes[:, :-num_migrate]

            migrants = torch.roll(migrants, 1, dims = (1,))

            selected_genes = cat((selected_genes, migrants), dim = 1)

            if has_elites:
                selected_genes = cat((elites, selected_genes), dim = 1)

        # concat children

        genes = torch.cat((selected_genes, children), dim = 1)

        # mutate

        if self.mutation_std_dev > 0:

            if has_elites:
                elites, genes = genes[:, :1], genes[:, 1:]

            genes.add_(torch.randn_like(genes) * self.mutation_std_dev)

            if has_elites:
                genes = torch.cat((elites, genes), dim = 1)

        genes = self.merge_islands(genes)

        self.genes.copy_(l2norm(genes))

        self.step.add_(1)

        return selected_gene_ids # return the selected gene ids, for the outer learning orchestrator to determine which mutations to accept

# main class

class BlackboxGradientSensing(Module):

    def __init__(
        self,
        actor: Module,
        *,
        accelerator: Accelerator | None = None,
        state_norm: StateNorm | Module | dict | None  = None,
        actor_is_recurrent = False,
        latent_gene_pool: LatentGenePool | dict | None = None,
        concat_latent_to_state = False, # if False, will pass in the latents as a kwarg `latent`, else try to concat it to the state
        crossover_every_step = 1,
        crossover_after_step = 0,
        num_env_interactions = 1000,
        noise_pop_size = 40,
        noise_std_dev: dict[str, float] | float = 0.1, # Appendix F in paper, appears to be constant for sim and real
        mutate_latent_genes = False,
        latent_gene_noise_std_dev = 1e-4,
        factorized_noise = True,
        orthogonalized_noise = True,
        num_selected = 8,    # of the population, how many of the best performing noise perturbations to accept
        num_rollout_repeats = 3,
        optim_klass = Adam,
        learning_rate = 8e-2,
        weight_decay = 1e-4,
        betas = (0.9, 0.95),
        max_timesteps = 500,
        calc_fitness: Callable[[Tensor], Tensor] | None = None,
        param_names: set[str] | str | None = None,
        modules_to_optimize: set[str] | str | None = None,
        show_progress = True,
        optim_kwargs: dict = dict(),
        optim_step_post_hook: Callable | None = None,
        accelerate_kwargs: dict = dict(),
        num_std_below_mean_thres_accept = 0., # for each reward + anti, if they are below this number of standard deviations below the mean, reject it
        frac_genes_pass_thres_accept = 0.9,    # in population based training, the fraction of genes that must be all above a given reward threshold for that noise to be accepted
        cpu = False,
        torch_compile_actor = True,
        use_ema = False,
        ema_decay = 0.9,
        update_model_with_ema_every = 100,
        sample_actions_from_actor = True
    ):
        super().__init__()
        assert num_selected < noise_pop_size, f'number of selected noise must be less than the total population of noise'

        # ES(1+1) related

        self.num_selected = num_selected
        self.noise_pop_size = noise_pop_size
        self.num_rollout_repeats = num_rollout_repeats

        self.orthogonalized_noise = orthogonalized_noise    # orthogonalized noise - todo: add the fast hadamard-rademacher ones proposed in paper
        self.factorized_noise = factorized_noise            # maybe factorized gaussian noise

        # use accelerate to manage distributed

        if not exists(accelerator):

            if cpu:
                assert 'cpu' not in accelerate_kwargs
                accelerate_kwargs = {'cpu': True, **accelerate_kwargs}

            accelerator = Accelerator(**accelerate_kwargs)

        device = accelerator.device
        self.accelerator = accelerator

        # net

        self.actor = actor.to(device)

        self.use_ema = use_ema
        self.ema_actor = EMA(actor, beta = ema_decay, update_model_with_ema_every = update_model_with_ema_every, include_online_model = False) if use_ema else None

        self.torch_compile_actor = torch_compile_actor

        self.actor_is_recurrent = actor_is_recurrent # if set to True, actor must pass out the memory on forward on the second position, then receive it as a kwarg of `hiddens`

        named_params = dict(actor.named_parameters())
        named_modules = dict(actor.named_modules())

        # whether to sample actions from the actor

        self.sample_actions_from_actor = sample_actions_from_actor

        # handle only a subset of parameters being optimized

        if isinstance(param_names, str):
            param_names = {param_names}

        # also handle if module names are passed in
        # ex. optimizing some gating / routing neural network that ties together a bunch of other pretrained policies

        if isinstance(modules_to_optimize, str):
            modules_to_optimize = {modules_to_optimize}

        if exists(modules_to_optimize):
            param_names = default(param_names, set())

            for module_name in modules_to_optimize:
                module = named_modules[module_name]
                module_param_names = dict(module.named_parameters()).keys()
                module_param_names_with_prefix = [f'{module_name}.{param_name}' for param_name in module_param_names]

                param_names |= set(module_param_names_with_prefix)

        param_names = default(param_names, set(named_params.keys()))

        # validate and set parameters to optimize

        assert len(param_names) > 0, f'no parameters to optimize with evolutionary strategy'

        self.param_names = param_names

        # noise std deviations, which can be one fixed value, or tailored to specific value per parameter name

        if isinstance(noise_std_dev, float):
            noise_std_dev = {name: noise_std_dev for name in self.param_names}

        self.noise_std_dev = noise_std_dev

        # env interactions

        self.max_timesteps = max_timesteps

        # gene pool, another axis for scaling and bitter lesson

        num_genes = 1
        gene_pool = None

        if isinstance(latent_gene_pool, dict):
            gene_pool = LatentGenePool(**latent_gene_pool)
            gene_pool.to(device)

            num_genes = gene_pool.num_genes

        self.actor_accepts_latents = exists(gene_pool)
        self.concat_latent_to_state = concat_latent_to_state

        self.gene_pool = gene_pool
        self.num_genes = num_genes

        def default_calc_fitness(reward_stats):
            return reduce(reward_stats[:, 0], 'g s e -> g', 'mean')

        self.calc_fitness = default(calc_fitness, default_calc_fitness)

        self.crossover_every_step = crossover_every_step
        self.crossover_after_step = crossover_after_step

        # whether to do heritable mutations to the latent genes

        self.mutate_latent_genes = mutate_latent_genes

        self.latent_gene_noise_std_dev = latent_gene_noise_std_dev

        # optim

        optim_params = [named_params[param_name] for param_name in self.param_names]

        self.optim = optim_klass(optim_params, lr = learning_rate, betas = betas, **optim_kwargs)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # hooks

        if exists(optim_step_post_hook):
            def hook(*_):
                optim_step_post_hook()

            self.optim.register_step_post_hook(hook)

        # maybe state norm

        if isinstance(state_norm, dict):
            state_norm = StateNorm(**state_norm)

        self.use_state_norm = exists(state_norm)

        if self.use_state_norm:
            self.state_norm = state_norm
            state_norm.to(device)

        # progress bar

        self.show_progress = show_progress

        # number of interactions with environment for learning

        self.num_env_interactions = num_env_interactions

        # calculate num of episodes per learning cycle for this machine

        world_size, rank = accelerator.num_processes, accelerator.process_index

        # for each gene, roll out for each noise candidate

        gene_indices = torch.arange(num_genes)
        mutation_indices = torch.arange(noise_pop_size + 1)

        gene_mutation_indices = torch.cartesian_prod(gene_indices, mutation_indices)
        rollouts_for_machine = gene_mutation_indices.chunk(world_size)[rank]

        self.register_buffer('rollouts_for_machine', rollouts_for_machine, persistent = False)

        # for each reward and its anti, the number of standard deviations below the baseline they can be for acceptance

        self.num_std_below_mean_thres_accept = num_std_below_mean_thres_accept

        # the fraction of genes that must be above the given reward threshold as defined by the variable above, in order for said noise to be accepted

        assert 0 <= frac_genes_pass_thres_accept <= 1.
        self.frac_genes_pass_thres_accept = frac_genes_pass_thres_accept

        # expose a few computed variables

        self.num_episodes_per_learning_cycle = self.rollouts_for_machine.shape[0] * num_rollout_repeats * 2

        self.is_main = rank == 0

        # keep track of number of steps

        self.register_buffer('step', tensor(0))

    def sync_seed_(self):
        acc = self.accelerator
        rand_int = torch.randint(0, int(1e7), (), device = acc.device)
        seed = acc.reduce(rand_int)
        torch.manual_seed(seed.item())

    def log(self, **data):
        return self.accelerator.log(data, step = self.step.item())

    def save(self, path, overwrite = False):

        acc = self.accelerator

        acc.wait_for_everyone()

        if not acc.is_main_process:
            return

        path = Path(path)
        assert overwrite or not path.exists()

        pkg = dict(
            actor = self.actor.state_dict(),
            ema_actor = self.ema_actor.state_dict() if self.use_ema else None,
            state_norm = self.state_norm.state_dict() if self.use_state_norm else None,
            latents = self.gene_pool.state_dict() if exists(self.gene_pool) else None,
            step = self.step
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)

        assert path.exists()

        pkg = torch.load(str(path), weights_only = True)

        self.actor.load_state_dict(pkg['actor'])

        if exists(self.gene_pool):
            self.gene_pool.load_state_dict(pkg['latents'])

        if self.use_ema:
            assert 'ema_actor' in pkg
            self.ema_actor.load_state_dict(pkg['ema_actor'])

        if self.use_state_norm:
            assert 'state_norm' in pkg
            self.state_norm.load_state_dict(pkg['state_norm'])

        self.step.copy_(pkg['step'])

    def return_wrapped_actor(self) -> ActorWrapper | Module:

        if not (self.use_state_norm or self.actor_accepts_latents):
            return self.actor

        wrapped_actor = ActorWrapper(
            self.actor,
            state_norm = self.state_norm if self.use_state_norm else None,
            latent_gene_pool = self.gene_pool if self.actor_accepts_latents else None,
        )

        return wrapped_actor

    @torch.inference_mode()
    def forward(
        self,
        maybe_envs,
        num_env_interactions = None,
        show_progress = None,
        seed = None,
        max_timesteps_per_interaction = None,
    ):
        max_timesteps_per_interaction = default(max_timesteps_per_interaction, self.max_timesteps)
        show_progress = default(show_progress, self.show_progress)
        num_env_interactions = default(num_env_interactions, self.num_env_interactions)

        (
            learning_rate,
            num_selected,
            noise_pop_size,
            num_rollout_repeats,
            factorized_noise,
            orthogonalized_noise,
            noise_std_dev
        ) = (
            self.learning_rate,
            self.num_selected,
            self.noise_pop_size,
            self.num_rollout_repeats,
            self.factorized_noise,
            self.orthogonalized_noise,
            self.noise_std_dev
        )

        acc, optim = self.accelerator, self.optim

        actor = self.actor if not self.use_ema else self.ema_actor.ema_model

        is_recurrent_actor = self.actor_is_recurrent

        if self.torch_compile_actor:
            actor = torch.compile(actor)

        is_distributed, is_main, device = (
            acc.use_distributed,
            acc.is_main_process,
            acc.device
        )

        tqdm = partial(orig_tqdm, disable = not is_main or not show_progress)

        if exists(seed):
            torch.manual_seed(seed)

        # params

        params = dict(self.actor.named_parameters())

        # outer learning update progress bar

        learning_updates = tqdm(range(num_env_interactions), position = 0)

        for _ in learning_updates:

            self.step.add_(1)

            # synchronize a global seed

            if is_distributed:
                self.sync_seed_()

            # keep track of the rewards received per noise and its negative

            pop_size_with_baseline = noise_pop_size + 1

            reward_stats = torch.zeros((
                self.num_genes,           # latent genes for cross over
                pop_size_with_baseline,   # mutation
                2,                        # mutation with its anti
                num_rollout_repeats       # reducing variance with repeat
            )).to(device)

            # episode seed is shared for one learning cycle
            # todo - allow for multiple episodes per learning cycle or mutation accumulation

            episode_seed = torch.randint(0, int(1e7), ()).item()

            random.seed(episode_seed)

            # create noises upfront

            episode_states = []
            noises = dict()

            for key, param in params.items():

                if key not in self.param_names:
                    continue

                param_noise_std_dev = noise_std_dev[key]


                if factorized_noise and param.ndim == 2:
                    i, j = param.shape

                    rows = torch.randn((pop_size_with_baseline, i), device = device)
                    cols = torch.randn((pop_size_with_baseline, j), device = device)

                    if orthogonalized_noise:
                        rows = orthogonal_(rows)
                        cols = orthogonal_(cols)

                    rows, cols = tuple(t.sign() * t.abs().sqrt() for t in (rows, cols))

                    noises_for_param = einx.multiply('p i, p j -> p i j', rows, cols)

                elif orthogonalized_noise and param.ndim == 2:
                    # p - population size

                    noises_for_param = torch.randn((pop_size_with_baseline, *param.shape), device = device)

                    noises_for_param, packed_shape = pack([noises_for_param], 'p *')
                    orthogonal_(noises_for_param)
                    noises_for_param = first(unpack(noises_for_param, packed_shape, 'p *'))

                else:
                    noises_for_param = torch.randn((pop_size_with_baseline, *param.shape), device = device)

                noises_for_param[0].zero_() # first is for baseline

                noises[key] = noises_for_param * param_noise_std_dev

            # determine noise for latents

            if self.mutate_latent_genes and self.actor_accepts_latents:
                genes_shape = self.gene_pool.genes.shape
                all_latent_noises = torch.randn((pop_size_with_baseline, *genes_shape), device = device) * self.latent_gene_noise_std_dev

                all_latent_noises[0].zero_() # first for baseline

            # maybe domain randomization

            if isinstance(maybe_envs, (list, tuple)):
                env = choice(maybe_envs)
            else:
                env = maybe_envs

            # maybe shard the interaction with environments for the individual noise perturbations

            for gene_noise_index in tqdm(self.rollouts_for_machine.tolist(), desc = 'noise index', position = 1, leave = False):

                gene_index, noise_index = gene_noise_index

                # prepare the latent gene, if needed

                if self.actor_accepts_latents:
                    latent_gene = self.gene_pool[gene_index]

                    if self.mutate_latent_genes:
                        latent_gene_noises = all_latent_noises[:, gene_index]

                # prepare the mutation

                noise = {key: noises_for_param[noise_index] for key, noises_for_param in noises.items()}

                for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                    param_with_noise = {name: Parameter(param + noise[name] * sign) if name in self.param_names else param for name, param in params.items()}

                    for repeat_index in tqdm(range(num_rollout_repeats), desc = 'rollout repeat', position = 3, leave = False):

                        state = env.reset(seed = episode_seed)

                        if isinstance(state, tuple):
                            state, *_ = state

                        episode_states.clear()

                        total_reward = 0.

                        if is_recurrent_actor:
                            assert hasattr(actor, 'init_hiddens'), 'your actor must have an `init_hiddens` buffer if to be used recurrently'
                            mem = actor.init_hiddens

                        for timestep in range(max_timesteps_per_interaction):

                            state = from_numpy(state).to(device)

                            episode_states.append(state)

                            if self.use_state_norm:
                                self.state_norm.eval()
                                state = self.state_norm(state)

                            kwargs = dict()

                            if is_recurrent_actor:
                                kwargs.update(hiddens = mem)

                            actor_state_input = state

                            if self.actor_accepts_latents:

                                if self.mutate_latent_genes:
                                    latent_gene_noise = latent_gene_noises[noise_index] * sign

                                    latent_gene = latent_gene + latent_gene_noise

                                if self.concat_latent_to_state:
                                    actor_state_input = cat((state, latent_gene))
                                else:
                                    kwargs.update(latent = latent_gene)

                            actor_out = functional_call(actor, param_with_noise, actor_state_input, kwargs = kwargs)

                            # take care of recurrent network
                            # the nicest thing about ES is learning recurrence / memory without much hassle (in fact, can be non-differentiable)

                            if isinstance(actor_out, tuple):
                                action_or_logits, *actor_rest_out = actor_out
                            else:
                                action_or_logits = actor_out
                                actor_rest_out = []

                            if is_recurrent_actor:
                                assert len(actor_rest_out) > 0
                                mem, *_ = actor_rest_out

                            # sample

                            if self.sample_actions_from_actor:
                                action = gumbel_sample(action_or_logits)
                                action = item(action)
                            else:
                                action = item(action_or_logits)

                            env_out = env.step(action)

                            # flexible output from env

                            assert isinstance(env_out, tuple)

                            len_env_out = len(env_out)

                            if len_env_out >= 4:
                                next_state, reward, terminated, truncated, *_ = env_out
                                done = terminated or truncated
                            elif len_env_out == 3:
                                next_state, reward, done = env_out
                            elif len_env_out == 2:
                                next_state, reward = env_out
                                done = False
                            else:
                                raise RuntimeError('invalid number of items received from environment')

                            total_reward += float(reward)

                            if done:
                                break

                            state = next_state
                    
                        reward_stats[gene_index, noise_index, sign_index, repeat_index] = total_reward

            # maybe synchronize reward stats, as well as min episode length for updating state norm

            if is_distributed:
                reward_stats = acc.reduce(reward_stats)

                if self.use_state_norm:
                    episode_state_len = tensor(len(episode_states), device = device)

                    min_episode_state_len = acc.gather(episode_state_len).amin().item()

                    episode_states = episode_states[:min_episode_state_len]

            # update state norm with one episode worth (as it is repeated)

            if self.use_state_norm:
                self.state_norm.train()

                for state in episode_states:
                    self.state_norm(state)

            # update based on eq (3) and (4) in the paper
            # their contribution is basically to use reward deltas (for a given noise and its negative sign) for sorting for the 'elite' directions

            # g - latent / gene, n - noise / mutation, s - sign, e - episode

            reward_std = reward_stats.std()

            reward_mean = reduce(reward_stats, 'g n s e -> g n s', 'mean')

            # split out the baseline

            baseline_mean, reward_mean = reward_mean[..., 0, :].mean(dim = -1), reward_mean[..., 1:, :]

            reward_deltas = reward_mean[..., 0] - reward_mean[..., 1]

            # mask out any noise candidates whose max reward mean is greater than baseline

            reward_threshold_accept = baseline_mean - reward_std * self.num_std_below_mean_thres_accept

            max_reward_mean = torch.amax(reward_mean, dim = -1)

            accept_mask = einx.greater_equal('g n, g -> g n', max_reward_mean, reward_threshold_accept)
            accept_mask = reduce(accept_mask.float(), 'g n -> n', 'mean') >= self.frac_genes_pass_thres_accept

            reward_deltas = einx.multiply('g n, n', reward_deltas, accept_mask.float()) # just zero out the reward deltas that do not pass the threshold

            # progress bar

            pbar_descriptions = [
                f'rewards: {baseline_mean.mean().item():.2f}',
                f'best: {reward_mean.amax().item():.2f}',
                f'accepted: {accept_mask.sum().item()} / {noise_pop_size}'
            ]

            def calculate_weights(reward_deltas, log = True):

                # get the top performing noise indices

                k = min(num_selected, reward_deltas.numel() // 2)

                ranked_reward_deltas, ranked_reward_indices = reward_deltas.abs().topk(k, dim = -1)

                # get the weights for the weighted sum of the topk noise according to eq (3)

                weights = ranked_reward_deltas / reward_std.clamp(min = 1e-3)

                # multiply by sign

                weights *= torch.sign(reward_deltas.gather(-1, ranked_reward_indices))

                if log:
                    pbar_descriptions.append(f'best delta: {ranked_reward_deltas.amax().item():.2f}')

                return weights, ranked_reward_indices

            # get the weights for update

            (
                weights,
                ranked_reward_indices,
            ) = calculate_weights(reduce(reward_deltas, 'g n -> n', 'mean'))

            # update the param one by one

            for name, noise in noises.items():

                param = params[name]

                # add the best "elite" noise directions weighted by eq (3)

                best_noises = noise[1:][ranked_reward_indices]

                update = einsum(best_noises, weights, 'n ..., n -> ...')

                param.grad = -update

            # update latents if needed

            if self.actor_accepts_latents and self.mutate_latent_genes:

                (
                    weights,
                    ranked_reward_indices,
                ) = calculate_weights(reward_deltas, log = False)

                # [n] g d, g sel -> sel g d

                ranked_reward_indices = rearrange(ranked_reward_indices, 'g sel -> sel g')

                ranked_reward_indices_for_gather = repeat(ranked_reward_indices, '... -> ... d', d = all_latent_noises.shape[-1])

                sel_noises = all_latent_noises.gather(0, ranked_reward_indices_for_gather)

                # weighted update, accounting for population dimension

                update = einsum(sel_noises, weights, 'sel g ..., g sel -> g ...')

                genes = self.gene_pool.genes

                # add to update

                genes.add_(update * learning_rate)

            # decay for norm gammas back to identity

            for mod in actor.modules():
                if isinstance(mod, nn.RMSNorm):
                    mod.weight.lerp_(torch.ones_like(mod.weight), self.weight_decay)

            # use optimizer to manage step

            optim.step()
            optim.zero_grad()

            # maybe ema

            if self.use_ema:
                self.ema_actor.update()

            # maybe crossover, if a genetic population is present
            # the crossover needs to happen before the mutation, as we will discard the mutation contributions from the genes that get selected out.

            if (
                exists(self.gene_pool) and
                self.step.item() > self.crossover_after_step and
                divisible_by(self.step.item(), self.crossover_every_step)
            ):
                # only include baseline for now, but could include the mutation rewards for selecting for meta-learning attributes.

                fitnesses = self.calc_fitness(reward_stats)

                self.sync_seed_()
                self.gene_pool.evolve(fitnesses)

            # logging

            learning_updates.set_description(join(pbar_descriptions, ' | '))

            # log to experiment tracker

            self.log(
                rewards = baseline_mean.mean().item()
            )
