from __future__ import annotations

from copy import deepcopy
from random import randrange
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import torch
from torch import nn, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter
from torch.optim import Adam
from torch.func import functional_call

import torch.distributed as dist
torch.set_float32_matmul_precision('high')

from torch.nn.utils.parametrizations import weight_norm

from einops import reduce, rearrange, einsum, pack, unpack

from ema_pytorch import EMA

from tqdm import tqdm as orig_tqdm

from accelerate import Accelerator

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def xnor(x, y):
    return not (x ^ y)

def join(arr, delimiter):
    return delimiter.join(arr)

def is_empty(t):
    return t.numel() == 0

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
        hidden_dim = 32,
        accepts_latent = False,
        dim_latent = None
    ):
        super().__init__()
        self.mem_norm = nn.RMSNorm(hidden_dim)

        self.proj_in = nn.Linear(dim_state, hidden_dim + 1, bias = False)
        self.proj_in = weight_norm(self.proj_in, name = 'weight', dim = None)

        self.to_embed = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.to_embed = weight_norm(self.to_embed, name = 'weight', dim = None)

        self.to_logits = nn.Linear(hidden_dim, num_actions, bias = False)
        self.to_logits = weight_norm(self.to_logits, name = 'weight', dim = None)

        self.norm_weights_()

        # for genes -> expression network (the analogy is growing on me)

        self.accepts_latent = accepts_latent
        if accepts_latent:
            assert exists(dim_latent)

            self.encode_latent = nn.Sequential(
                nn.Linear(dim_latent, hidden_dim),
                nn.SiLU()
            )

        self.register_buffer('init_hiddens', torch.zeros(hidden_dim))

    def norm_weights_(self):
        for param in self.parameters():
            if not isinstance(param, nn.Linear):
                continue

            param.parametrization.weight.original.copy_(param.weight)

    def forward(
        self,
        x,
        hiddens = None,
        latent = None
    ):
        assert xnor(exists(latent), self.accepts_latent)

        x = self.proj_in(x)
        x, forget = x[:-1], x[-1]

        x = F.silu(x)

        if exists(hiddens):
            past_mem = self.mem_norm(hiddens) * forget.sigmoid()
            x = x + past_mem

        if self.accepts_latent:
            x = x * self.encode_latent(latent)

        x = self.to_embed(x)
        x = F.silu(x)

        return self.to_logits(x), x

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
        num_genes,
        num_selected,
        tournament_size,
        num_elites = 1, # exempt from genetic mutation
        mutation_std_dev = 0.1,
    ):
        super().__init__()
        assert num_genes > 2

        self.num_genes = num_genes

        assert 2 <= num_selected < num_genes, f'must select at least 2 genes for mating'

        self.num_selected = num_selected
        self.tournament_size = tournament_size

        self.genes = nn.Parameter(l2norm(torch.randn(num_genes, dim)))

        self.num_elites = num_elites # todo - redo with affinity maturation algorithm from artificial immune system field
        self.mutation_std_dev = mutation_std_dev

    def __getitem__(self, idx):
        return l2norm(self.genes[idx])

    @torch.inference_mode()
    def evolve_with_cross_over(
        self,
        fitnesses,
        temperature = 1.5
    ):
        device, num_selected = fitnesses.device, self.num_selected
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.num_genes

        sorted_fitness, sorted_gene_ids = fitnesses.sort(dim = -1, descending = True)

        selected_gene_ids = sorted_gene_ids[:num_selected]
        selected_fitness = sorted_fitness[:num_selected]

        selected_genes = self[selected_gene_ids]

        # tournament

        num_children = self.num_genes - num_selected

        batch_randperm = torch.randn((num_children, num_selected), device = device).argsort(dim = -1)
        tourn_ids = batch_randperm[:, :self.tournament_size]

        tourn_fitness_ids = sorted_fitness[tourn_ids]

        parent_ids = tourn_fitness_ids.topk(2, dim = -1).indices

        parents = selected_genes[parent_ids]

        # cross over

        parent1, parent2 = parents.unbind(dim = 1)

        children = parent1.lerp(parent2, (torch.randn_like(parent1) / temperature).sigmoid())

        genes = torch.cat((selected_genes, children), dim = 0)

        # mutate

        if self.mutation_std_dev > 0:

            elites, genes = genes[:1], genes[1:]

            genes.add_(torch.randn_like(genes) * self.mutation_std_dev)

            genes = torch.cat((elites, genes), dim = 0)

        self.genes.copy_(l2norm(genes))

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
        crossover_every_step = 2,
        crossover_after_step = 0,
        num_env_interactions = 1000,
        noise_pop_size = 40,
        noise_std_dev = 0.1, # Appendix F in paper, appears to be constant for sim and real
        factorized_noise = True,
        num_selected = 8,    # of the population, how many of the best performing noise perturbations to accept
        num_rollout_repeats = 3,
        optim_klass = Adam,
        learning_rate = 8e-2,
        weight_decay = 1e-4,
        betas = (0.9, 0.95),
        max_timesteps = 400,
        calc_fitness: Callable[[Tensor], Tensor] | None = None,
        param_names: set[str] | str | None = None,
        modules_to_optimize: set[str] | str | None = None,
        show_progress = True,
        optim_kwargs: dict = dict(),
        optim_step_post_hook: Callable | None = None,
        post_noise_added_hook: Callable | None = None,
        accelerate_kwargs: dict = dict(),
        threshold_accept_learning_cycle = 2,
        cpu = False,
        torch_compile_actor = True,
        use_ema = False,
        ema_decay = 0.95,
        update_model_with_ema_every = 100
    ):
        super().__init__()
        assert num_selected < noise_pop_size, f'number of selected noise must be less than the total population of noise'

        # ES(1+1) related

        self.num_selected = num_selected
        self.noise_pop_size = noise_pop_size
        self.noise_std_dev = noise_std_dev
        self.num_rollout_repeats = num_rollout_repeats
        self.factorized_noise = factorized_noise # maybe factorized gaussian noise

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

        # gene pool, another axis for scaling and bitter lesson

        num_genes = 1
        gene_pool = None

        if isinstance(latent_gene_pool, dict):
            gene_pool = LatentGenePool(**latent_gene_pool)
            gene_pool.to(device)

            num_genes = gene_pool.num_genes

        self.actor_accepts_latents = exists(gene_pool)

        self.gene_pool = gene_pool
        self.num_genes = num_genes

        def default_calc_fitness(reward_stats):
            return reduce(reward_stats[:, 0], 'g s e -> g', 'mean')

        self.calc_fitness = default(calc_fitness, default_calc_fitness)

        self.crossover_every_step = crossover_every_step
        self.crossover_after_step = crossover_after_step

        # optim

        optim_params = [named_params[param_name] for param_name in self.param_names]

        self.optim = optim_klass(optim_params, lr = learning_rate, betas = betas)

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

        # threshold of valid reward deltas in order to accept weighted mutations

        assert 2 <= threshold_accept_learning_cycle <= noise_pop_size
        self.threshold_accept_learning_cycle = threshold_accept_learning_cycle

        # expose a few computed variables

        self.num_episodes_per_learning_cycle = self.rollouts_for_machine.shape[0]

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
        env,
        num_env_interactions = None,
        show_progress = None,
        seed = None,
        max_timesteps_per_interaction = 500,
    ):
        show_progress = default(show_progress, self.show_progress)
        num_env_interactions = default(num_env_interactions, self.num_env_interactions)

        (
            num_selected,
            noise_pop_size,
            num_rollout_repeats,
            factorized_noise,
            noise_std_dev
        ) = self.num_selected, self.noise_pop_size, self.num_rollout_repeats, self.factorized_noise, self.noise_std_dev

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

            # create noises upfront

            episode_states = []
            noises = dict()

            for key, param in params.items():

                if factorized_noise and param.ndim == 2:
                    i, j = param.shape

                    rows = torch.randn((pop_size_with_baseline, i, 1), device = device)
                    cols = torch.randn((pop_size_with_baseline, 1, j), device = device)
                    rows, cols = tuple(t.sign() * t.abs().sqrt() for t in (rows, cols))

                    noises_for_param = rows * cols
                else:
                    noises_for_param = torch.randn((pop_size_with_baseline, *param.shape), device = device)

                noises_for_param[0].zero_() # first is for baseline

                noises[key] = noises_for_param * noise_std_dev

            # maybe shard the interaction with environments for the individual noise perturbations

            for gene_noise_index in tqdm(self.rollouts_for_machine.tolist(), desc = 'noise index', position = 1, leave = False):

                gene_index, noise_index = gene_noise_index

                # prepare the latent gene, if needed

                if self.actor_accepts_latents:
                    latent_gene = self.gene_pool[gene_index]

                # prepare the mutation

                noise = {key: noises_for_param[noise_index] for key, noises_for_param in noises.items()}

                for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                    param_with_noise = {name: Parameter(param + noise[name] * sign) for name, param in params.items()}

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

                            if self.actor_accepts_latents:
                                kwargs.update(latent = latent_gene)

                            actor_out = functional_call(actor, param_with_noise, state, kwargs = kwargs)

                            # take care of recurrent network
                            # the nicest thing about ES is learning recurrence / memory without much hassle (in fact, can be non-differentiable)

                            if isinstance(actor_out, tuple):
                                action_logits, *actor_rest_out = actor_out
                            else:
                                action_logits = actor_out
                                actor_rest_out = []

                            if is_recurrent_actor:
                                assert len(actor_rest_out) > 0
                                mem, *_ = actor_rest_out

                            # sample

                            action = gumbel_sample(action_logits)
                            action = action.item()

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
                sel_gene_indices = self.gene_pool.evolve_with_cross_over(fitnesses)

                reward_stats = reward_stats[sel_gene_indices]

            # update based on eq (3) and (4) in the paper
            # their contribution is basically to use reward deltas (for a given noise and its negative sign) for sorting for the 'elite' directions

            # g - latent / gene, n - noise / mutation, s - sign, e - episode

            reward_std = reward_stats.std()

            reward_mean = reduce(reward_stats, 'g n s e -> n s', 'mean')

            baseline_mean, reward_mean = reward_mean[0].mean(), reward_mean[1:]

            reward_deltas = reward_mean[:, 0] - reward_mean[:, 1]

            # mask out any noise candidates whose max reward mean is greater than baseline

            accept_mask = torch.amax(reward_mean, dim = -1) > baseline_mean

            reward_deltas = reward_deltas[accept_mask]

            if reward_deltas.numel() < self.threshold_accept_learning_cycle:
                continue

            num_accepted = accept_mask.sum().item()

            # get the top performing noise indices

            k = min(num_selected, reward_deltas.numel() // 2)

            ranked_reward_deltas, ranked_reward_indices = reward_deltas.abs().topk(k, dim = 0)

            # get the weights for the weighted sum of the topk noise according to eq (3)

            weights = ranked_reward_deltas / reward_std.clamp(min = 1e-3)

            # multiply by sign

            weights *= torch.sign(reward_deltas[ranked_reward_indices]
    )
            # update the param one by one

            for param, noise in zip(params.values(), noises.values()):

                # add the best "elite" noise directions weighted by eq (3)

                best_noises = noise[1:][accept_mask][ranked_reward_indices]

                update = einsum(best_noises, weights, 'n ..., n -> ...')

                param.grad = -update

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

            # progress bar

            learning_updates.set_description(join([
                f'rewards: {baseline_mean.mean().item():.2f}',
                f'best: {reward_mean.amax().item():.2f}',
                f'best delta: {ranked_reward_deltas.amax().item():.2f}',
                f'accepted: {num_accepted} / {noise_pop_size}'
            ], ' | '))

            # log to experiment tracker

            self.log(
                rewards = baseline_mean.mean().item()
            )
