from __future__ import annotations

from copy import deepcopy
from random import randrange
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.optim import Adam
from torch.func import functional_call

import torch.distributed as dist
torch.set_float32_matmul_precision('high')

from torch.nn.utils.parametrizations import weight_norm

from einops import reduce, rearrange, einsum, pack, unpack

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
        dim,
        eps = 1e-5,
    ):
        # equation (3) in https://arxiv.org/abs/2410.09754
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.register_buffer('step', tensor(1))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_variance', torch.ones(dim))

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
        accepts_condition = False,
        dim_condition = None
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

        self.accepts_condition = accepts_condition
        if accepts_condition:
            assert exists(dim_condition)

            self.encode_condition = nn.Sequential(
                nn.Linear(dim_condition, hidden_dim),
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
        condition = None
    ):
        assert xnor(exists(condition), self.accepts_condition)

        x = self.proj_in(x)
        x, forget = x[:-1], x[-1]

        x = F.silu(x)

        if exists(hiddens):
            past_mem = self.mem_norm(hiddens) * forget.sigmoid()
            x = x + past_mem

        if self.accepts_condition:
            x = x * self.encode_condition(condition)

        x = self.to_embed(x)
        x = F.silu(x)

        return self.to_logits(x), x

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
    ):
        super().__init__()
        assert num_genes > 2

        self.num_genes = num_genes

        assert 2 <= num_selected < num_genes, f'must select at least 2 genes for mating'

        self.num_selected = num_selected
        self.tournament_size = tournament_size

        self.genes = nn.Parameter(l2norm(torch.randn(num_genes, dim)))

    def __getitem__(self, idx):
        return l2norm(self.genes[idx])

    @torch.inference_mode()
    def evolve_with_cross_over(
        self,
        fitnesses,
        temperature = 1.
    ):
        device, num_selected = fitnesses.device, self.num_selected
        assert fitnesses.ndim == 1 and fitnesses.shape[0] == self.num_genes

        sorted_fitness, sorted_gene_ids = fitnesses.sort(dim = -1)

        selected_gene_ids = sorted_gene_ids[-num_selected:]
        selected_fitness = sorted_fitness[-num_selected:]

        selected_pool = self[selected_gene_ids]

        # tournament

        num_children = self.num_genes - num_selected

        batch_randperm = torch.randn((num_children, num_selected), device = device).argsort(dim = -1)
        tourn_ids = batch_randperm[:, :self.tournament_size]

        tourn_fitness_ids = sorted_fitness[tourn_ids]

        parent_ids = tourn_fitness_ids.topk(2, dim = -1).indices

        parents = selected_pool[parent_ids]

        # cross over

        parent1, parent2 = parents.unbind(dim = 1)

        children = parent1.lerp(parent2, (torch.randn_like(parent1) / temperature).sigmoid())

        pool = torch.cat((children, selected_pool), dim = 0)

        # next generation

        self.genes.copy_(l2norm(pool))

# main class

class BlackboxGradientSensing(Module):

    def __init__(
        self,
        actor: Module,
        *,
        accelerator: Accelerator | None = None,
        dim_state = None,
        use_state_norm = True,
        actor_is_recurrent = False,
        dim_gene = None,
        num_genes = 1,
        num_selected_genes = 1,
        actor_is_latent_conditioned = False,
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
        param_names: set[str] | None = None,
        show_progress = True,
        optim_kwargs: dict = dict(),
        optim_step_post_hook: Callable | None = None,
        post_noise_added_hook: Callable | None = None,
        accelerate_kwargs: dict = dict(),
        latent_gene_kwargs: dict = dict(),
        cpu = False,
        torch_compile_actor = True
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
                accelerate_kwargs.update(cpu = True)

            accelerator = Accelerator(**accelerate_kwargs)

        device = accelerator.device
        self.accelerator = accelerator

        # net

        self.actor = actor.to(device)

        self.torch_compile_actor = torch_compile_actor

        self.actor_is_recurrent = actor_is_recurrent # if set to True, actor must pass out the memory on forward on the second position, then receive it as a kwarg of `hiddens`

        named_params = dict(actor.named_parameters())
        self.param_names = default(param_names, set(named_params.keys()))

        # gene pool, another axis for scaling and bitter lesson

        has_gene_pool = num_genes > 1

        self.has_gene_pool = has_gene_pool

        if has_gene_pool:
            self.gene_pool = LatentGenePool(dim_gene, num_genes, **latent_gene_kwargs)

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

        self.use_state_norm = use_state_norm

        if use_state_norm:
            assert exists(dim_state), f'if using state normalization, must pass in `dim_state`'
            self.state_norm = StateNorm(dim_state)
            self.state_norm.to(device)

        # progress bar

        self.show_progress = show_progress

        # number of interactions with environment for learning

        self.num_env_interactions = num_env_interactions

        # calculate num of episodes per learning cycle for this machine

        world_size, rank = accelerator.num_processes, accelerator.process_index

        noise_indices = torch.arange(noise_pop_size + 1)
        noises_for_machine = noise_indices.chunk(world_size)[rank]
        self.register_buffer('noises_for_machine', noises_for_machine, persistent = False)
    
        # expose a few computed variables

        self.num_episodes_per_learning_cycle = self.noises_for_machine.shape[0]

        self.is_main = rank == 0

        # keep track of number of steps

        self.register_buffer('step', tensor(0))

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
            state_norm = self.state_norm.state_dict() if self.use_state_norm else None
        )

        torch.save(pkg, str(path))

    def load(self, path):
        path = Path(path)

        assert path.exists()

        pkg = torch.load(str(path), weights_only = True)

        self.actor.load_state_dict(pkg['actor'])

        if self.use_state_norm:
            assert 'state_norm' in pkg
            self.state_norm.load_state_dict(pkg['state_norm'])

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

        acc, optim, actor = self.accelerator, self.optim, self.actor

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
                rand_int = torch.randint(0, int(1e7), (), device = device)
                seed = acc.reduce(rand_int)
                torch.manual_seed(seed.item())

            # keep track of the rewards received per noise and its negative

            pop_size_with_baseline = noise_pop_size + 1
            reward_stats = torch.zeros((pop_size_with_baseline, 2, num_rollout_repeats)).to(device)

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

            for noise_index in tqdm(self.noises_for_machine.tolist(), desc = 'noise index', position = 1, leave = False):

                noise = {key: noises_for_param[noise_index] for key, noises_for_param in noises.items()}

                for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                    param_with_noise = {name: (noise[name] * sign + param) for name, param in params.items()}

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
                    
                        reward_stats[noise_index, sign_index, repeat_index] = total_reward

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

            # n - noise, s - sign, e - episode

            reward_std = reward_stats.std()

            reward_mean = reduce(reward_stats, 'n s e -> n s', 'mean')

            baseline_mean, reward_mean = reward_mean[0].mean(), reward_mean[1:]

            reward_deltas = reward_mean[:, 0] - reward_mean[:, 1]

            # mask out any noise candidates whose max reward mean is greater than baseline

            accept_mask = torch.amax(reward_mean, dim = -1) > baseline_mean

            reward_deltas = reward_deltas[accept_mask]

            if reward_deltas.numel() < 2:
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
