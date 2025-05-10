from __future__ import annotations

from shutil import rmtree
from math import ceil
from copy import deepcopy
from random import randrange
from functools import partial

import numpy as np

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.func import functional_call
import torch.distributed as dist
torch.set_float32_matmul_precision('high')

from torch.nn.utils.parametrizations import weight_norm

from einops import reduce, rearrange, reduce, einsum, pack, unpack

from adam_atan2_pytorch import AdoptAtan2

from tqdm import tqdm as orig_tqdm

from accelerate import Accelerator

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

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
    ):
        super().__init__()
        self.mem_norm = nn.RMSNorm(hidden_dim)

        self.proj_in = nn.Linear(dim_state, hidden_dim + 1, bias = False)
        self.proj_in = weight_norm(self.proj_in, name = 'weight', dim = None)

        self.to_embed = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.to_embed = weight_norm(self.to_embed, name = 'weight', dim = None)

        self.to_logits = nn.Linear(hidden_dim, num_actions, bias = False)
        self.to_logits = weight_norm(self.to_logits, name = 'weight', dim = None)

        self.register_buffer('init_mem', torch.zeros(hidden_dim))

    @torch.compile
    def forward(
        self,
        x,
        past_mem
    ):
        x = self.proj_in(x)
        x, forget = x[:-1], x[-1]

        x = F.silu(x)

        past_mem = self.mem_norm(past_mem) * forget.sigmoid()
        x = x + past_mem

        x = self.to_embed(x)
        x = F.silu(x)

        return self.to_logits(x), x

# main class

class BlackboxGradientSensing(Module):

    def __init__(
        self,
        actor: Module,
        *,
        accelerator: Accelerator | None = None,
        dim_state = None,
        use_state_norm = True,
        num_env_interactions = 1000,
        noise_pop_size = 40,
        noise_std_dev = 0.1, # Appendix F in paper, appears to be constant for sim and real
        factorized_noise = True,
        num_selected = 8,    # of the population, how many of the best performing noise perturbations to accept
        num_rollout_repeats = 3,
        optim_klass = AdoptAtan2,
        learning_rate = 8e-2,
        weight_decay = 1e-4,
        betas = (0.9, 0.95),
        max_timesteps = 400,
        show_progress = True,
        optim_kwargs: dict = dict(
            cautious_factor = 0.1
        ),
        accelerate_kwargs: dict = dict()
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
            accelerator = Accelerator(**accelerate_kwargs)

        device = accelerator.device
        self.accelerator = accelerator

        # net

        self.actor = actor.to(device)

        # optim

        self.optim = optim_klass(actor.parameters(), lr = learning_rate, betas = betas)

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

    @torch.inference_mode()
    def forward(
        self,
        env,
        num_env_interactions = None,
        show_progress = None,
        seed = None,
        max_timesteps_per_interaction = 500
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

        is_distributed, world_size, rank, is_main, device = (
            acc.use_distributed,
            acc.num_processes,
            acc.process_index,
            acc.is_main_process,
            acc.device
        )

        tqdm = partial(orig_tqdm, disable = not is_main or not show_progress)

        if exists(seed):
            torch.manual_seed(seed)

        # params

        params = dict(self.actor.named_parameters())

        progress_bar = tqdm(range(num_env_interactions), position = 0)

        for _ in progress_bar:

            # synchronize a global seed

            if is_distributed:
                seed = accelerator.reduce(tensor(randrange(int(1e7)), device = device))
                torch.manual_seed(seed.item())

            # keep track of the rewards received per noise and its negative

            pop_size_with_baseline = noise_pop_size + 1
            reward_stats = torch.zeros((pop_size_with_baseline, 2, num_rollout_repeats)).to(device)

            episode_seed = randrange(int(1e7))

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

            noise_indices = torch.arange(noise_pop_size + 1, device = device)
            noises_for_machine = noise_indices.chunk(world_size)[rank].tolist()

            assert len(noises_for_machine) > 0

            for noise_index in tqdm(noises_for_machine, desc = 'noise index', position = 1, leave = False):

                noise = {key: noises_for_param[noise_index] for key, noises_for_param in noises.items()}

                for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                    param_with_noise = {name: (noise[name] * sign + param) for name, param in params.items()}

                    for repeat_index in tqdm(range(num_rollout_repeats), desc = 'rollout repeat', position = 3, leave = False):

                        state = env.reset(seed = episode_seed)

                        if isinstance(state, tuple):
                            state, *_ = state

                        episode_states.clear()

                        total_reward = 0.

                        mem = actor.init_mem

                        for timestep in range(max_timesteps_per_interaction):

                            state = from_numpy(state).to(device)

                            episode_states.append(state)

                            if self.use_state_norm:
                                self.state_norm.eval()
                                state = self.state_norm(state)

                            action_logits, mem = functional_call(actor, param_with_noise, (state, mem))

                            action = gumbel_sample(action_logits)
                            action = action.item()

                            next_state, reward, terminated, truncated, *_ = env.step(action)

                            total_reward += float(reward)

                            done = terminated or truncated

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

                # decay for rmsnorm back to identity

                if isinstance(param, (nn.RMSNorm, nn.LayerNorm)):
                    param.data.gamma.lerp_(torch.ones_like(param.gamma), weight_decay)

            optim.step()
            optim.zero_grad()

            progress_bar.set_description(join([
                f'best: {reward_mean.amax().item():.2f}',
                f'best delta: {ranked_reward_deltas.amax().item():.2f}',
                f'accepted: {num_accepted} / {noise_pop_size}'
            ], ' | '))
