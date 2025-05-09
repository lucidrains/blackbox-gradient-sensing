from __future__ import annotations

import fire
from shutil import rmtree
from math import ceil
from copy import deepcopy
from random import randrange
from functools import partial

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.func import functional_call
import torch.distributed as dist
torch.set_float32_matmul_precision('high')

from einops import reduce, rearrange, reduce, einsum, pack, unpack

from torch.nn.utils.parametrizations import weight_norm

from adam_atan2_pytorch import AdoptAtan2

from tqdm import tqdm as orig_tqdm

import gymnasium as gym

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
        assert state.shape[-1] == self.dim, f'expected feature dimension of {self.dim} but received {x.shape[-1]}'

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
        state_dim,
        hidden_dim,
        num_actions,
    ):
        super().__init__()
        self.mem_norm = nn.RMSNorm(hidden_dim)

        self.proj_in = nn.Linear(state_dim, hidden_dim + 1, bias = False)
        self.proj_in = weight_norm(self.proj_in, name = 'weight', dim = None)

        self.to_embed = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.to_embed = weight_norm(self.to_embed, name = 'weight', dim = None)

        self.to_logits = nn.Linear(hidden_dim, num_actions, bias = False)
        self.to_logits = weight_norm(self.to_logits, name = 'weight', dim = None)

        self.norm_weights_()

        self.register_buffer('init_mem', torch.zeros(hidden_dim))

    def norm_weights_(self):
        for param in self.parameters():
            if not isinstance(param, nn.Linear):
                continue

            param.parametrization.weight.original.copy_(param.weight)

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

# main

@torch.inference_mode()
def main(
    env_name = 'LunarLander-v3',
    total_learning_updates = 1000,
    noise_pop_size = 40,
    noise_std_dev = 0.1, # Appendix F in paper, appears to be constant for sim and real
    factorized_noise = True,
    num_elites = 8,
    num_rollout_repeats = 3,
    learning_rate = 8e-2,
    weight_decay = 1e-4,
    cautious_factor = 0.1,
    betas = (0.9, 0.95),
    max_timesteps = 400,
    actor_hidden_dim = 32,
    seed = None,
    render = True,
    clear_videos = True,
    video_folder = './lunar-bgs-recording',
    min_eps_before_update = 500
):
    accelerator = Accelerator()

    is_distributed, world_size, rank, is_main, device = (
        accelerator.use_distributed,
        accelerator.num_processes,
        accelerator.process_index,
        accelerator.is_main_process,
        accelerator.device
    )

    tqdm = partial(orig_tqdm, disable = not is_main)

    assert num_elites < noise_pop_size

    env = gym.make(env_name, render_mode = 'rgb_array')

    if render and is_main:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        den = (noise_pop_size + 1) * 2 * num_rollout_repeats
        total_eps_before_update = ceil(min_eps_before_update / den) * den

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, total_eps_before_update),
            disable_logger = True
        )

    accelerator.wait_for_everyone()

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if exists(seed):
        torch.manual_seed(seed)

    # actor

    actor = Actor(
        state_dim,
        actor_hidden_dim,
        num_actions = num_actions
    ).to(device)

    for param in actor.parameters():
        accelerator.reduce(param.data)

    # params

    params = dict(actor.named_parameters())

    # optim

    optim = AdoptAtan2(actor.parameters(), lr = learning_rate, betas = betas, cautious_factor = cautious_factor)

    # state norm

    state_norm = StateNorm(state_dim).to(device)

    learning_updates_pbar = tqdm(range(total_learning_updates), position = 0)

    for _ in learning_updates_pbar:

        # synchronize a global seed

        if is_distributed:
            seed = accelerator.reduce(tensor(randrange(int(1e7)), device = device))
            torch.manual_seed(seed.item())

        # keep track of the rewards received per noise and its negative

        reward_stats = torch.zeros((noise_pop_size + 1, 2, num_rollout_repeats)).to(device)

        episode_seed = randrange(int(1e7))

        # create noises upfront

        episode_states = []
        noises = dict()

        for key, param in params.items():

            if factorized_noise and param.ndim == 2:
                i, j = param.shape

                rows = torch.randn((noise_pop_size + 1, i, 1), device = device)
                cols = torch.randn((noise_pop_size + 1, 1, j), device = device)
                rows, cols = tuple(t.sign() * t.abs().sqrt() for t in (rows, cols))

                noises_for_param = rows * cols
            else:
                noises_for_param = torch.randn((noise_pop_size + 1, *param.shape), device = device)

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

                    state, _ = env.reset(seed = episode_seed)

                    episode_states.clear()

                    total_reward = 0.

                    mem = actor.init_mem

                    for timestep in range(max_timesteps):

                        state = torch.from_numpy(state).to(device)

                        episode_states.append(state)

                        state_norm.eval()
                        normed_state = state_norm(state)

                        action_logits, mem = functional_call(actor, param_with_noise, (normed_state, mem))

                        action = gumbel_sample(action_logits)
                        action = action.item()

                        next_state, reward, terminated, truncated, _ = env.step(action)

                        total_reward += float(reward)

                        done = terminated or truncated

                        if done:
                            break

                        state = next_state
                
                    reward_stats[noise_index, sign_index, repeat_index] = total_reward

        # maybe synchronize reward stats, as well as min episode length for updating state norm

        if is_distributed:
            reward_stats = accelerator.reduce(reward_stats)

            episode_state_len = tensor(len(episode_states), device = device)

            min_episode_state_len = accelerator.gather(episode_state_len).amin().item()

            episode_states = episode_states[:min_episode_state_len]

        # update state norm with one episode worth (as it is repeated)

        state_norm.train()

        for state in episode_states:
            state_norm(state)

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

        k = min(num_elites, reward_deltas.numel() // 2)

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

            if isinstance(param, nn.RMSNorm):
                param.data.gamma.lerp_(torch.ones_like(param.gamma), weight_decay)

        optim.step()
        optim.zero_grad()

        actor.norm_weights_()

        learning_updates_pbar.set_description(join([
            f'best: {reward_mean.amax().item():.2f}',
            f'best delta: {ranked_reward_deltas.amax().item():.2f}',
            f'accepted: {num_accepted} / {noise_pop_size}'
        ], ' | '))

if __name__ == '__main__':
    fire.Fire(main)
