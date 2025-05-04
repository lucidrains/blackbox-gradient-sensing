from __future__ import annotations

import fire
from shutil import rmtree
from random import randrange

import torch
from torch import nn, tensor, cat, stack
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.distributions import Categorical
from torch.func import functional_call

from einops import reduce, rearrange, einsum

from tqdm import tqdm

import gymnasium as gym

# constants

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# mlp

class MLP(Module):

    def __init__(
        self,
        dim,
        dim_hidden = None,
        depth = 2,
        expansion_factor = 2,
    ):
        super().__init__()
        dim_hidden = default(dim_hidden, dim * expansion_factor)

        layers = []

        self.proj_in = nn.Linear(dim, dim_hidden)

        dim_inner = dim_hidden * expansion_factor

        # layers

        for ind in range(depth):

            layer = nn.Sequential(
                nn.RMSNorm(dim_hidden),
                nn.Linear(dim_hidden, dim_inner),
                nn.ReLU(),
                nn.Linear(dim_inner, dim_hidden),
            )

            layers.append(layer)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(dim_hidden)

    def forward(self, x):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        out = self.final_norm(x)

        if no_batch:
            out = rearrange(out, '1 ... -> ...')

        return out

# networks

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        mlp_depth = 1,
    ):
        super().__init__()

        self.net = MLP(
            state_dim,
            dim_hidden = hidden_dim,
            depth = mlp_depth,
        )

        self.action_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        hidden = self.net(x)
        return self.action_head(hidden).softmax(dim = -1)

# main

def main(
    env_name = 'LunarLander-v3',
    total_learning_updates = 1000,
    noise_pop_size = 50,
    topk_elites = 4,
    num_rollouts_before_update = 10,
    learning_rate = 1e-1,
    max_timesteps = 500,
    actor_hidden_dim = 64,
    seed = None,
    render = True,
    clear_videos = True,
    video_folder = './lunar-bgs-recording',
):
    assert topk_elites < noise_pop_size

    env = gym.make(env_name, render_mode = 'rgb_array')

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        total_eps_before_update = noise_pop_size * 2 * num_rollouts_before_update

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = video_folder,
            name_prefix = 'lunar-video',
            episode_trigger = lambda eps_num: divisible_by(eps_num, total_eps_before_update),
            disable_logger = True
        )

    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    if exists(seed):
        torch.manual_seed(seed)

    # actor

    actor = Actor(
        state_dim,
        actor_hidden_dim,
        num_actions = num_actions
    )

    actor = actor.to(device)

    params = dict(actor.named_parameters())

    learning_updates_pbar = tqdm(range(total_learning_updates), position = 0)

    for _ in learning_updates_pbar:

        # keep track of the rewards received per noise and its negative

        noises = []
        reward_stats = torch.zeros((noise_pop_size, 2, num_rollouts_before_update)).to(device)

        for noise_index in tqdm(range(noise_pop_size), desc = 'noise index', position = 1, leave = False):

            noise = {name: torch.randn_like(param) for name, param in params.items()}
            episode_seeds = torch.randint(0, int(1e7), (num_rollouts_before_update,))

            for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                param_with_noise = {name: (noise[name] * sign + params[name]) for name in params}

                for episode_index in tqdm(range(num_rollouts_before_update), desc = 'episodes', position = 3, leave = False):

                    state, _ = env.reset(seed = episode_seeds[episode_index].item())
                    state = torch.from_numpy(state).to(device)

                    total_reward = 0.

                    for timestep in range(max_timesteps):

                        with torch.inference_mode():
                            action_probs = functional_call(actor, param_with_noise, state)

                        dist = Categorical(action_probs)
                        action = dist.sample()
                        action = action.item()

                        next_state, reward, terminated, truncated, _ = env.step(action)

                        total_reward += float(reward)

                        done = terminated or truncated

                        if done:
                            break
                
                    reward_stats[noise_index, sign_index, episode_index] = total_reward

            noises.append(noise)

        # stack all the noises

        param_names = params.keys()
        stacked_noises = [torch.stack(tensors) for tensors in zip(*[noise.values() for noise in noises])]

        stacked_noises = dict(zip(param_names, stacked_noises))

        # update based on eq (3) and (4) in the paper
        # their contribution is basically to use reward deltas (for a given noise and its negative sign) for ranking the top perturbations

        # n - noise, s - sign, e - episode

        reward_mean = reduce(reward_stats, 'n s e -> n s', 'mean')

        reward_deltas = reward_mean[:, 0] - reward_mean[:, 1]

        # get the topk elite indices

        ranked_reward_deltas, ranked_reward_indices = reward_deltas.topk(topk_elites, dim = 0)

        # get the weights for the weighted sum of the topk noise according to eq (3)

        reward_std = reward_mean[ranked_reward_indices].std()
        weights = (ranked_reward_deltas / reward_std.clamp(min = 1e-3)) * learning_rate

        # update the param one by one

        for param, stacked_noise in zip(params.values(), stacked_noises.values()):
            elite_noises = stacked_noise[ranked_reward_indices]

            update = einsum(elite_noises, weights, 'n ..., n -> ...')

            param.data.add_(update)

        learning_updates_pbar.set_description(f'best: {reward_mean.amax().item():.2f} | best delta: {ranked_reward_deltas.amax().item():.2f}')

if __name__ == '__main__':
    fire.Fire(main)
