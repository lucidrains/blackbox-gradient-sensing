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

from einops import reduce, rearrange, einsum, pack, unpack

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

def is_empty(t):
    return t.numel() == 0

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
    noise_pop_size = 20,
    topk_elites = 5,
    num_rollouts_before_update = 2,
    learning_rate = 1e-3,
    weight_decay_strength = 0.1,
    max_timesteps = 250,
    actor_hidden_dim = 32,
    noise_scale = 1.,
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

        total_eps_before_update = (noise_pop_size + 1) * 2 * num_rollouts_before_update

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

        reward_stats = torch.zeros((noise_pop_size + 1, 2, num_rollouts_before_update)).to(device)

        episode_seeds = torch.randint(0, int(1e7), (num_rollouts_before_update,))

        # create noises upfront

        noises = dict()

        for key, param in params.items():
            noises_for_param = torch.randn((noise_pop_size + 1, *param.shape), device = device)

            noises_for_param, ps = pack([noises_for_param], 'n *')
            nn.init.orthogonal_(noises_for_param)
            noises_for_param, = unpack(noises_for_param, ps, 'n *')

            noises_for_param[0].zero_() # first one is for baseline
            noises[key] = noises_for_param

        for noise_index in tqdm(range(noise_pop_size + 1), desc = 'noise index', position = 1, leave = False):

            noise = {key: noises_for_param[noise_index] for key, noises_for_param in noises.items()}

            for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                param_with_noise = {name: (noise[name] * sign + param) for name, param in params.items()}

                for episode_index in tqdm(range(num_rollouts_before_update), desc = 'episodes', position = 3, leave = False):

                    state, _ = env.reset(seed = episode_seeds[episode_index].item())

                    total_reward = 0.

                    for timestep in range(max_timesteps):

                        with torch.inference_mode():
                            state = torch.from_numpy(state).to(device)
                            action_probs = functional_call(actor, param_with_noise, state)

                        dist = Categorical(action_probs)
                        action = dist.sample()
                        action = action.item()

                        next_state, reward, terminated, truncated, _ = env.step(action)

                        total_reward += float(reward)

                        done = terminated or truncated

                        if done:
                            break

                        state = next_state
                
                    reward_stats[noise_index, sign_index, episode_index] = total_reward

        # remove the baseline noise

        noises = {key: noises_for_param[1:] for key, noises_for_param in noises.items()}

        # update based on eq (3) and (4) in the paper
        # their contribution is basically to use reward deltas (for a given noise and its negative sign) for ranking the top perturbations

        # n - noise, s - sign, e - episode

        noise_indices = torch.arange(noise_pop_size, device = reward_stats.device)

        baseline, reward_stats = reward_stats[0], reward_stats[1:]

        reward_std = reward_stats.std()

        reward_mean = reduce(reward_stats, 'n s e -> n s', 'mean')

        should_update = (reward_mean > torch.median(baseline)).any(dim = -1)

        noise_indices = noise_indices[should_update]

        reward_mean = reward_mean[should_update]

        if is_empty(reward_mean):
            learning_updates_pbar.set_description(f'no rewards past baseline {baseline.mean().item():.3f}')
            continue

        num_rewards_greater_than_baseline = reward_mean.shape[0]

        reward_deltas = reward_mean[:, 0] - reward_mean[:, 1]

        # get the topk elite indices

        ranked_reward_deltas, ranked_reward_indices = reward_deltas.abs().topk(min(topk_elites, num_rewards_greater_than_baseline), dim = 0)

        # get the weights for the weighted sum of the topk noise according to eq (3)

        weights = ranked_reward_deltas / reward_std.clamp(min = 1e-3) * learning_rate

        # modulate the weights by sign and whether the reward for pos/neg noise is greater than baseline or not

        delta_signs = reward_deltas[ranked_reward_indices].sign() # since using absolute value

        weights *= delta_signs

        # update the param one by one

        sel_noise_indices = noise_indices[ranked_reward_indices]

        for param, noise in zip(params.values(), noises.values()):

            best_noises = noise[sel_noise_indices]

            update = einsum(best_noises, weights, 'n ..., n -> ...')

            param.data.add_(update)

            # weight decay

            param.norm(p = 1).backward()
            param.data.sub_(param.grad * learning_rate * weight_decay_strength)
            param.grad = None

        learning_updates_pbar.set_description(f'best: {reward_mean.amax().item():.2f} | best delta: {ranked_reward_deltas.amax().item():.2f} | {num_rewards_greater_than_baseline} rewards past baseline')

if __name__ == '__main__':
    fire.Fire(main)
