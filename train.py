from __future__ import annotations

import fire
from shutil import rmtree
from math import ceil
from random import randrange

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.func import functional_call
torch.set_float32_matmul_precision('high')

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

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(t, temp = 1.):
    is_greedy = temp <= 0.

    if not is_greedy:
        t = (t / temp) + gumbel_noise(t)

    return t.argmax(dim = -1)

# networks

class LinearAttention(Module):
    """ small linear attention module for memory """

    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.to_forget = nn.Linear(dim, 1, bias = False)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

    def forward(self, x, past_mem):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k = map(F.silu, (q, k))

        new_mem = einsum(k, v, '... d, ... e -> ... d e')

        forget = self.to_forget(x).sigmoid()

        mem = past_mem.lerp(new_mem, forget)

        retrieved = einsum(q, mem, '... d, ... d e -> ... e')
        return retrieved, mem

class Actor(Module):
    def __init__(
        self,
        state_dim,
        hidden_dim,
        num_actions,
        depth = 2,
        expansion_factor = 2.
    ):
        super().__init__()

        self.proj_in = nn.Linear(state_dim, hidden_dim)

        dim_inner = int(hidden_dim * expansion_factor)

        # layers

        layers = []

        for ind in range(depth):

            layer = nn.Sequential(
                nn.RMSNorm(hidden_dim),
                nn.Linear(hidden_dim, dim_inner),
                nn.ReLU(),
                nn.Linear(dim_inner, hidden_dim),
            )

            layers.append(layer)

        # small memory layer

        self.mem = LinearAttention(hidden_dim)

        # final layer out

        self.layers = ModuleList(layers)

        self.final_norm = nn.RMSNorm(hidden_dim)

        self.to_logits = nn.Linear(hidden_dim, num_actions)

        self.register_buffer('init_mem', torch.zeros(hidden_dim, hidden_dim))

    @torch.compile
    def forward(
        self,
        x,
        past_mem
    ):

        x = self.proj_in(x)

        for layer in self.layers:
            x = layer(x) + x

        retrieved, next_mem = self.mem(x, past_mem)

        x = x + retrieved

        embed = self.final_norm(x)

        logits = self.to_logits(embed)
        return logits, next_mem

# main

def main(
    env_name = 'LunarLander-v3',
    total_learning_updates = 1000,
    noise_pop_size = 40,
    topk_elites = 8,
    num_rollout_repeats = 2,
    learning_rate = 1e-3,
    weight_decay_strength = 1e-2,
    max_timesteps = 250,
    actor_hidden_dim = 32,
    seed = None,
    render = True,
    clear_videos = True,
    video_folder = './lunar-bgs-recording',
    min_eps_before_update = 500
):
    assert topk_elites < noise_pop_size

    env = gym.make(env_name, render_mode = 'rgb_array')

    if render:
        if clear_videos:
            rmtree(video_folder, ignore_errors = True)

        den = noise_pop_size * 2 * num_rollout_repeats
        total_eps_before_update = ceil(min_eps_before_update / den) * den

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

        reward_stats = torch.zeros((noise_pop_size, 2, num_rollout_repeats)).to(device)

        episode_seed = randrange(int(1e7))

        # create noises upfront

        noises = dict()

        for key, param in params.items():
            noises_for_param = torch.randn((noise_pop_size, *param.shape), device = device)

            noises_for_param, ps = pack([noises_for_param], 'n *')
            nn.init.orthogonal_(noises_for_param)
            noises_for_param, = unpack(noises_for_param, ps, 'n *')

            noises[key] = noises_for_param

        for noise_index in tqdm(range(noise_pop_size), desc = 'noise index', position = 1, leave = False):

            noise = {key: noises_for_param[noise_index] for key, noises_for_param in noises.items()}

            for sign_index, sign in tqdm(enumerate((1, -1)), desc = 'sign', position = 2, leave = False):

                param_with_noise = {name: (noise[name] * sign + param) for name, param in params.items()}

                for repeat_index in tqdm(range(num_rollout_repeats), desc = 'rollout repeat', position = 3, leave = False):

                    state, _ = env.reset(seed = episode_seed)

                    total_reward = 0.

                    mem = actor.init_mem

                    for timestep in range(max_timesteps):

                        with torch.inference_mode():
                            state = torch.from_numpy(state).to(device)
                            action_logits, mem = functional_call(actor, param_with_noise, (state, mem))

                        action = gumbel_sample(action_logits)
                        action = action.item()

                        next_state, reward, terminated, truncated, _ = env.step(action)

                        total_reward += float(reward)

                        done = terminated or truncated

                        if done:
                            break

                        state = next_state
                
                    reward_stats[noise_index, sign_index, repeat_index] = total_reward

        # update based on eq (3) and (4) in the paper
        # their contribution is basically to use reward deltas (for a given noise and its negative sign) for sorting for the 'elite' directions

        # n - noise, s - sign, e - episode

        reward_std = reward_stats.std()

        reward_mean = reduce(reward_stats, 'n s e -> n s', 'mean')

        reward_deltas = reward_mean[:, 0] - reward_mean[:, 1]

        # get the topk elite indices

        ranked_reward_deltas, ranked_reward_indices = reward_deltas.abs().topk(topk_elites, dim = 0)

        # get the weights for the weighted sum of the topk noise according to eq (3)

        weights = ranked_reward_deltas / reward_std.clamp(min = 1e-3) * learning_rate

        # modulate the weights by sign and whether the reward for pos/neg noise is greater than baseline or not

        delta_signs = reward_deltas[ranked_reward_indices].sign() # since using absolute value

        weights *= delta_signs

        # update the param one by one

        for param, noise in zip(params.values(), noises.values()):

            best_noises = noise[ranked_reward_indices]

            update = einsum(best_noises, weights, 'n ..., n -> ...')

            param.data.add_(update)

            # weight decay

            param.norm(p = 1).backward()
            param.data.sub_(param.grad * learning_rate * weight_decay_strength)
            param.grad = None

        learning_updates_pbar.set_description(f'best: {reward_mean.amax().item():.2f} | best delta: {ranked_reward_deltas.amax().item():.2f}')

if __name__ == '__main__':
    fire.Fire(main)
