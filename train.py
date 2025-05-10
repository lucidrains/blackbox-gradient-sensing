from blackbox_gradient_sensing import BlackboxGradientSensing, Actor

# hyperparams

num_noises = 40  # number of noise perturbations, from which top is chosen for a weighted update - in paper this was 200 for sim, 3 for real
num_repeats = 3  # number of repeats (j in eq) - in paper they did ~10 for sim, then 3 for real

# env related, for example using gymansium

from math import ceil
from shutil import rmtree

import gymnasium as gym

# lunar lander env with periodic recording of the learning progress to ./recording

sim = gym.make('LunarLander-v3', render_mode = 'rgb_array')

video_folder = './recording'
rmtree(video_folder, ignore_errors = True)

den = (num_noises + 1) * 2 * num_repeats
total_eps_before_update = ceil(500 / den) * den

sim = gym.wrappers.RecordVideo(
    env = sim,
    video_folder = video_folder,
    name_prefix = 'lunar-lander',
    episode_trigger = lambda eps_num: (eps_num % total_eps_before_update == 0),
    disable_logger = True
)

dim_state = sim.observation_space.shape[0]

# instantiate BlackboxGradientSensing with the Actor (with right number of actions), and then forward your environment for the actor to learn from it
# you can also supply your own Actor, which simply receives a state tensor and outputs action logits

actor = Actor(
    dim_state = dim_state,
    num_actions = sim.action_space.n
)

bgs = BlackboxGradientSensing(
    actor,
    dim_state = dim_state,
    noise_pop_size = num_noises,
    num_rollout_repeats = num_repeats,
    actor_is_recurrent = True
)

bgs(sim, 1000, torch_compile = True) # pass the simulation environment in - say for 1000 interactions with env

# after much training, finetune on real env

actor.save('./sim-trained-actor.pt')
