from adam_atan2_pytorch import AdoptAtan2
from blackbox_gradient_sensing import BlackboxGradientSensing, Actor

# sim environment, example using gymansium

import gymnasium as gym

continuous = True

sim = gym.make(
    'LunarLander-v3',
    render_mode = 'rgb_array',
    continuous = continuous
)

dim_state = sim.observation_space.shape[0]

# hyperparams

num_noises = 100      # number of noise perturbations, from which top is chosen for a weighted update - in paper this was 200 for sim, 3 for real
num_selected = 15     # number of elite perturbations chosen
num_repeats = 4       # number of repeats (j in eq) - in paper they did ~10 for sim, then 3 for real

use_genetic_algorithm = False
dim_gene = 32
num_genes = 3
num_selected = 2
tournament_size = 2

# recording

min_eps_before_update = 1000

# instantiate BlackboxGradientSensing with the Actor (with right number of actions), and then forward your environment for the actor to learn from it
# you can also supply your own Actor, which simply receives a state tensor and outputs action logits

num_actions = sim.action_space.n if not continuous else sim.action_space.shape[0]

actor = Actor(
    dim_state = dim_state,
    num_actions = num_actions,
    continuous = continuous,
    dim_latent = dim_gene,
    accepts_latent = use_genetic_algorithm,
    sample = True,
    weight_norm_linears = True
)

bgs = BlackboxGradientSensing(
    actor,    
    noise_pop_size = num_noises,
    num_selected = num_selected,
    num_rollout_repeats = num_repeats,
    actor_is_recurrent = True,
    use_ema = True,
    optim_klass = AdoptAtan2,
    optim_step_post_hook = lambda: actor.norm_weights_(),
    torch_compile_actor = True,
    mutate_latent_genes = True,
    crossover_after_step = 100,
    crossover_every_step = 50,
    num_std_below_mean_thres_accept = -0.25,
    sample_actions_from_actor = False,
    factorized_noise = True,
    orthogonalized_noise = True,
    optim_kwargs = dict(
        cautious_factor = 0.1,
    ),
    state_norm = dict(
        dim_state = dim_state
    ),
    latent_gene_pool = dict(
        dim = dim_gene,
        num_islands = 1,
        num_genes_per_island = num_genes,
        num_selected = num_selected,
        tournament_size = tournament_size
    ) if use_genetic_algorithm else None
)

# recording logic

if bgs.is_main:
    from math import ceil
    from shutil import rmtree

    video_folder = './recording'
    rmtree(video_folder, ignore_errors = True)

    den = bgs.num_episodes_per_learning_cycle

    total_eps_before_update = ceil(min_eps_before_update / den) * den

    sim = gym.wrappers.RecordVideo(
        env = sim,
        video_folder = video_folder,
        name_prefix = 'lunar-lander',
        episode_trigger = lambda eps_num: (eps_num % total_eps_before_update == 0),
        disable_logger = True
    )

# pass the simulation environment in - say for 1000 interactions with env

bgs(sim, 10000)

# after much training, save and then finetune on real environment

bgs.save('./sim-trained-actor-and-state-norm.pt')
