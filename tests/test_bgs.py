import pytest

import torch
from torch import nn

from blackbox_gradient_sensing.bgs import (
    BlackboxGradientSensing,
    Actor,
    LatentGenePool
)

# mock env

import numpy as np

class Sim:
    def reset(self, seed = None):
        return np.random.randn(5) # state

    def step(self, actions):
        return np.random.randn(5), np.random.randn(1), False # state, reward, done

# test BGS

@pytest.mark.parametrize('factorized_noise', (True, False))
@pytest.mark.parametrize('use_custom_actor', (True, False))
@pytest.mark.parametrize('use_state_norm', (True, False))
@pytest.mark.parametrize('actor_is_recurrent', (True, False))
@pytest.mark.parametrize('use_genetic_algorithm', (True, False))
@pytest.mark.parametrize('use_ema', (True, False))
def test_bgs(
    factorized_noise,
    use_custom_actor,
    use_state_norm,
    actor_is_recurrent,
    use_genetic_algorithm,
    use_ema
):

    sim = Sim()

    actor = Actor(
        dim_state = 5,
        num_actions = 2,
        dim_latent = 32,
        accepts_latent = use_genetic_algorithm
    ) # actor with weight norm

    # test custom actor

    if use_custom_actor:
        actor = nn.Linear(5, 2)

        if actor_is_recurrent or use_genetic_algorithm:
            pytest.skip()

    # maybe state norm

    state_norm = None

    if use_state_norm:
        state_norm = dict(dim_state = 5)

    # maybe genetic algorithm

    latent_gene_pool = None

    if use_genetic_algorithm:
        latent_gene_pool = dict(
            dim = 32,
            num_genes = 3,
            num_selected = 2,
            tournament_size = 2
        )

    # main evo strat orchestrator

    bgs = BlackboxGradientSensing(
        actor = actor,
        noise_pop_size = 2,      # number of noise perturbations
        num_selected = 1,         # topk noise selected for update
        num_rollout_repeats = 1,   # how many times to redo environment rollout, per noise
        torch_compile_actor = False,
        max_timesteps = 1,
        factorized_noise = factorized_noise,
        state_norm = state_norm,
        actor_is_recurrent = actor_is_recurrent,
        latent_gene_pool = latent_gene_pool,
        cpu = True,
        use_ema = use_ema
    )

    bgs(sim, 2) # pass the simulation environment in - say for 100 interactions with env

    # after much training, save your learned policy (and optional state normalization) for finetuning on real env

    bgs.save('./actor-and-state-norm.pt', overwrite = True)

    bgs.load('./actor-and-state-norm.pt')

# test genetic algorithm

def test_cross_over():
    gene_pool = LatentGenePool(dim = 32, num_genes = 3, num_selected = 2, tournament_size = 2)

    fitness = torch.randn(3)

    gene_pool.evolve_with_cross_over(fitness)
