import pytest

from torch import nn
from blackbox_gradient_sensing.bgs import BlackboxGradientSensing, Actor

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
def test_bgs(
    factorized_noise,
    use_custom_actor,
    use_state_norm,
    actor_is_recurrent
):

    sim = Sim()

    actor = Actor(dim_state = 5, num_actions = 2) # actor with weight norm

    # test custom actor

    if use_custom_actor:
        actor = nn.Linear(5, 2)

        if actor_is_recurrent:
            pytest.skip()

    # maybe state norm

    state_norm = None

    if use_state_norm:
        state_norm = dict(dim_state = 5)

    # main evo strat orchestrator

    bgs = BlackboxGradientSensing(
        actor = actor,
        dim_gene = 16,
        num_genes = 3,
        num_selected_genes = 2,
        noise_pop_size = 10,      # number of noise perturbations
        num_selected = 2,         # topk noise selected for update
        num_rollout_repeats = 1,   # how many times to redo environment rollout, per noise
        torch_compile_actor = False,
        latent_gene_kwargs = dict(
            num_selected = 2,
            tournament_size = 2 
        ),
        cpu = True,
        factorized_noise = factorized_noise,
        state_norm = state_norm,
        actor_is_recurrent = actor_is_recurrent
    )

    bgs(sim, 2) # pass the simulation environment in - say for 100 interactions with env

    # after much training, save your learned policy (and optional state normalization) for finetuning on real env

    bgs.save('./actor-and-state-norm.pt', overwrite = True)

    bgs.load('./actor-and-state-norm.pt')
