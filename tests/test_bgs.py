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
def test_bgs(
    factorized_noise,
    use_custom_actor,
    use_state_norm
):

    sim = Sim()

    if use_custom_actor:
        # contrived network from state of 5 dimension to two actions
        actor = nn.Linear(5, 2)
    else:
        actor = Actor(dim_state = 5, num_actions = 2)

    # main evo strat orchestrator

    bgs = BlackboxGradientSensing(
        actor = actor,
        dim_state = 5,
        dim_gene = 16,
        num_genes = 3,
        num_selected_genes = 2,
        noise_pop_size = 10,      # number of noise perturbations
        num_selected = 2,         # topk noise selected for update
        num_rollout_repeats = 1,   # how many times to redo environment rollout, per noise
        cpu = True,
        torch_compile_actor = False,
        latent_gene_kwargs = dict(
            num_selected = 2,
            tournament_size = 2 
        ),
        factorized_noise = factorized_noise,
        use_state_norm = use_state_norm
    )

    bgs(sim, 2) # pass the simulation environment in - say for 100 interactions with env

    # after much training, save your learned policy (and optional state normalization) for finetuning on real env

    bgs.save('./actor-and-state-norm.pt', overwrite = True)

    bgs.load('./actor-and-state-norm.pt')
