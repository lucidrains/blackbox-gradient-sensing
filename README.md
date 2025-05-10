## Blackbox Gradient Sensing (wip)

Explorations into Blackbox Gradient Sensing (BGS), an evolutionary strategies approach proposed in a [Google Deepmind paper](https://arxiv.org/abs/2207.06572) for Table Tennis

Note: This paper is from 2022, and PPO is now being used for sim2real for humanoid robots (contradicting the author). However, this is the only work that I know of that successfully deployed a policy trained with ES, so worth putting out there, even if it is not quite there yet.

Will also improvise in a population based variant. Of all the things going on in evolutionary field, I believe crossover may be one of the most important.

## Install

```bash
$ pip install blackbox-gradient-sensing
```

## Usage

```python
# mock env

import numpy as np

class Sim:
    def reset(self, seed = None):
        return np.random.randn(5) # state

    def step(self, actions):
        return np.random.randn(5), np.random.randn(1), False, False # state, reward, terminated, truncated

sim = Sim()

# instantiate BlackboxGradientSensing with the Actor (with right number of actions), and then forward your environment for the actor to learn from it
# you can also supply your own Actor, which simply receives a state tensor and outputs action logits

from bgs import BlackboxGradientSensing, Actor

actor = Actor(
    dim_state = 5,
    num_actions = 2
)

bgs = BlackboxGradientSensing(
    actor,
    dim_state = 5,
    noise_pop_size = 10,
    num_rollout_repeats = 1
)

bgs(sim, 1000) # pass the simulation environment in - say for 1000 interactions with env

# after much training, save your learned policy for finetuning on real env

actor.save('./sim-trained-actor.pt')
```

## Example

```python
$ pip install -r requirements.txt  # or `uv pip install`, to keep up with the times
```

You may need to run the following if you see an error related to `swig`

```bash
$ apt install swig -y
```

Then

```bash
$ python train.py
```

## Distributed using 🤗 accelerate

First

```bash
$ accelerate config
```

Then

```bash
$ accelerate launch train.py
```

## Citations

```bibtex
@inproceedings{Abeyruwan2022iSim2RealRL,
    title   = {i-Sim2Real: Reinforcement Learning of Robotic Policies in Tight Human-Robot Interaction Loops},
    author  = {Saminda Abeyruwan and Laura Graesser and David B. D'Ambrosio and Avi Singh and Anish Shankar and Alex Bewley and Deepali Jain and Krzysztof Choromanski and Pannag R. Sanketi},
    booktitle = {Conference on Robot Learning},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:250526228}
}
```

```bibtex
@article{Lee2024SimBaSB,
    title   = {SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning},
    author  = {Hojoon Lee and Dongyoon Hwang and Donghu Kim and Hyunseung Kim and Jun Jet Tai and Kaushik Subramanian and Peter R. Wurman and Jaegul Choo and Peter Stone and Takuma Seno},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.09754},
    url     = {https://api.semanticscholar.org/CorpusID:273346233}
}
```

```bibtex
@article{Palenicek2025ScalingOR,
    title   = {Scaling Off-Policy Reinforcement Learning with Batch and Weight Normalization},
    author  = {Daniel Palenicek and Florian Vogt and Jan Peters},
    journal = {ArXiv},
    year    = {2025},
    volume  = {abs/2502.07523},
    url     = {https://api.semanticscholar.org/CorpusID:276258971}
}
```
