## Blackbox Gradient Sensing

Implementation and explorations into Blackbox Gradient Sensing (BGS), an evolutionary strategies approach proposed in a [Google Deepmind paper](https://arxiv.org/abs/2207.06572) for Table Tennis

Note: This paper is from 2022, and PPO is now being used for sim2real for humanoid robots (contradicting the author). However, this is the only work that I know of that successfully deployed a policy trained with ES, so worth putting out there, even if it is not quite there yet.

Will also incorporate the latent population variant used in [EPO](https://github.com/lucidrains/evolutionary-policy-optimization). Of all the things going on in evolutionary field, I believe crossover may be one of the most important. This may be the ultimate [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

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
        return np.random.randn(5), np.random.randn(1), False # state, reward, done

sim = Sim()

# instantiate BlackboxGradientSensing with the Actor (with right number of actions), and then forward your environment for the actor to learn from it
# you can also supply your own Actor, which simply receives a state tensor and outputs action logits

from torch import nn
from blackbox_gradient_sensing import BlackboxGradientSensing

actor = nn.Linear(5, 2) # contrived network from state of 5 dimension to two actions

bgs = BlackboxGradientSensing(
    actor = actor,
    noise_pop_size = 10,      # number of noise perturbations
    num_selected = 2,         # topk noise selected for update
    num_rollout_repeats = 1   # how many times to redo environment rollout, per noise
)

bgs(sim, 100) # pass the simulation environment in - say for 100 interactions with env

# after much training, save your learned policy (and optional state normalization) for finetuning on real env

bgs.save('./actor-and-state-norm.pt')
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

```bibtex
@misc{Rubin2024,
    author  = {Ohad Rubin},
    url     = {https://medium.com/@ohadrubin/exploring-weight-decay-in-layer-normalization-challenges-and-a-reparameterization-solution-ad4d12c24950}
}
```

```bibtex
@inproceedings{Wang2025EvolutionaryPO,
    title   = {Evolutionary Policy Optimization},
    author  = {Jianren Wang and Yifan Su and Abhinav Gupta and Deepak Pathak},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:277313729}
}
```

```bibtex
@inproceedings{Kumar2025QuestioningRO,
    title   = {Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis},
    author  = {Akarsh Kumar and Jeff Clune and Joel Lehman and Kenneth O. Stanley},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:278740993}
}
```
