## Blackbox Gradient Sensing

Explorations into Blackbox Gradient Sensing (BGS), an evolutionary strategies approach proposed in a [Google Deepmind paper](https://arxiv.org/abs/2207.06572) for Table Tennis

Note: This paper is from 2022, and PPO is now being used for sim2real for humanoid robots (contradicting the author). However, this is the only work that I know of that successfully deployed a policy trained with ES, so worth putting out there, even if it is not quite there yet.

## Usage

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
