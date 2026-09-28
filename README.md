# Logical Specifications for Ethical Agents

How can we state what an ethical agent must do precisely enough to check it? This project
explores **logical specifications** of ethical behaviour for learning agents: writing ethical
norms and principles in a formal language, training neural networks that should respect them,
attacking those networks with PGD, and formally verifying them with
[Vehicle](https://github.com/vehicle-lang/vehicle) and Marabou.

It has two parts, each building on one paper.

| folder | specification | builds on |
|---|---|---|
| [`moral-classification/`](moral-classification/README.md) | Moral Machine dilemmas in natural language, each cultural cluster's choice as **LTL** | Ethical Reward Machine (NeSy 2024), Exploring Verification Frameworks for Social Choice Alignment (NeSy 2025) |
| [`moral-regression/`](moral-regression/README.md) | ethical principles as **properties** of a rescue agent's network | Interpretable Moral Decision-making under Epistemic Uncertainty (NeSy 2026) |

## Natural language to LTL, across cultures: `moral-classification/`

**[Ethical Reward Machine](https://link.springer.com/chapter/10.1007/978-3-031-71167-1_10)**
(Ciupa and Belle, NeSy 2024, LNCS 14979) studies reward design with ethical constraints in
reinforcement learning. It integrates ethical constraints based on **Act Deontology** and
**Utilitarianism** into reinforcement learning through a symbolic language, in simulated driving
and search-and-rescue domains. The ethical principles change the agent's behaviour significantly
when it faces a dilemma, without increasing runtime.

**[Exploring Verification Frameworks for Social Choice Alignment](https://proceedings.mlr.press/v284/ciupa25a.html)** (Ciupa, Belle and
Komendantskaya, NeSy 2025, PMLR 284:439-446) is a position paper. It argues that ethical dilemmas
are not decided in a social vacuum: to earn the trust of all human users, an agent's alignment
with moral preferences should be verified with neurosymbolic methods. It proposes applying
formal robustness properties to social choice modelling, to check that deep neural network
classifiers form stable social preference clusters. Initial results show such models are
vulnerable to perturbations in moral-critical scenarios, suggesting a verification-training loop.

**Motivation.** An Ethical Reward Machine enforces constraints given in symbolic form, but what
people consider the right choice is expressed in natural language, and differs between cultures.
`moral-classification/` takes dilemmas from the Moral Machine experiment (Awad et al., 2018),
writes them as sentences, and labels each with the choice of each cultural cluster, expressed as
an LTL constraint on who is protected (`G ! kill_pedestrians_ahead`, ...). A classifier learns a
cluster's choices from the text and is verified for robustness to typos, following the
[ANTONIO](https://github.com/ANTONIONLP/ANTONIO) pipeline for NLP verification, so one can ask
whether a model is robustly aligned with one culture's preferences. Connecting the formulas to
Ethical Reward Machines is future work.

## Moral decisions under epistemic uncertainty: `moral-regression/`

**[Interpretable Moral Decision-making under Epistemic Uncertainty](https://www.research.ed.ac.uk/en/publications/interpretable-moral-decision-making-under-epistemic-uncertainty/)**
(Ciupa, Belle and Sewell, NeSy 2026; [full paper](https://www.pure.ed.ac.uk/ws/portalfiles/portal/660548025/CiupaEtalNESY2026InterpretableMoralDecision-making.pdf))
addresses agents that must act on incomplete information in safety-critical multi-agent systems.
There, ethical principles from consequentialist, deontological and virtue-based traditions can
prescribe conflicting actions. Purely neural reinforcement learning scales but is not
interpretable, and symbolic rule-based systems struggle to adapt under uncertainty. The paper
introduces a neurosymbolic framework combining Probabilistic Logic Programming, case-based
reasoning and policy-gradient reinforcement learning, with ethical principles encoded as
probabilistic logical rules. In grid-world search-and-rescue simulations it stays interpretable
and ethically aligned while improving under uncertainty.

**Motivation.** If ethical principles guide an agent, can we check that a trained network
actually follows them? `moral-regression/` simulates a rescue agent whose accumulated risk carries
over from room to room, trains a network to set its effort, and shows why PGD robustness is the
wrong requirement when the correct output changes with the input. It then states the paper's
ethical principles as Vehicle properties of the network, with the first, maximise expected
utility, as a worked example, and asks whether the principle conflicts reported in the paper also
appear under formal verification.

## Getting started

Each folder is self-contained, with its own `requirements.txt` and its own Python 3.11 `.venv`
(Python 3.11 is needed for Vehicle and Marabou). See each folder's README for installation and
the order in which to run its notebooks.

```
.
├── moral-classification     - Moral Machine dilemmas -> each cluster's choice as LTL, PGD and Vehicle verification
└── moral-regression         - rescue agent effort, PGD study, and ethical principles as Vehicle properties
```
