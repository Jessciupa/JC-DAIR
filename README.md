# JC-DAIR 2026
Course material and project notes for DAIR, covering robust training and verification of NLP classifiers.

[Lab 2](Lab2/README.md) contains the example for NLP for training with PGD and hyperrectangles. The generation of the hyperectangles are not in this repo but available at [ANTONIO](https://github.com/ANTONIONLP/ANTONIO)

This data is pre-processed but is an example from ANTONIO. For the project, please select your own Dataset to apply the course teachings in XAI, Training, and eventually Verification.

While not constrained to, here are some examples in While not constrained to, here are some in [Dataset Examples](DatasetExamples.md).


Installation (recommended for lab computers)
------------
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
hash -r
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
```
