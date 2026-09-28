# moral-classification

**Moral Machine dilemmas** written as sentences, each labelled with the choice people in three
cultural clusters made, and with that choice as an **LTL formula** saying who is protected. A
small network learns a cluster's choices from the text, and is then attacked with PGD and
formally verified with Vehicle, following the [ANTONIO](https://github.com/ANTONIONLP/ANTONIO)
pipeline for NLP verification:

```
dilemma -> sentence embedding (all-MiniLM-L6-v2, 384 dimensions) -> PCA to 30 dimensions, scaled to [0, 1]
        -> ReLU network (30 -> 128 -> 2) -> stay or swerve
```

1. embed the dilemmas and their character perturbations
2. build a hyperrectangle around each dilemma and its perturbations
3. train a classifier, with and without PGD adversarial training inside the hyperrectangles
4. verify with Vehicle and Marabou that every point in a hyperrectangle keeps the cluster's choice

Because the clusters sometimes choose differently, the same sentence can carry different labels.
This follows the position paper *Exploring Verification Frameworks for Social Choice Alignment*
(Ciupa, Belle and Komendantskaya, NeSy 2025): moral dilemmas are not decided in a social vacuum,
so verification should ask whether a model robustly aligns with a group's preferences.

Structure
------------
```
.
├── data
│   └── dilemmas.jsonl                    - the dilemmas: text, each cluster's votes and choice, LTL formula, split
├── dataset                               - how the data was built from the raw Moral Machine responses
│   ├── moral_machine.py                  - counts votes per dilemma and cluster: python dataset/moral_machine.py SharedResponses.csv
│   ├── votes.csv.gz                      - its output: every dilemma users from all three clusters answered
│   ├── build.py                          - writes data/dilemmas.jsonl: python dataset/build.py
│   ├── country_cluster_map.csv           - country -> cultural cluster (Awad et al., 2018)
│   └── ltl.py                            - LTL formula trees in canonical and raw form
├── src
│   ├── data.py                           - loading the dilemmas and a cluster's labels
│   ├── perturbations.py                  - character perturbations
│   ├── embed.py                          - sentence embedding, PCA and scaling
│   ├── hyperrectangles.py                - building hyperrectangles, and checking whether they contradict each other
│   ├── prepare.py                        - precomputes embeddings/: python src/prepare.py
│   └── verify_boxes.py                   - verifies many hyperrectangles, one Vehicle call per box
├── embeddings                            - precomputed embeddings, perturbations and hyperrectangles
├── NLP-Robustness.ipynb                  - data, base training, cultural clusters, PGD, adversarial training, export to ONNX
├── hyperrectangle-verification.ipynb     - Vehicle verification of the hyperrectangles, the coursework, and an extension
├── spec.vcl                              - Vehicle specification: robustness inside one hyperrectangle
├── verify.sh                             - verifies one hyperrectangle: ./verify.sh [box index] [network.onnx] [cluster]
├── requirements.txt                      - packages for the notebooks and verification
└── requirements-embed.txt                - optional: pandas, sentence-transformers and LIME, to rebuild the data
```

Installation
------------
This folder has its own environment. From inside `moral-classification/`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
hash -r
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
Python 3.11 is needed for Vehicle and Marabou. Select this `.venv` as the notebook kernel, and
run `NLP-Robustness.ipynb` before `hyperrectangle-verification.ipynb`: it writes the networks to
`models/` (not committed).

The dataset and embeddings are precomputed, so the notebooks need neither the raw Moral Machine
data nor sentence-transformers. To rebuild them, or for the LIME coursework, also install the
optional packages:
```
uv pip install --index-strategy unsafe-best-match -r requirements-embed.txt
python dataset/moral_machine.py path/to/SharedResponses.csv   # about 8 minutes
python dataset/build.py
python src/prepare.py
```

The dataset
------------

### The Moral Machine

The [Moral Machine](https://www.moralmachine.net/) experiment (Awad et al., *The Moral Machine
experiment*, Nature 2018) asked millions of people online what a self-driving car with sudden
brake failure should do. Every dilemma has two outcomes: **stay** on course or **swerve**, and
different characters die in each. The dilemmas vary nine factors: intervention (stay or swerve),
passengers vs pedestrians, legality (crossing on green or red), gender, age, fitness, social
status, number of characters, and species. There are 20 kinds of character, from a baby in a
stroller to an elderly woman, a doctor, a criminal or a cat.

The raw responses are public at [osf.io/3hvt2](https://osf.io/3hvt2/)
(`SharedResponses.csv.tar.gz`, 3.2 GB, about 70 million rows). Awad et al. grouped the 130
countries with enough respondents into three **cultural clusters**:

| cluster | countries | examples |
|---|---|---|
| western | 67 | United States, United Kingdom, Germany, Russia, Brazil |
| eastern | 28 | Japan, China, India, Saudi Arabia, Indonesia |
| southern | 35 | France, Mexico, Argentina, Colombia |

### How it is built

1. **Votes.** `dataset/moral_machine.py` pairs the two outcomes of each of the 33.6 million
   answered dilemmas, maps each respondent's country to its cluster, and counts, for each of the
   8.6 million distinct dilemmas, how many people in each cluster chose to stay and to swerve.
   `dataset/votes.csv.gz` keeps the 124,834 dilemmas that all three clusters answered.
2. **Selection.** `dataset/build.py` keeps the **2,295 dilemmas with at least 100 votes in every
   cluster**, so each cluster's majority is reliable.
3. **Labels.** A cluster's label is the choice most of its respondents made. A tie gives no label
   (2 western, 6 eastern, 0 southern). The vote counts and swerve shares are kept, so you can
   require a margin.
4. **Text.** Each dilemma is written out with one fixed template, in the style of the Moral
   Machine website's own descriptions:

   > A self-driving car has sudden brake failure. If it stays on course, it will drive through the
   > pedestrian crossing ahead, killing two boys and one woman. If it swerves, it will drive through
   > the pedestrian crossing in the other lane, killing two men and one elderly woman.

5. **LTL.** Each choice is written as the constraint it imposes: staying kills the characters
   ahead, so it protects those in the other lane, and vice versa. Staying in the dilemma above is
   `G ( ! ( kill_pedestrians_in_other_lane ) )`; swerving where the other lane is a barrier is
   `G ( ! ( kill_passengers ) )`.
6. **Split.** 80% of the dilemmas are used for training (1,836) and 20% for testing (459), at
   random with seed 0, the same for every cluster.

### Each row of `data/dilemmas.jsonl`

| field | contents |
|---|---|
| `id`, `natural`, `scenario_type`, `split` | the dilemma, its text, which factor it compares (Age, Fitness, Gender, Species, Social Status or Random), train or test |
| `stay`, `swerve` | for each outcome: passengers or pedestrians, the crossing signal (0 none, 1 green, 2 red), and the characters who die |
| `western`, `eastern`, `southern` | for each cluster: `votes`, `swerve_share`, `choice` (stay, swerve or null for a tie), and the choice as `formula` (canonical LTL) and `raw_ltl` |

### Counts

| | stay | swerve | tie |
|---|---|---|---|
| western | 1,305 | 988 | 2 |
| eastern | 1,308 | 981 | 6 |
| southern | 1,237 | 1,058 | 0 |

The clusters disagree on **240** of the 2,287 dilemmas all three label (10.5%). Most disagreements
are close votes, e.g. 50% of western respondents swerving against 40% of eastern respondents.

By scenario type: Age 680, Fitness 637, Gender 517, Species 268, Random 118, Social Status 75.
Dilemmas that only compare the number of characters have too many distinct casts to reach 100
votes each, so none are left.

Results
------------
From the worked example (western cluster, seed 0, one character perturbation per operation, as in
ANTONIO):

| | base | PGD-trained |
|---|---|---|
| test accuracy | 76.9% | 66.5% |
| accuracy on perturbed test dilemmas | 69.5% | 62.2% |
| hyperrectangles surviving PGD | 3.4% | 13.6% |
| hyperrectangles verified (of 1,834) | 48 (2.6%) | 220 (12.0%) |
| of which 'stay' / 'swerve' | 45 / 3 | 220 / 0 |
| predicts swerve (western respondents swerve on 44.2%) | 32.9% | 17.2% |

Verification used a 60-second limit per hyperrectangle; 2 (base) and 18 (PGD-trained) timed out.

**Character typos make poor boxes for this data.** Every dilemma uses the same wording and differs
only in its characters, so a typo in a character ("one cat" -> "one cta") moves the embedding
about as far as switching to a different dilemma. The boxes are as wide as the spread of the
dilemmas themselves, and **43% of them intersect a box with the opposite choice**. Robustness on
every box is therefore impossible, and PGD training resolves the conflicts by leaning towards
'stay': it verifies over four times as many boxes as the base model, but not a single 'swerve'
box. The coursework asks you to build better boxes, with synonyms or epsilon-cubes.

**The cluster models partly follow their own culture.** Training one model per cluster, each
reaches 75-79% test accuracy; on the 47 disputed test dilemmas the three models disagree with
each other on 40%, against 23% on all test dilemmas.

Limits
------------

- **One template.** Every dilemma is described in the same words, so the network can only learn
  from the characters and the structure, and typos behave unlike typos in free text.
- **Majorities hide close votes.** A 51% majority and a 90% majority get the same label.
- **Clusters are coarse.** Each cluster averages over many countries and respondents, and the
  Moral Machine's respondents are not a representative sample of any country.
- **Selection.** Requiring 100 votes in every cluster keeps the common dilemmas, which are mostly
  the six single-attribute scenario types, and drops almost all random and utilitarian ones.
