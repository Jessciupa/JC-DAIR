# Dataset Examples

Key things for this project to keep in mind when searching for databases. Semantically-relevant to the domain perturbations i.e. R U a Robot? Are U a Robot? (WORD PERTUBATION). Both need to be classified as a question whether the chatbot is a robot to legally disclose for safety. These pertubations in the example are from the embedding space. 

# NLP to Your Dataset Framework needs the following:

1. Encoder to embedding space. Allows for the exploration of robustness training and later verification with the embedding gap problem!
2. Label Preserving Variance Occurance in the Domain. This is the domain-specific semantic pertubations i.e. typos for NLP or input variations
3. If possible, a small label space i.e. binary classification (can you simplfy it for the specification?)

# 1. Long Horizon Planning

NL —> LTL (Paper: [https://arxiv.org/abs/2303.08006](https://arxiv.org/abs/2303.08006))

[https://github.com/IBM/nl2ltl](https://github.com/IBM/nl2ltl)

[https://github.com/UM-ARM-Lab/Efficient-Eng-2-LTL](https://github.com/UM-ARM-Lab/Efficient-Eng-2-LTL)

Does pertubations in the NL input affect the sequence of actions? 

"go to A, then B" vs "go to B, then A", which have the same words but a different temporal order. It will be interesting to see whether sentence embeddings even separate these.

Examples:

- **Classification task.** Classify each command by its LTL structure. Can be multi-class (5 classes) or binary.
- **Safe perturbations.** Swapping atomic propositions ("blue room" to "red room") keeps the formula structure, so the label should not change.
- **Label-changing perturbations.** Changing ordering words ("first", "then", "before") or avoidance words ("never", "avoid") changes the formula structure. These must be kept out of the hyperrectangles.

# Prosthetic Example

sEMG —> Handmovements (Paper(s): [https://ninapro.hevs.ch/publications.html](https://ninapro.hevs.ch/publications.html))

https://ninapro.hevs.ch/

Does pertubations in the embedding space by PGD attacks or pre-embedding perturbations cause misclassification?

i.e. Dropping a cup rather than grasping a cup could be safety-critical dependent on the scenario

Examples: 

- **Classification task.** Classify each grasps label from 14-channel sEMG. Can keep it simple to just the binary if thinking of industrial narrow applications.
- **Pre vs. Post Embedding perturbations.** Geometric and defined in embedding space or Raw sEMG pertubations (physical and defined on the raw signal)
- **Label-space:** Recommended to keep to binary. A confusion matrix to see the most confused pair to motivate the verification and training work is suggested. I.e. the pair that is most likely for misclassification motivates the safety-case for verification.

# Voice Command Example:

Speech audio —> Intent (Paper: [https://arxiv.org/abs/1904.03670](https://arxiv.org/abs/1904.03670))

[https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)

[https://huggingface.co/speechbrain/slu-direct-fluent-speech-commands-librispeech-asr](https://huggingface.co/speechbrain/slu-direct-fluent-speech-commands-librispeech-asr)

Does perturbation in the audio (speaker, accent, background noise) or in the wording change which action the robot executes?

i.e. "turn the heat up" being executed as "turn the heat down", or the lights switched off rather than on, could be safety-critical depending on the scenario.

- **Classification task.** The dataset is 16 kHz single-channel .wav files, each containing a single utterance used for controlling smart-home appliances or virtual assistants, for example "put on the music" or "turn up the heat in the kitchen". Each audio is labeled with three slots: action, object, and location. The 248 phrases map to 31 unique intents. Keep it binary by choosing opposite actions on the same object, e.g. increase vs decrease heat.
- **Label-changing perturbations.** "Turn on" vs "turn off", "up" vs "down", "kitchen" vs "bedroom". These are one word apart with a different action, so they must stay out of the hyperrectangles.
- **Pre vs post embedding perturbations.** As in the prosthetic example: PGD or ε-balls on the embedding from a frozen speech encoder (e.g. wav2vec 2.0 or Whisper), compared with noise, speaker and speed variations on the raw audio.

The official split already separates speakers: 23,132 utterances from 77 speakers for training, 3,118 utterances from another 10 speakers for validation, and 3,793 utterances from the remaining 10 speakers for testing. Unseen speakers are therefore a natural, recorded label-preserving test, much like later days in Ninapro DB6.