"""Character-level perturbations of a sentence, following ANTONIO.

Each perturbation changes one inner letter of one word of three or more letters:

    swap      two adjacent letters              elderly -> eledrly
    replace   a letter by a keyboard neighbour  elderly -> elserly
    delete    a letter                          elderly -> elerly
    insert    a random letter                   elderly -> eldqerly
    repeat    a letter                          elderly -> elderrly

Only the words that name the characters are perturbed (PERTURBABLE). Every dilemma shares
the same fixed wording ("A self-driving car has sudden brake failure..."), so the reduced
embedding has never seen that wording vary: a typo there moves the embedding further than
the difference between two dilemmas. The structure words (stays, swerves, passengers,
pedestrian, red, green, the numbers) are left alone too, since a typo there could change
which dilemma the sentence describes. Whether typos in the character words keep every label
is one of the questions in the coursework.
"""

import string

import numpy as np

# The words that name the characters, the part of the text that varies between dilemmas
PERTURBABLE = {
    'man', 'men', 'woman', 'women', 'pregnant', 'baby', 'babies', 'stroller', 'strollers',
    'boy', 'boys', 'girl', 'girls', 'male', 'female', 'athlete', 'athletes', 'executive',
    'executives', 'doctor', 'doctors', 'large', 'elderly', 'homeless', 'person', 'people',
    'criminal', 'criminals', 'dog', 'dogs', 'cat', 'cats',
}

# Neighbouring keys on a QWERTY keyboard
KEYBOARD = {
    'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg', 'y': 'tuh', 'u': 'yij',
    'i': 'uok', 'o': 'ipl', 'p': 'ol', 'a': 'qsz', 's': 'awdz', 'd': 'sefx', 'f': 'drgc',
    'g': 'fthv', 'h': 'gyjb', 'j': 'hukn', 'k': 'jilm', 'l': 'kop', 'z': 'asx',
    'x': 'zdc', 'c': 'xfv', 'v': 'cgb', 'b': 'vhn', 'n': 'bjm', 'm': 'nk',
}


def _swap(word, i, rng):
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _replace(word, i, rng):
    c = word[i]
    neighbours = KEYBOARD.get(c.lower(), c)
    new = neighbours[rng.integers(len(neighbours))]
    return word[:i] + (new.upper() if c.isupper() else new) + word[i + 1:]


def _delete(word, i, rng):
    return word[:i] + word[i + 1:]


def _insert(word, i, rng):
    return word[:i] + string.ascii_lowercase[rng.integers(26)] + word[i:]


def _repeat(word, i, rng):
    return word[:i] + word[i] + word[i:]


OPERATIONS = {'swap': _swap, 'replace': _replace, 'delete': _delete, 'insert': _insert, 'repeat': _repeat}


def _candidate_words(tokens):
    """Indices of the words that may be perturbed: character words of three or more letters."""
    return [k for k, t in enumerate(tokens)
            if sum(c.isalpha() for c in t) >= 3 and t.strip(string.punctuation).lower() in PERTURBABLE]


def perturb(sentence, operation, rng):
    """Apply one character operation to one inner letter of a random candidate word."""
    tokens = sentence.split(' ')
    candidates = _candidate_words(tokens)
    if not candidates:
        return sentence

    k = candidates[rng.integers(len(candidates))]
    word = tokens[k]
    # Inner letters only (not the first or last character), and never punctuation
    positions = [i for i in range(1, len(word) - 1) if word[i].isalpha()]
    if operation == 'swap':
        positions = [i for i in positions if i + 1 < len(word) - 1 and word[i + 1].isalpha()]
    if not positions:
        return sentence

    i = positions[rng.integers(len(positions))]
    tokens[k] = OPERATIONS[operation](word, i, rng)
    return ' '.join(tokens)


def character_perturbations(sentences, n_per_operation=1, seed=42):
    """Perturb every sentence n_per_operation times with each of the five operations.

    Returns (perturbed_sentences, source_index, operation_names). Perturbations that
    reproduce the original sentence or an earlier perturbation are skipped.
    """
    rng = np.random.default_rng(seed)
    perturbed, source, operations = [], [], []
    for s_index, sentence in enumerate(sentences):
        seen = {sentence}
        for operation in OPERATIONS:
            for _ in range(n_per_operation):
                p = perturb(sentence, operation, rng)
                if p not in seen:
                    seen.add(p)
                    perturbed.append(p)
                    source.append(s_index)
                    operations.append(operation)
    return np.array(perturbed), np.array(source), np.array(operations)
