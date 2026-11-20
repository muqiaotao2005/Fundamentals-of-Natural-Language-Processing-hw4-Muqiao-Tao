import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import hashlib
import re
from functools import lru_cache
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


DEFAULT_TRANSFORM_PROB = 0.35

_TOKEN_PATTERN = re.compile(r"<[^>]*>|&(?:#\w+|\w+);|\w+(?:['’-]\w+)*|[^\w\s]")
_PROTECTED_WORDS = set("""
    no not never neither nor nobody nothing nowhere none cannot without
    hardly barely rarely scarcely but however although though unless despite
    except only very more most less least almost quite just really rather
    someone somebody anyone anybody everyone everybody something anything
    everything somewhere anywhere everywhere
    well worth instead also even still always ever already
""".split())
_PROTECTED_PHRASES = (
    ("take", "place"), ("takes", "place"), ("took", "place"), ("taking", "place"),
    ("by", "chance"), ("in", "love"), ("kind", "of"), ("sort", "of"),
)

# Neighboring QWERTY keys for the vowels (one internal vowel per selected word).
_VOWEL_NEIGHBORS = {
    "a": "qwsz", "e": "wrsd", "i": "uojk", "o": "ipkl", "u": "yihj",
}


@lru_cache(maxsize=20000)
def _synonym_candidates(word, pos):
    """Use one WordNet sense with the same POS, not synonyms of every sense."""
    synsets = wordnet.synsets(word, pos=pos)
    if not synsets:
        return ()
    lemmas = synsets[0].lemmas()
    # Skip inflected forms that WordNet silently reduced to a different lemma.
    if word not in {lemma.name().lower() for lemma in lemmas}:
        return ()
    candidates = {
        lemma.name() for lemma in lemmas
        if lemma.name().isascii() and lemma.name().isalpha()
        and lemma.name().islower() and lemma.name() != word
        and lemma.name() not in _PROTECTED_WORDS and lemma.count() > 0
    }
    # Require this to also be the candidate's first sense. For example, this
    # avoids replacing "time" by "clip" or "police" by "law" in rare senses.
    return tuple(sorted(candidate for candidate in candidates
                        if wordnet.synsets(candidate, pos=pos)[0] == synsets[0]))


def custom_transform(example, method="typos", probability=DEFAULT_TRANSFORM_PROB, seed=0):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    if method not in {"typos", "synonyms"}:
        raise ValueError("method must be 'typos' or 'synonyms'")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")

    text = example["text"]
    transformed = dict(example)
    if probability == 0:
        return transformed

    # Stable per-review randomness: previews, evaluation, and augmentation agree.
    # The label is deliberately not used to choose edits.
    digest = hashlib.sha256(f"{seed}:{method}:{text}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    spans = list(_TOKEN_PATTERN.finditer(text))
    tokens = [span.group() for span in spans]
    lower_tokens = [token.lower() for token in tokens]
    protected_positions = set()
    for phrase in _PROTECTED_PHRASES:
        for i in range(len(tokens) - len(phrase) + 1):
            if tuple(lower_tokens[i:i + len(phrase)]) == phrase:
                protected_positions.update(range(i, i + len(phrase)))
    tags = pos_tag(tokens) if method == "synonyms" else [(t, None) for t in tokens]
    pos_mapping = {"NN": "n", "JJ": "a", "RB": "r"}

    pieces = []
    previous_end = 0
    for index, (span, (token, tag)) in enumerate(zip(spans, tags)):
        replacement = token
        word = token.lower()
        eligible = (token.isascii() and token.isalpha() and len(token) >= 4
                    and word not in _PROTECTED_WORDS and index not in protected_positions)
        if eligible and method == "typos":
            positions = [i for i in range(1, len(word) - 1)
                         if word[i] in _VOWEL_NEIGHBORS]
            if positions and rng.random() < probability:
                i = rng.choice(positions)
                letter = rng.choice(_VOWEL_NEIGHBORS[word[i]])
                if token[i].isupper():
                    letter = letter.upper()
                replacement = token[:i] + letter + token[i + 1:]
        elif eligible and tag in pos_mapping:
            candidates = _synonym_candidates(word, pos_mapping[tag])
            if candidates and rng.random() < probability:
                replacement = rng.choice(candidates)
                if token.isupper():
                    replacement = replacement.upper()
                elif token.istitle():
                    replacement = replacement.capitalize()

        # Replace spans in the original text to preserve spaces, punctuation,
        # contractions, HTML tags, and entities exactly.
        pieces.extend([text[previous_end:span.start()], replacement])
        previous_end = span.end()
    pieces.append(text[previous_end:])
    transformed["text"] = "".join(pieces)

    ##### YOUR CODE ENDS HERE ######

    return transformed
