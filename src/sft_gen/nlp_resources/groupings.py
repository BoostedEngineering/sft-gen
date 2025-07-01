from functools import cache
from typing import Optional

import nltk
from nltk.tokenize import sent_tokenize
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_SIMILARITY_THRESHOLD,
    NLTK_DATA_REQUIREMENTS,
)

# Download NLTK data only if not already present
for data_name in NLTK_DATA_REQUIREMENTS:
    try:
        nltk.data.find(f"tokenizers/{data_name}")
    except LookupError:
        nltk.download(data_name)

# Initialize model once at module level
model = SentenceTransformer(DEFAULT_MODEL_NAME)


def generate_groupings(text: str) -> list[list[str]]:
    sentences = get_sentences_from_text(text)
    return group_sentences(sentences)


def get_sentences_from_text(text: str) -> list[str]:
    return sent_tokenize(text)


@cache
def get_embeddings(text: str) -> ndarray:
    return model.encode(text)


def check_similarity(
    embedding1: ndarray,
    embedding2: ndarray,
    threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    return cosine_similarity([embedding1, embedding2])[0][1] > threshold


def group_sentences(
    sentences: list[str], threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD
) -> list[list[str]]:
    groups: list[list[str]] = []
    current_sentence_group: list[str] = [sentences[0]]
    sentence_group_embedding: ndarray = get_embeddings(sentences[0])

    for i in range(1, len(sentences)):
        next_embedding = get_embeddings(sentences[i])
        if check_similarity(sentence_group_embedding, next_embedding, threshold):
            current_sentence_group.append(sentences[i])
        else:
            groups.append(current_sentence_group.copy())
            current_sentence_group = [sentences[i]]
        sentence_group_embedding = get_embeddings(" ".join(current_sentence_group))

    groups.append(current_sentence_group)
    return groups
