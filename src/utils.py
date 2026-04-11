"""Shared utilities for STARK-QA.

This module centralises small, dependency-free helpers that are used across the
RAG, KAG, fusion and evaluation components: corpus fingerprinting (for cache
invalidation), sentence splitting, answer normalisation and lightweight lexical
similarity. Keeping these in one place avoids subtle behavioural drift between
components (e.g. two different definitions of "normalise an answer").
"""

import re
import hashlib
from typing import Dict, List, Iterable

# Words that carry little discriminative meaning for short-answer QA. Used by the
# lexical-overlap helpers and the rule-based answer extractors.
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'of', 'to', 'in', 'on', 'at',
    'by', 'for', 'with', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    'he', 'she', 'his', 'her', 'him', 'i', 'we', 'you', 'do', 'did', 'does',
    'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose',
    'so', 'then', 'than', 'too', 'very', 'can', 'will', 'would', 'should',
}

# Question words that should never be returned verbatim as an answer.
QUESTION_WORDS = {'who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom'}


def fingerprint_stories(stories: Dict[str, str]) -> str:
    """Return a stable hash describing a story corpus.

    Used to invalidate caches: if the set of stories (names or content) changes,
    the fingerprint changes and any cached index built from a different corpus is
    rebuilt instead of being silently reused.
    """
    hasher = hashlib.sha256()
    for name in sorted(stories.keys()):
        hasher.update(name.encode('utf-8', errors='ignore'))
        hasher.update(b'\x00')
        hasher.update(str(len(stories[name])).encode('utf-8'))
        # Hash a bounded sample of the content so very large corpora stay fast
        # while still being sensitive to content changes.
        content = stories[name]
        hasher.update(content[:512].encode('utf-8', errors='ignore'))
        hasher.update(content[-512:].encode('utf-8', errors='ignore'))
        hasher.update(b'\x01')
    return hasher.hexdigest()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences without requiring NLTK.

    FairytaleQA text uses spaced punctuation (e.g. "cottage ."), so we split on
    sentence-final punctuation and also treat the quotation marks common in the
    corpus as soft boundaries.
    """
    if not text:
        return []
    # Normalise whitespace first.
    text = re.sub(r'\s+', ' ', text).strip()
    # Split on . ! ? optionally followed by a closing quote.
    parts = re.split(r'(?<=[.!?])\s+|(?<=[.!?])["\']\s+', text)
    sentences = [p.strip(' "\'') for p in parts if p and p.strip(' "\'')]
    return sentences


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer (alphanumeric + apostrophes)."""
    return re.findall(r"[a-z0-9']+", text.lower())


def content_tokens(text: str) -> List[str]:
    """Tokens with stopwords and very short tokens removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def lexical_overlap(query: str, passage: str) -> float:
    """Jaccard-style content-word overlap between a query and a passage.

    Cheap, dependency-free relevance signal used to rank sentences/contexts when
    no learned model is available.
    """
    q = set(content_tokens(query))
    p = set(content_tokens(passage))
    if not q or not p:
        return 0.0
    return len(q & p) / len(q | p)


def normalize_answer(text: str) -> str:
    """Normalise an answer for Exact-Match / F1 scoring (SQuAD-style).

    Lowercase, strip punctuation, drop articles and collapse whitespace.
    """
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_text_for_index(text: str) -> str:
    """Collapse whitespace; used when storing story text."""
    return re.sub(r'\s+', ' ', text or '').strip()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Remove duplicates while preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
