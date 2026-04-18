"""Dependency-free extractive answerer.

Given a question and a set of candidate sentences (typically the most relevant
sentences returned by RAG plus the knowledge-graph contexts), this picks the
best sentence and, for some question types, extracts a focused span. It is the
quality core of the no-API path: good extraction lets STARK-QA and the baselines
produce strong ROUGE/BLEU/F1 without any language model.
"""

import re
from typing import List

from src.utils import content_tokens, split_sentences

# Location preposition that typically introduces the answer to a "where" question.
_LOCATION_PREP_RE = re.compile(
    r'\b(in|at|by|near|on|inside|beside|under|over|within|into|upon|to)\b',
    re.IGNORECASE,
)
_EMOTIONS = ['terrified', 'frightened', 'delighted', 'disappointed', 'pleased',
             'scared', 'afraid', 'angry', 'happy', 'sad', 'glad', 'surprised',
             'astonished', 'sorry', 'content', 'ashamed', 'frightened']
_MAX_WORDS = 45

# Synthetic / meta sentences that should never be returned as an answer.
_SYNTHETIC_RE = re.compile(
    r'(\bis a (person|object|concept)\b|^related entities|^related:|'
    r'^story context|^knowledge graph|^entity details|^character info)',
    re.IGNORECASE,
)


def _clean(sentence: str) -> str:
    return sentence.strip(' "\',.;:').strip()


def _cap_words(text: str, max_words: int = _MAX_WORDS) -> str:
    words = _clean(text).split()
    return ' '.join(words[:max_words]) if len(words) > max_words else _clean(text)


def _best_sentences(question: str, candidates: List[str]) -> List[str]:
    """Rank candidate sentences by content-word overlap with the question."""
    q_terms = set(content_tokens(question))
    scored, seen = [], set()
    for sentence in candidates:
        sentence = _clean(sentence)
        key = sentence.lower()
        if len(sentence) < 3 or key in seen or _SYNTHETIC_RE.search(sentence):
            continue
        seen.add(key)
        s_terms = set(content_tokens(sentence))
        if not s_terms:
            continue
        # Recall of question terms, with a mild brevity preference on ties.
        overlap = len(q_terms & s_terms)
        scored.append((overlap - 0.002 * len(s_terms), sentence))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored]


def _location_span(sentence: str) -> str:
    """Return the location phrase: from the first location preposition to clause end."""
    match = _LOCATION_PREP_RE.search(sentence)
    if not match:
        return ""
    span = sentence[match.start():]
    # Stop at a clause boundary if one occurs (keeps the phrase tight).
    cut = re.search(r'[;]|,\s+(?:and|but|who|which|so|then)\b', span)
    if cut and len(span[:cut.start()].split()) >= 3:
        span = span[:cut.start()]
    return _cap_words(span)


def _who_span(sentence: str) -> str:
    """Extract the subject of a "there was/were/lived <X>" construction."""
    m = re.search(r'\bthere\s+(?:was|were|lived)\s+(?:once\s+)?(?:upon\s+a\s+time\s+)?'
                  r'(.*?)(?:,|\bwho\b|\bthat\b|\bwhich\b|$)', sentence, re.IGNORECASE)
    if m and m.group(1).strip():
        return _cap_words(m.group(1))
    return ""


def extract_answer(question: str, candidates: List[str]) -> str:
    """Produce a concise extractive answer from candidate sentences."""
    if not candidates:
        return "I don't have enough information to answer this question."

    ranked = _best_sentences(question, candidates)
    if not ranked:
        return "I don't have enough information to answer this question."

    best = ranked[0]
    q_lower = question.lower()

    if q_lower.startswith('where'):
        for sentence in ranked[:3]:
            span = _location_span(sentence)
            if span:
                return span
        return _cap_words(best)

    if q_lower.startswith('how') and ('feel' in q_lower or 'felt' in q_lower):
        for sentence in ranked[:3]:
            low = sentence.lower()
            for emotion in _EMOTIONS:
                if emotion in low:
                    return emotion
        return _cap_words(best)

    if q_lower.startswith('who'):
        for sentence in ranked[:3]:
            span = _who_span(sentence)
            if span:
                return span
        return _cap_words(best)

    # Default: the most relevant full sentence (length-capped for precision).
    return _cap_words(best)


def candidates_from_context(context: str) -> List[str]:
    """Split a combined context block back into candidate sentences."""
    cleaned = re.sub(r'(?im)^\s*[A-Z][A-Za-z ]+:\s*', ' ', context)
    return split_sentences(cleaned)
