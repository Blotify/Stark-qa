import os
import json
import re
import logging
from typing import Dict, List, Any
from collections import defaultdict

from src.utils import fingerprint_stories, split_sentences, content_tokens, dedupe_preserve_order

try:
    import config as _cfg
    KG_MAX_ENTITIES = getattr(_cfg, 'KG_MAX_ENTITIES', 8)
    KG_MAX_CONTEXTS = getattr(_cfg, 'KG_MAX_CONTEXTS', 8)
except Exception:  # pragma: no cover
    KG_MAX_ENTITIES, KG_MAX_CONTEXTS = 8, 8

_NON_NAME_WORDS = {
    'the', 'and', 'but', 'for', 'with', 'once', 'upon', 'time', 'there', 'was',
    'were', 'had', 'has', 'this', 'that', 'they', 'them', 'then', 'when', 'while',
    'now', 'oh', 'but', 'come', 'here', 'what', 'who', 'how', 'why', 'where',
}

# Maps generic question words to concrete entity surface forms in the corpus.
_SEMANTIC_MATCHES = {
    'wife': ['queen', 'woman', 'mother', 'goodwife'],
    'husband': ['king', 'man', 'father', 'goodman'],
    'woman': ['wife', 'queen', 'mother', 'goodwife', 'lady'],
    'man': ['husband', 'king', 'father', 'goodman', 'lord'],
    'animal': ['cow', 'horse', 'pig', 'sheep', 'chicken', 'hen', 'cat', 'dog'],
    'animals': ['cow', 'cows', 'hens', 'cat', 'dog', 'horse'],
    'food': ['bread', 'cake', 'bannock', 'soup', 'porridge', 'milk'],
    'place': ['house', 'cottage', 'castle', 'kingdom', 'forest', 'village'],
}


class SimpleKnowledgeGraph:
    """Lightweight, dependency-free knowledge graph.

    Entities (characters, objects/places, descriptive concepts) and simple
    relationships are extracted with regular expressions, together with the
    sentences in which each entity appears. Retrieval matches a question to
    entities (exact -> partial -> semantic) and returns the most relevant stored
    sentences plus matching relationships.
    """

    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []
        self.stories: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    _CHARACTER_PATTERNS = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        r'\b(?:king|queen|prince|princess|lord|lady|sir|master|goodman|goodwife|old\s+(?:man|woman)|young\s+(?:man|woman))\b',
        r'\b(?:tailor|smith|miller|weaver|shepherd|cook|huntsman|guard|soldier|knight)\b',
        r'\b(?:father|mother|son|daughter|brother|sister|wife|husband|child|children)\b',
    ]
    _OBJECT_PATTERNS = [
        r'\b(?:sword|blade|knife|dagger|hammer|axe|bow|arrow|spear|club|stick|rod|staff)\w*\b',
        r'\b(?:dress|gown|cloak|coat|hat|bonnet|shoe|boot|ring|necklace|crown|tiara|belt|girdle)\w*\b',
        r'\b(?:bread|cake|bannock|soup|porridge|milk|water|wine|ale|meat|fish|apple|berry)\w*\b',
        r'\b(?:castle|palace|house|cottage|hut|tower|room|kitchen|chamber|hall|barn|mill|smithy|shop)\w*\b',
        r'\b(?:kingdom|village|town|city|forest|wood|mountain|hill|valley|river|lake|sea|bridge|road|path)\w*\b',
        r'\b(?:horse|cow|pig|sheep|goat|chicken|hen|cock|cat|dog|wolf|fox|bear|deer|bird|dragon)\w*\b',
        r'\b(?:magic|magical|enchanted|cursed|golden|silver|crystal|diamond|ruby|emerald|treasure|gold|jewel)\w*\b',
    ]
    _PHRASE_PATTERN = (
        r'\b(?:beautiful|ugly|brave|wise|foolish|kind|cruel|young|old|big|small|tall|'
        r'short|strong|weak|rich|poor|happy|sad|angry|golden|silver|magic|enchanted)\s+\w+\b'
    )

    def extract_simple_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []

        def collect(patterns, etype):
            found = set()
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    token = match.group(0).strip().lower()
                    if len(token) > 2 and token not in _NON_NAME_WORDS:
                        found.add(token)
            for token in found:
                entities.append({'text': token, 'type': etype})

        collect(self._CHARACTER_PATTERNS, 'PERSON')
        collect(self._OBJECT_PATTERNS, 'OBJECT')

        phrases = set()
        for match in re.finditer(self._PHRASE_PATTERN, text, re.IGNORECASE):
            phrase = match.group(0).strip().lower()
            if len(phrase) > 5:
                phrases.add(phrase)
        for phrase in phrases:
            entities.append({'text': phrase, 'type': 'CONCEPT'})

        return entities

    def extract_simple_relationships(self, text: str, entity_names: List[str]) -> List[Dict[str, Any]]:
        relationships = []
        patterns = [
            (r'(\w+)\s+(fought|defeated|killed|saved|helped|met|found)\s+(\w+)', 'action'),
            (r'(\w+)\s+(had|owned|carried|held)\s+(\w+)', 'possession'),
            (r'(\w+)\s+(lived in|went to|came from)\s+(\w+)', 'location'),
            (r'(\w+)\s+(was|became)\s+(\w+)', 'state'),
        ]
        entity_lower = {e.lower() for e in entity_names}
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject, predicate, obj = match.groups()
                if subject.lower() in entity_lower or obj.lower() in entity_lower:
                    relationships.append({
                        'subject': subject.lower(),
                        'predicate': predicate,
                        'object': obj.lower(),
                        'type': rel_type,
                    })
        return relationships

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def build_knowledge_graph(self, stories: Dict[str, str]):
        self.logger.info("Building knowledge graph...")
        cache_file = os.path.join(self.cache_dir, "simple_kg.json")
        fingerprint = fingerprint_stories(stories)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                if cache_data.get('fingerprint') == fingerprint and cache_data.get('entities'):
                    self.entities = cache_data['entities']
                    self.relationships = cache_data.get('relationships', [])
                    self.logger.info(
                        f"Loaded KG from cache: {len(self.entities)} entities, "
                        f"{len(self.relationships)} relationships")
                    return
                self.logger.info("KG cache stale/empty; rebuilding")
            except Exception as e:
                self.logger.warning(f"KG cache load failed: {e}; rebuilding")

        self.stories = stories
        self.entities, self.relationships = {}, []

        for story_name, story_text in stories.items():
            sentences = split_sentences(story_text)
            entities = self.extract_simple_entities(story_text)

            for entity in entities:
                name = entity['text']
                node = self.entities.setdefault(name, {
                    'original_text': entity['text'],
                    'type': entity['type'],
                    'stories': [],
                    'contexts': [],
                })
                if story_name not in node['stories']:
                    node['stories'].append(story_name)
                for sentence in sentences:
                    if name in sentence.lower() and len(sentence) > 15:
                        node['contexts'].append(sentence)

            entity_names = [e['text'] for e in entities]
            for rel in self.extract_simple_relationships(story_text, entity_names):
                rel['story'] = story_name
                self.relationships.append(rel)

        # Deduplicate stored contexts to keep retrieval clean and the cache small.
        for node in self.entities.values():
            node['contexts'] = dedupe_preserve_order(node['contexts'])[:20]

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({
                    'fingerprint': fingerprint,
                    'entities': self.entities,
                    'relationships': self.relationships,
                }, f, indent=2)
            self.logger.info("Saved knowledge graph to cache")
        except Exception as e:
            self.logger.warning(f"KG cache save failed: {e}")

        self.logger.info(
            f"Built KG with {len(self.entities)} entities and "
            f"{len(self.relationships)} relationships")

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _match_entities(self, question_lower: str, question_words: set) -> List[str]:
        matched = [name for name in self.entities if name in question_lower]
        if matched:
            return dedupe_preserve_order(matched)

        # Partial word overlap.
        for name in self.entities:
            if question_words & set(name.split()):
                matched.append(name)
        if matched:
            return dedupe_preserve_order(matched)

        # Semantic expansion.
        for word in question_words:
            for surface in _SEMANTIC_MATCHES.get(word, []):
                if surface in self.entities:
                    matched.append(surface)
        return dedupe_preserve_order(matched)

    def retrieve_context(self, question: str) -> str:
        if not self.entities and not self.relationships:
            return ""

        question_lower = question.lower()
        question_words = set(content_tokens(question))
        relevant_info: List[str] = []
        added = set()

        matched_entities = self._match_entities(question_lower, question_words)

        for name in matched_entities[:KG_MAX_ENTITIES]:
            node = self.entities[name]
            if node['type'] != 'CONCEPT':
                fact = f"{node['original_text']} is a {node['type'].lower()}"
                if fact not in added:
                    relevant_info.append(fact)
                    added.add(fact)

            ranked = sorted(
                node['contexts'],
                key=lambda c: len(question_words & set(content_tokens(c))),
                reverse=True,
            )
            for ctx in ranked[:3]:
                if ctx not in added:
                    relevant_info.append(f"Context: {ctx}")
                    added.add(ctx)

        for rel in self.relationships:
            text = f"{rel['subject']} {rel['predicate']} {rel['object']}"
            if question_words & set(content_tokens(text)):
                if text not in added:
                    relevant_info.append(text)
                    added.add(text)
            if len(relevant_info) >= KG_MAX_CONTEXTS + KG_MAX_ENTITIES:
                break

        # Broad fallback: entities whose contexts overlap the question.
        if not relevant_info:
            broad = []
            for name, node in self.entities.items():
                joined = ' '.join(node['contexts'][:2]).lower()
                overlap = len(question_words & set(content_tokens(joined)))
                if overlap > 0:
                    broad.append((overlap, node))
            broad.sort(key=lambda x: x[0], reverse=True)
            for _, node in broad[:3]:
                if node['contexts'] and node['contexts'][0] not in added:
                    relevant_info.append(f"Related: {node['contexts'][0]}")
                    added.add(node['contexts'][0])

        context = ". ".join(relevant_info[:KG_MAX_CONTEXTS])
        if context:
            self.logger.info(f"KG retrieved {len(relevant_info)} pieces of context")
        return context

    def retrieve_evidence_sentences(self, question: str) -> List[str]:
        """Return only the real story sentences linked to matched entities.

        Unlike ``retrieve_context``, this excludes synthetic facts ("X is a
        person") and relationship triples, so the result is a clean pool of
        candidate sentences for the extractive answerer.
        """
        if not self.entities:
            return []
        question_lower = question.lower()
        question_words = set(content_tokens(question))
        matched_entities = self._match_entities(question_lower, question_words)

        sentences, seen = [], set()
        for name in matched_entities[:KG_MAX_ENTITIES]:
            node = self.entities[name]
            ranked = sorted(
                node['contexts'],
                key=lambda c: len(question_words & set(content_tokens(c))),
                reverse=True,
            )
            for ctx in ranked[:3]:
                key = ctx.lower()
                if key not in seen:
                    seen.add(key)
                    sentences.append(ctx)
        return sentences

    def answer_question(self, question: str) -> str:
        """Standalone KAG answer (used when the KG is evaluated on its own)."""
        context = self.retrieve_context(question)
        if not context:
            return "I don't have enough information to answer this question."
        # Return the most question-relevant sentence from the retrieved context.
        question_words = set(content_tokens(question))
        sentences = [s for s in split_sentences(context) if len(s) > 10]
        if not sentences:
            return context[:200]
        sentences.sort(key=lambda s: len(question_words & set(content_tokens(s))), reverse=True)
        return sentences[0]
