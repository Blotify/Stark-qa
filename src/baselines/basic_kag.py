import re
import logging
from typing import Dict, List
from collections import defaultdict

# Optional dependencies. Basic-KAG works without them (regex NER + dict graph).
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class _DictGraph:
    """Minimal undirected graph used when networkx is unavailable."""

    def __init__(self):
        self._adj = defaultdict(set)

    def add_node(self, node):
        self._adj.setdefault(node, set())

    def add_edge(self, a, b):
        self._adj[a].add(b)
        self._adj[b].add(a)

    def __contains__(self, node):
        return node in self._adj

    def neighbors(self, node):
        return list(self._adj.get(node, set()))

    def nodes(self):
        return list(self._adj.keys())

    def number_of_nodes(self):
        return len(self._adj)


class BasicKAG:
    """Basic KAG baseline: entity extraction + co-occurrence graph.

    Uses spaCy NER when available, otherwise a regex-only extractor; uses
    networkx when available, otherwise a small built-in graph. Either way it
    builds a co-occurrence graph and answers from graph neighbourhoods, which is
    deliberately weaker than the combined STARK-QA system.
    """

    _CHARACTER_PATTERNS = [
        r'\b(?:king|queen|prince|princess|lord|lady|sir|master|goodman|goodwife)\b',
        r'\b(?:tailor|smith|miller|weaver|shepherd|cook|huntsman)\b',
        r'\b(?:father|mother|son|daughter|brother|sister|wife|husband)\b',
        r'\b(?:old\s+(?:man|woman)|young\s+(?:man|woman))\b',
    ]
    _OBJECT_PATTERNS = [
        r'\b(?:castle|kingdom|forest|village|cottage|house)\b',
        r'\b(?:sword|ring|crown|treasure|cloak|dress)\b',
        r'\b(?:horse|dragon|wolf|fox|bear|bird)\b',
    ]

    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)

        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning(
                    "spaCy model 'en_core_web_sm' not found; using regex-only "
                    "entity extraction (run `python -m spacy download en_core_web_sm` "
                    "to enable NER)")
                self.nlp = None

        self.graph = nx.Graph() if NETWORKX_AVAILABLE else _DictGraph()
        self.entity_to_stories: Dict[str, List[str]] = {}

    def extract_simple_entities(self, text: str) -> List[str]:
        entities = []
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ('PERSON', 'ORG', 'GPE', 'NORP'):
                        entities.append(ent.text.lower().strip())
            except Exception as e:
                self.logger.warning(f"spaCy NER failed: {e}")

        text_lower = text.lower()
        for pattern in self._CHARACTER_PATTERNS + self._OBJECT_PATTERNS:
            entities.extend(re.findall(pattern, text_lower))

        return list({e for e in entities if e and len(e) > 1})

    def build_knowledge_graph(self, stories: Dict[str, str]):
        self.logger.info("Building Basic-KAG knowledge graph...")
        for story_name, story_text in stories.items():
            entities = self.extract_simple_entities(story_text)
            for entity in entities:
                self.graph.add_node(entity)
                self.entity_to_stories.setdefault(entity, [])
                if story_name not in self.entity_to_stories[entity]:
                    self.entity_to_stories[entity].append(story_name)
            for i, e1 in enumerate(entities):
                for e2 in entities[i + 1:]:
                    if e1 != e2:
                        self.graph.add_edge(e1, e2)
        self.logger.info(f"Basic-KAG built graph with {self.graph.number_of_nodes()} entities")

    def _neighbors(self, entity, limit=None):
        """Neighbours of an entity as a list.

        networkx's ``Graph.neighbors`` returns an iterator (not sliceable); the
        built-in ``_DictGraph`` returns a list. Normalising to a list here keeps
        the call sites simple and correct for both backends.
        """
        if entity not in self.graph:
            return []
        neighbours = list(self.graph.neighbors(entity))
        return neighbours[:limit] if limit is not None else neighbours

    def retrieve_context(self, question: str) -> str:
        question_entities = self.extract_simple_entities(question)
        related = set()
        for entity in question_entities:
            if entity in self.graph:
                related.add(entity)
                related.update(self._neighbors(entity, 5))
        return "Related entities: " + ", ".join(sorted(related)) if related else ""

    def answer_question(self, question: str) -> str:
        question_lower = question.lower()
        question_entities = self.extract_simple_entities(question)

        if not question_entities:
            qwords = {w for w in question_lower.split() if len(w) > 2}
            for entity in self.graph.nodes():
                if qwords & set(entity.split()):
                    question_entities.append(entity)

        if question_entities:
            if question_lower.startswith('who'):
                for entity in question_entities:
                    if entity in ('king', 'queen', 'prince', 'princess', 'man', 'woman',
                                  'tailor', 'smith', 'miller', 'cook', 'weaver'):
                        return f"the {entity}"
                return "characters from the story"
            if question_lower.startswith('where'):
                for entity in question_entities:
                    if entity in ('cottage', 'house', 'castle', 'kingdom', 'forest', 'village'):
                        return f"the {entity}"
                return "a place in the story"
            if question_lower.startswith('what'):
                return ", ".join(question_entities[:3])
            # Default: name the most relevant entity and its neighbours.
            entity = question_entities[0]
            neighbours = self._neighbors(entity, 3)
            if neighbours:
                return f"{entity} (related to {', '.join(neighbours)})"
            return entity

        return "I don't have enough information to answer this question."
