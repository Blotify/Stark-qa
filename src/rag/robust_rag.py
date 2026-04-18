import os
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import fingerprint_stories, split_sentences, content_tokens

# Configuration with safe fallbacks if config.py is unavailable.
try:
    import config as _cfg
    CHUNK_SIZE = getattr(_cfg, 'RAG_CHUNK_SIZE', 200)
    CHUNK_OVERLAP = getattr(_cfg, 'RAG_CHUNK_OVERLAP', 40)
    TOP_K = getattr(_cfg, 'RAG_TOP_K', 5)
    HYBRID_ALPHA = getattr(_cfg, 'RAG_HYBRID_ALPHA', 0.6)
    DENSE_MODEL_NAME = getattr(_cfg, 'SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
except Exception:  # pragma: no cover - defensive
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, HYBRID_ALPHA = 200, 40, 5, 0.6
    DENSE_MODEL_NAME = 'all-MiniLM-L6-v2'

# Optional heavy dependencies. The system degrades gracefully to TF-IDF only.
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1]. Returns zeros if there is no spread."""
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.ones_like(scores) if hi > 0 else np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class RobustRAG:
    """Hybrid (dense + sparse) RAG that works even with minimal dependencies.

    Retrieval fuses normalised dense-embedding similarity with TF-IDF cosine
    similarity, then applies a light question-type re-ranking pass. When
    sentence-transformers / FAISS are unavailable the system transparently falls
    back to TF-IDF-only retrieval, which is still effective for this corpus.
    """

    def __init__(self, cache_dir: str = "cache/", chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP, hybrid_alpha: float = HYBRID_ALPHA):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hybrid_alpha = hybrid_alpha
        self.logger = logging.getLogger(__name__)

        self._init_models()

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000, stop_words='english', ngram_range=(1, 2)
        )

        self.stories: Dict[str, str] = {}
        self.story_chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []

        self.dense_index = None
        self.dense_embeddings_matrix = None
        self.tfidf_matrix = None

    @property
    def embedding_mode(self) -> str:
        """Identifier for the active retrieval mode, stored in the cache.

        A cache built in TF-IDF-only mode must not be reused once a real dense
        model is available (and vice versa), so this string is part of cache
        validity in addition to the corpus fingerprint.
        """
        return DENSE_MODEL_NAME if self.embeddings_available else "tfidf"

    def _init_models(self):
        """Initialise the dense embedding model if the dependency is present."""
        self.dense_model = None
        self.embeddings_available = False
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence-transformers not available; using TF-IDF only")
            return
        try:
            self.logger.info(f"Loading dense model '{DENSE_MODEL_NAME}'...")
            self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
            self.embeddings_available = True
        except Exception as e:
            self.logger.warning(f"Could not load dense model ({e}); using TF-IDF only")
            self.dense_model = None
            self.embeddings_available = False

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def chunk_story(self, story_text: str) -> List[str]:
        """Sentence-aware chunking.

        Sentences are accumulated until the chunk reaches ``chunk_size`` words; a
        trailing ``chunk_overlap`` words are carried into the next chunk so
        context spanning a boundary is not lost. Keeping sentences intact (vs the
        old fixed word-window) yields cleaner, more answerable passages.
        """
        sentences = split_sentences(story_text)
        if not sentences:
            return []

        chunks, current, current_len = [], [], 0
        for sentence in sentences:
            words = sentence.split()
            if current_len + len(words) > self.chunk_size and current:
                chunks.append(' '.join(current))
                # Build overlap from the tail of the current chunk.
                overlap_words = ' '.join(current).split()[-self.chunk_overlap:]
                current = overlap_words[:]
                current_len = len(current)
            current.extend(words)
            current_len += len(words)
        if current:
            chunks.append(' '.join(current))
        return [c for c in chunks if c.strip()]

    def index_stories(self, stories: Dict[str, str]):
        """Build (or load) the hybrid index for the given story corpus."""
        self.logger.info("Indexing stories with RobustRAG...")
        cache_file = os.path.join(self.cache_dir, "robust_rag_index.json")
        fingerprint = fingerprint_stories(stories)

        if os.path.exists(cache_file):
            try:
                if self._load_cache(cache_file, fingerprint):
                    self.logger.info("Loaded RAG index from cache")
                    return
                self.logger.info("Cache fingerprint mismatch; rebuilding RAG index")
            except Exception as e:
                self.logger.warning(f"Cache load failed: {e}; rebuilding RAG index")

        self.stories = stories
        self.story_chunks, self.chunk_metadata = [], []
        for story_name, story_text in stories.items():
            for i, chunk in enumerate(self.chunk_story(story_text)):
                self.story_chunks.append(chunk)
                self.chunk_metadata.append({'story_name': story_name, 'chunk_id': i})

        if not self.story_chunks:
            self.logger.warning("No chunks were produced from the supplied stories")
            return

        self.logger.info(f"Created {len(self.story_chunks)} chunks from {len(stories)} stories")

        self._build_dense_index()

        self.logger.info("Building TF-IDF index...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.story_chunks)

        try:
            self._save_cache(cache_file, fingerprint)
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")

    def _build_dense_index(self):
        if not self.embeddings_available:
            return
        try:
            self.logger.info("Creating dense embeddings...")
            embeddings = self.dense_model.encode(
                self.story_chunks, batch_size=16, show_progress_bar=False,
                convert_to_numpy=True,
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)
            self.dense_embeddings_matrix = embeddings.astype('float32')
            if FAISS_AVAILABLE:
                try:
                    self.dense_index = faiss.IndexFlatIP(embeddings.shape[1])
                    self.dense_index.add(self.dense_embeddings_matrix)
                    self.logger.info("FAISS index created")
                except Exception as e:
                    self.logger.warning(f"FAISS indexing failed: {e}; using numpy")
                    self.dense_index = None
        except Exception as e:
            self.logger.error(f"Dense embedding creation failed: {e}; TF-IDF only")
            self.embeddings_available = False
            self.dense_embeddings_matrix = None

    def _save_cache(self, cache_file: str, fingerprint: str):
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_data = {
            'fingerprint': fingerprint,
            'embedding_mode': self.embedding_mode,
            'stories': self.stories,
            'chunk_metadata': self.chunk_metadata,
            'story_chunks': self.story_chunks,
            'embeddings_available': self.embeddings_available,
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        if self.dense_embeddings_matrix is not None:
            np.save(cache_file.replace('.json', '_embeddings.npy'), self.dense_embeddings_matrix)
        self.logger.info("Saved RAG index to cache")

    def _load_cache(self, cache_file: str, fingerprint: str) -> bool:
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        # Both the corpus fingerprint and the retrieval mode must match, so a
        # TF-IDF-only cache is rebuilt once a real dense model is available.
        if cache_data.get('fingerprint') != fingerprint:
            return False
        if cache_data.get('embedding_mode') != self.embedding_mode:
            self.logger.info(
                f"Cache embedding mode '{cache_data.get('embedding_mode')}' != "
                f"'{self.embedding_mode}'; rebuilding")
            return False

        self.stories = cache_data['stories']
        self.chunk_metadata = cache_data['chunk_metadata']
        self.story_chunks = cache_data['story_chunks']
        if not self.story_chunks:
            return False

        if self.embeddings_available:
            embeddings_file = cache_file.replace('.json', '_embeddings.npy')
            if not os.path.exists(embeddings_file):
                # Mode says dense but the embeddings file is missing; rebuild.
                return False
            try:
                self.dense_embeddings_matrix = np.load(embeddings_file).astype('float32')
                if FAISS_AVAILABLE:
                    self.dense_index = faiss.IndexFlatIP(self.dense_embeddings_matrix.shape[1])
                    self.dense_index.add(self.dense_embeddings_matrix)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embeddings: {e}")
                return False

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.story_chunks)
        return True

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _dense_scores(self, query: str) -> np.ndarray:
        """Cosine similarity of the query against every chunk (dense)."""
        if not self.embeddings_available or self.dense_model is None or self.dense_embeddings_matrix is None:
            return np.zeros(len(self.story_chunks))
        try:
            q = self.dense_model.encode([query], convert_to_numpy=True)
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
            return np.dot(self.dense_embeddings_matrix, q[0]).astype(float)
        except Exception as e:
            self.logger.error(f"Dense scoring failed: {e}")
            return np.zeros(len(self.story_chunks))

    def _sparse_scores(self, query: str) -> np.ndarray:
        """TF-IDF cosine similarity of the query against every chunk."""
        if self.tfidf_matrix is None:
            return np.zeros(len(self.story_chunks))
        try:
            qv = self.tfidf_vectorizer.transform([query])
            return cosine_similarity(qv, self.tfidf_matrix).flatten().astype(float)
        except Exception as e:
            self.logger.error(f"Sparse scoring failed: {e}")
            return np.zeros(len(self.story_chunks))

    def retrieve_ranked(self, question: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """Return the top_k (chunk, fused_score) pairs using normalised hybrid fusion."""
        if not self.story_chunks:
            return []

        dense = _minmax_normalize(self._dense_scores(question))
        sparse = _minmax_normalize(self._sparse_scores(question))

        if self.embeddings_available and dense.any():
            fused = self.hybrid_alpha * dense + (1 - self.hybrid_alpha) * sparse
        else:
            fused = sparse  # TF-IDF only mode

        fused = self._apply_question_type_boost(question, fused)

        order = np.argsort(fused)[::-1]
        results = []
        for idx in order:
            if fused[idx] <= 0:
                break
            results.append((self.story_chunks[idx], float(fused[idx])))
            if len(results) >= top_k:
                break
        return results

    def _apply_question_type_boost(self, question: str, fused: np.ndarray) -> np.ndarray:
        """Lightly boost chunks that contain markers matching the question type."""
        q_lower = question.lower()
        boosted = fused.copy()

        marker_sets = []
        if q_lower.startswith('who'):
            marker_sets.append(({'king', 'queen', 'prince', 'princess', 'man', 'woman',
                                 'cook', 'tailor', 'weaver', 'smith', 'miller', 'wife',
                                 'husband', 'father', 'mother'}, 0.08))
        elif q_lower.startswith('where'):
            marker_sets.append(({'cottage', 'house', 'castle', 'forest', 'village',
                                 'palace', 'kitchen', 'kingdom', 'mountain', 'river'}, 0.08))
        elif q_lower.startswith('why'):
            marker_sets.append(({'because', 'since', 'so', 'wanted', 'needed', 'decided',
                                 'afraid', 'order'}, 0.08))
        elif q_lower.startswith('how') and ('feel' in q_lower or 'felt' in q_lower):
            marker_sets.append(({'scared', 'frightened', 'pleased', 'disappointed',
                                 'terrified', 'delighted', 'happy', 'sad', 'afraid',
                                 'angry', 'glad'}, 0.12))

        if not marker_sets:
            return boosted

        for i, chunk in enumerate(self.story_chunks):
            chunk_lower = chunk.lower()
            for markers, weight in marker_sets:
                if any(m in chunk_lower for m in markers):
                    boosted[i] += weight
        return boosted

    def retrieve_context(self, question: str, top_k: int = TOP_K) -> str:
        """Retrieve relevant context as a single joined string."""
        ranked = self.retrieve_ranked(question, top_k)
        if not ranked:
            return ""
        return "\n\n".join(chunk for chunk, _ in ranked)

    def retrieve_sentences(self, question: str, top_k_chunks: int = TOP_K,
                           top_n_sentences: int = 5) -> List[str]:
        """Return the most question-relevant sentences from the top chunks.

        Useful for extractive answering: it narrows broad chunks down to the few
        sentences most likely to contain a short answer.
        """
        ranked = self.retrieve_ranked(question, top_k_chunks)
        q_terms = set(content_tokens(question))
        scored = []
        seen = set()
        for chunk, chunk_score in ranked:
            for sentence in split_sentences(chunk):
                key = sentence.lower()
                if key in seen:
                    continue
                seen.add(key)
                s_terms = set(content_tokens(sentence))
                overlap = len(q_terms & s_terms)
                scored.append((sentence, overlap + 0.25 * chunk_score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_n_sentences]]

    def answer_question(self, question: str) -> str:
        """Extractive baseline answer: best sentence from the retrieved context."""
        sentences = self.retrieve_sentences(question, top_k_chunks=TOP_K, top_n_sentences=1)
        if sentences:
            return sentences[0]
        context = self.retrieve_context(question)
        return context[:200] if context else "I don't have enough information to answer this question."
