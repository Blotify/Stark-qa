import logging
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils import split_sentences, content_tokens


class BasicRAG:
    """Basic RAG baseline: TF-IDF retrieval + extractive answering only.

    Deliberately simple (no dense embeddings, no hybrid fusion, no KAG) so it
    serves as a clean lower bound for the combined STARK-QA system.
    """

    def __init__(self, cache_dir: str = "cache/", chunk_size: int = 200):
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.stories: Dict[str, str] = {}
        self.story_chunks: List[str] = []
        self.tfidf_matrix = None

    def chunk_story(self, story_text: str) -> List[str]:
        words = story_text.split()
        return [' '.join(words[i:i + self.chunk_size])
                for i in range(0, len(words), self.chunk_size)
                if words[i:i + self.chunk_size]]

    def index_stories(self, stories: Dict[str, str]):
        self.logger.info("Indexing stories with Basic-RAG...")
        self.stories = stories
        self.story_chunks = []
        for story_text in stories.values():
            self.story_chunks.extend(self.chunk_story(story_text))
        if not self.story_chunks:
            self.logger.warning("Basic-RAG: no chunks produced")
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(self.story_chunks)
        self.logger.info(f"Basic-RAG indexed {len(self.story_chunks)} chunks")

    def retrieve_context(self, question: str, top_k: int = 3) -> str:
        if self.tfidf_matrix is None:
            return ""
        qv = self.vectorizer.transform([question])
        sims = cosine_similarity(qv, self.tfidf_matrix).flatten()
        top = np.argsort(sims)[::-1][:top_k]
        return "\n\n".join(self.story_chunks[i] for i in top if sims[i] > 0)

    def answer_question(self, question: str) -> str:
        context = self.retrieve_context(question)
        if not context:
            return "I don't have enough information to answer this question."
        q_terms = set(content_tokens(question))
        best, best_score = "", -1
        for sentence in split_sentences(context):
            score = len(q_terms & set(content_tokens(sentence)))
            if score > best_score and len(sentence) > 3:
                best, best_score = sentence, score
        return best if best else "Unable to find a relevant answer."
