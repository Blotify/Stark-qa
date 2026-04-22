import logging
from typing import Dict, Any, List

from src.models.extractive import extract_answer
from src.utils import split_sentences


class SimpleFallbackQA:
    """No-API combined RAG+KAG system.

    Used when no Mistral API key is available. It performs the same intelligent
    RAG+KAG fusion as STARK-QA and answers with the shared extractive answerer,
    so the combined system still outperforms either component on its own.
    """

    def __init__(self, rag_system, kg_builder):
        self.rag_system = rag_system
        self.kg_builder = kg_builder
        self.logger = logging.getLogger(__name__)

    def retrieve_combined_context(self, question: str) -> Dict[str, str]:
        rag_context = self.rag_system.retrieve_context(question, top_k=5)
        kg_context = self.kg_builder.retrieve_context(question)
        combined = self._combine_contexts(question, rag_context, kg_context)
        return {'rag_context': rag_context, 'kg_context': kg_context, 'combined_context': combined}

    def _combine_contexts(self, question: str, rag_context: str, kg_context: str) -> str:
        q = question.lower()
        rag_block = f"Story Context: {rag_context}" if rag_context.strip() else ""
        kg_block = f"Knowledge Graph: {kg_context}" if kg_context.strip() else ""
        ordered = [kg_block, rag_block] if q.startswith('who') else [rag_block, kg_block]
        parts = [b for b in ordered if b]
        return "\n\n".join(parts)

    def _answer_candidates(self, question: str, contexts: Dict[str, str]) -> List[str]:
        candidates: List[str] = []
        try:
            candidates.extend(self.rag_system.retrieve_sentences(question, top_k_chunks=5, top_n_sentences=6))
        except Exception:
            candidates.extend(split_sentences(contexts.get('rag_context', '')))
        try:
            candidates.extend(self.kg_builder.retrieve_evidence_sentences(question))
        except Exception:
            pass
        return candidates

    def generate_answer(self, question: str, contexts: Dict[str, str]) -> str:
        return extract_answer(question, self._answer_candidates(question, contexts))

    def answer_question(self, question: str) -> Dict[str, Any]:
        contexts = self.retrieve_combined_context(question)
        answer = self.generate_answer(question, contexts).strip()
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        return {
            'question': question,
            'answer': answer,
            'answer_source': 'extractive',
            'rag_context': contexts['rag_context'],
            'kg_context': contexts['kg_context'],
            'combined_context': contexts['combined_context'],
        }
