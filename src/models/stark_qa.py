import os
import re
import time
import random
import logging
from typing import Dict, Any, List

from src.models.extractive import extract_answer
from src.utils import split_sentences

# Configuration with safe fallbacks.
try:
    import config as _cfg
    RETRY_MAX_ATTEMPTS = getattr(_cfg, 'API_RETRY_MAX_ATTEMPTS', 6)
    RETRY_INITIAL_DELAY = getattr(_cfg, 'API_RETRY_INITIAL_DELAY', 2.0)
    RETRY_EXPONENTIAL_BASE = getattr(_cfg, 'API_RETRY_EXPONENTIAL_BASE', 2)
    MIN_REQUEST_INTERVAL = getattr(_cfg, 'API_MIN_REQUEST_INTERVAL', 1.1)
    MISTRAL_MODEL = getattr(_cfg, 'MISTRAL_MODEL', 'mistral-small-latest')
    MISTRAL_TEMPERATURE = getattr(_cfg, 'MISTRAL_TEMPERATURE', 0.0)
    MISTRAL_MAX_TOKENS = getattr(_cfg, 'MISTRAL_MAX_TOKENS', 80)
except Exception:  # pragma: no cover
    RETRY_MAX_ATTEMPTS, RETRY_INITIAL_DELAY, RETRY_EXPONENTIAL_BASE = 6, 2.0, 2
    MIN_REQUEST_INTERVAL = 1.1
    MISTRAL_MODEL, MISTRAL_TEMPERATURE, MISTRAL_MAX_TOKENS = 'mistral-small-latest', 0.0, 80

# Optional Mistral client. The import path varies across SDK versions:
#   - some 1.x/2.x builds expose `from mistralai import Mistral`
#   - 2.4.x packages it as a namespace package: `from mistralai.client import Mistral`
#   - legacy (<1.0) uses `from mistralai.client import MistralClient` (client.chat(...))
# We try them in order so generation works regardless of the installed layout.
Mistral = None
MistralClient = None
MISTRAL_AVAILABLE = False
try:
    from mistralai import Mistral as _Mistral
    Mistral = _Mistral
    MISTRAL_AVAILABLE = True
except Exception:
    try:
        from mistralai.client import Mistral as _Mistral
        Mistral = _Mistral
        MISTRAL_AVAILABLE = True
    except Exception:
        try:
            from mistralai.client import MistralClient as _MistralClient
            MistralClient = _MistralClient
            MISTRAL_AVAILABLE = True
        except Exception:
            MISTRAL_AVAILABLE = False


class StarkQASystem:
    """STARK-QA: intelligent fusion of RAG and KAG.

    Retrieval combines hybrid RAG context with knowledge-graph context, ordered
    by question type so the most useful evidence appears first. Answer generation
    uses the Mistral API when configured, and otherwise a strong extractive
    fallback that operates over the *fused* RAG+KAG evidence — which is what makes
    the combined system beat either component alone even without an LLM.
    """

    def __init__(self, rag_system, kg_builder):
        self.rag_system = rag_system
        self.kg_builder = kg_builder
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.use_legacy = False
        # Allow a runtime override of the model id via the environment.
        self.model = os.getenv('MISTRAL_MODEL', MISTRAL_MODEL)
        self._last_request_ts = 0.0  # for proactive rate-limit throttling

        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            self.logger.info("MISTRAL_API_KEY not set; STARK-QA will use the extractive fallback")
            return
        if not MISTRAL_AVAILABLE:
            self.logger.info("mistralai not installed; STARK-QA will use the extractive fallback")
            return
        try:
            if Mistral is not None:
                self.client = Mistral(api_key=api_key)
                self.use_legacy = False
            elif MistralClient is not None:
                self.client = MistralClient(api_key=api_key)
                self.use_legacy = True
        except Exception as e:
            self.logger.error(f"Failed to initialise Mistral client: {e}")
            self.client = None

    # ------------------------------------------------------------------ #
    # Retrieval + fusion
    # ------------------------------------------------------------------ #
    def retrieve_combined_context(self, question: str) -> Dict[str, str]:
        rag_context = self.rag_system.retrieve_context(question, top_k=5)
        kg_context = self.kg_builder.retrieve_context(question)
        # Feed the LLM the clean KG *evidence sentences* (no synthetic "X is a Y"
        # facts or relationship triples) to reduce noise; keep the full kg_context
        # in the returned dict for reporting/inspection.
        try:
            kg_for_llm = " ".join(self.kg_builder.retrieve_evidence_sentences(question))
        except Exception:
            kg_for_llm = kg_context
        combined = self._smart_combine_contexts(question, rag_context, kg_for_llm or kg_context)
        return {'rag_context': rag_context, 'kg_context': kg_context, 'combined_context': combined}

    def _smart_combine_contexts(self, question: str, rag_context: str, kg_context: str) -> str:
        """Order RAG and KAG evidence by question type (KAG-first for entities)."""
        q = question.lower()
        rag_block = f"Story Context: {rag_context}" if rag_context.strip() else ""
        kg_block = f"Knowledge Graph: {kg_context}" if kg_context.strip() else ""

        if q.startswith('who'):
            ordered = [kg_block, rag_block]          # entities first
        elif q.startswith(('where', 'why')):
            ordered = [rag_block, kg_block]          # descriptive text first
        else:
            ordered = [rag_block, kg_block]

        parts = [b for b in ordered if b]
        return "\n\n".join(parts) if parts else "Limited context available."

    def _answer_candidates(self, question: str, contexts: Dict[str, str]) -> List[str]:
        """Build the candidate-sentence pool used by the extractive fallback."""
        candidates: List[str] = []
        # Focused RAG sentences first (highest precision).
        try:
            candidates.extend(self.rag_system.retrieve_sentences(question, top_k_chunks=5, top_n_sentences=6))
        except Exception:
            candidates.extend(split_sentences(contexts.get('rag_context', '')))
        # Then clean KAG evidence sentences (no synthetic "X is a Y" facts).
        try:
            candidates.extend(self.kg_builder.retrieve_evidence_sentences(question))
        except Exception:
            pass
        return candidates

    # ------------------------------------------------------------------ #
    # Answer generation
    # ------------------------------------------------------------------ #
    def generate_answer(self, question: str, contexts: Dict[str, str]):
        """Return (answer, source) where source is 'mistral' or 'extractive'."""
        if self.client is None:
            return self._generate_fallback_answer(question, contexts), 'extractive'
        try:
            messages = self._construct_messages(question, contexts['combined_context'])
            response = self._call_mistral(messages)
            if response and response.strip():
                return self._postprocess(response), 'mistral'
        except Exception as e:
            self.logger.error(f"Mistral call failed: {e}; using extractive fallback")
        return self._generate_fallback_answer(question, contexts), 'extractive'

    def _call_mistral(self, messages: List[Dict[str, str]]):
        def _do_call():
            if hasattr(self.client, 'chat') and hasattr(self.client.chat, 'complete'):
                resp = self.client.chat.complete(
                    model=self.model, messages=messages,
                    temperature=MISTRAL_TEMPERATURE, max_tokens=MISTRAL_MAX_TOKENS)
                return resp.choices[0].message.content
            if hasattr(self.client, 'chat'):
                resp = self.client.chat(
                    model=self.model, messages=messages,
                    temperature=MISTRAL_TEMPERATURE, max_tokens=MISTRAL_MAX_TOKENS)
                return resp.choices[0].message.content
            return None

        return self._api_call_with_retry(_do_call)

    def _generate_fallback_answer(self, question: str, contexts: Dict[str, str]) -> str:
        candidates = self._answer_candidates(question, contexts)
        return extract_answer(question, candidates)

    # System instructions + few-shot examples teach the FairytaleQA answer style:
    # verbatim spans, lowercase, complete lists, single emotion words. The
    # examples are generic (not drawn from the test set).
    _SYSTEM_PROMPT = (
        "You are a precise reading-comprehension assistant for children's fairy "
        "tales. Answer each question using ONLY the given story context.\n"
        "Rules:\n"
        "- If the story states the answer directly (who/what/where/when facts), "
        "copy that exact phrase verbatim, including every item of a list.\n"
        "- If the answer is implied rather than stated (a feeling, an intention, "
        "what a character wants or will do, or a one-line summary of what "
        "happened), give a short answer in your own words inferred from the story.\n"
        "- For a 'how did X feel' question, answer with ONE emotion adjective "
        "(e.g. scared, frightened, delighted, angry, sad) -- not a noun.\n"
        "- Keep it as short as possible while complete; at most ONE sentence or "
        "phrase; never output multiple sentences; no explanations.\n"
        "- Write entirely in lowercase.\n"
        "- Begin with the natural answer word (a 'where' answer starts with "
        "'in'/'at'/'by'); do not restate the question."
    )
    _FEWSHOT = [
        ("Context: the king lived in a great palace on the top of a high hill .\n"
         "Question: where did the king live ?",
         "in a great palace on the top of a high hill"),
        ("Context: she had three sons , named peter , paul , and hans .\n"
         "Question: what were the names of her three sons ?",
         "peter , paul , and hans"),
        ("Context: the giant roared and stamped his feet when he saw the broken window .\n"
         "Question: how did the giant feel when he saw the broken window ?",
         "angry"),
        ("Context: the fox crept closer , his mouth watering at the sight of the "
         "plump hen .\n"
         "Question: what did the fox want to do ?",
         "eat the hen"),
    ]

    def _construct_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        for user_ex, assistant_ex in self._FEWSHOT:
            messages.append({"role": "user", "content": user_ex})
            messages.append({"role": "assistant", "content": assistant_ex})
        messages.append({
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}",
        })
        return messages

    @staticmethod
    def _postprocess(answer: str) -> str:
        """Normalise an LLM answer toward the FairytaleQA reference style."""
        text = answer.strip().strip('"\'').strip()
        # Drop an echoed "Answer:" prefix if the model adds one.
        text = re.sub(r'^\s*answer\s*:\s*', '', text, flags=re.IGNORECASE)
        # FairytaleQA references are lowercase single sentences; match that style.
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        # Guard against rare multi-sentence run-ons: keep the first sentence.
        sentences = split_sentences(text)
        if len(sentences) > 1:
            text = sentences[0]
        return text.strip()

    def answer_question(self, question: str) -> Dict[str, Any]:
        contexts = self.retrieve_combined_context(question)
        raw_answer, source = self.generate_answer(question, contexts)
        answer = (raw_answer or "").strip()
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        return {
            'question': question,
            'answer': answer,
            'raw_answer': raw_answer,
            'answer_source': source,
            'rag_context': contexts['rag_context'],
            'kg_context': contexts['kg_context'],
            'combined_context': contexts['combined_context'],
        }

    def _throttle(self):
        """Sleep so consecutive requests stay >= MIN_REQUEST_INTERVAL apart."""
        if MIN_REQUEST_INTERVAL <= 0:
            return
        elapsed = time.time() - self._last_request_ts
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    def _api_call_with_retry(self, api_call_func, max_retries: int = None, initial_delay: float = None):
        max_retries = RETRY_MAX_ATTEMPTS if max_retries is None else max_retries
        initial_delay = RETRY_INITIAL_DELAY if initial_delay is None else initial_delay
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                result = api_call_func()
                self._last_request_ts = time.time()
                return result
            except Exception as e:
                self._last_request_ts = time.time()
                last_exception = e
                if attempt >= max_retries:
                    self.logger.error(f"API call failed after {max_retries + 1} attempts: {e}")
                    raise
                # Retry on ANY error (rate limits, transient 5xx, capacity, network).
                # Rate-limit-like errors get a longer backoff than other transient ones.
                error_str = str(e).lower()
                is_rate_limit = any(s in error_str for s in (
                    "429", "rate limit", "too many requests", "capacity", "service tier"))
                base = initial_delay if is_rate_limit else min(initial_delay, 1.0)
                delay = min(base * (RETRY_EXPONENTIAL_BASE ** attempt), 30.0) + random.uniform(0, 1)
                self.logger.warning(
                    f"API error (attempt {attempt + 1}/{max_retries + 1}; "
                    f"{'rate-limit' if is_rate_limit else 'transient'}); retrying in {delay:.2f}s")
                time.sleep(delay)
        raise last_exception
