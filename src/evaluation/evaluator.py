import os
import re
import json
import time
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter

import pandas as pd

from src.utils import normalize_answer, tokenize

# BERTScore model (lighter than the default roberta-large). Env var wins, then
# config, then a sensible default.
try:
    import config as _cfg
    _CFG_BERT_MODEL = getattr(_cfg, 'BERT_SCORE_MODEL', 'distilbert-base-uncased')
except Exception:  # pragma: no cover
    _CFG_BERT_MODEL = 'distilbert-base-uncased'
BERT_SCORE_MODEL = os.getenv('BERT_SCORE_MODEL', _CFG_BERT_MODEL)

# Optional metric libraries; pure-Python fallbacks are used when missing.
try:
    from rouge_score import rouge_scorer
    ROUGE_LIB = True
except ImportError:
    ROUGE_LIB = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_LIB = True
except ImportError:
    NLTK_LIB = False

try:
    from bert_score import score as bert_score
    BERT_LIB = True
except ImportError:
    BERT_LIB = False

try:
    import textstat
    TEXTSTAT_LIB = True
except ImportError:
    TEXTSTAT_LIB = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend so plotting never blocks
    import matplotlib.pyplot as plt
    MATPLOTLIB_LIB = True
except ImportError:
    MATPLOTLIB_LIB = False


# ---------------------------------------------------------------------- #
# Pure-Python metric fallbacks
# ---------------------------------------------------------------------- #
def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _f(overlap: int, pred_n: int, ref_n: int) -> float:
    if pred_n == 0 or ref_n == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_n
    recall = overlap / ref_n
    return 2 * precision * recall / (precision + recall)


def _ngram_overlap(pred: List[str], ref: List[str], n: int) -> Tuple[int, int, int]:
    pg = Counter(tuple(pred[i:i + n]) for i in range(len(pred) - n + 1))
    rg = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
    overlap = sum((pg & rg).values())
    return overlap, sum(pg.values()), sum(rg.values())


def _rouge_py(pred: str, ref: str) -> Dict[str, float]:
    p, r = tokenize(pred), tokenize(ref)
    o1, pn1, rn1 = _ngram_overlap(p, r, 1)
    o2, pn2, rn2 = _ngram_overlap(p, r, 2)
    lcs = _lcs_length(p, r)
    return {
        'rouge1': _f(o1, pn1, rn1),
        'rouge2': _f(o2, pn2, rn2),
        'rougeL': _f(lcs, len(p), len(r)),
    }


def _bleu_py(pred: str, ref: str, max_n: int = 4) -> float:
    p, r = tokenize(pred), tokenize(ref)
    if not p:
        return 0.0
    import math
    log_sum, used = 0.0, 0
    for n in range(1, max_n + 1):
        overlap, pn, _ = _ngram_overlap(p, r, n)
        if pn == 0:
            continue
        # Add-1 (Laplace) smoothing keeps short answers from collapsing to 0.
        precision = (overlap + 1) / (pn + 1)
        log_sum += math.log(precision)
        used += 1
    if used == 0:
        return 0.0
    geo_mean = math.exp(log_sum / used)
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(len(p), 1))
    return bp * geo_mean


def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r'[aeiouy]+', word)
    count = len(groups)
    if word.endswith('e') and count > 1:
        count -= 1
    return max(count, 1)


def _readability_py(text: str) -> Tuple[float, float]:
    words = tokenize(text)
    sentences = max(len(re.findall(r'[.!?]+', text)) or 1, 1)
    if not words:
        return 0.0, 0.0
    syllables = sum(_syllables(w) for w in words)
    complex_words = sum(1 for w in words if _syllables(w) >= 3)
    wps = len(words) / sentences
    flesch = 206.835 - 1.015 * wps - 84.6 * (syllables / len(words))
    fog = 0.4 * (wps + 100 * (complex_words / len(words)))
    return flesch, fog


def _token_f1(pred: str, ref: str) -> float:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    return _f(overlap, len(p), len(r))


class ComprehensiveEvaluator:
    """Evaluation system for QA models.

    Computes ROUGE, BLEU, Exact-Match, token-F1, semantic similarity, BERTScore
    (when available), readability and timing. Every metric has a pure-Python
    fallback, so evaluation runs with only the core scientific stack installed.
    """

    def __init__(self, output_dir: str = "results/"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        self._initialize_metrics()

    def _initialize_metrics(self):
        self.rouge_scorer = None
        self.smoothie = None
        if ROUGE_LIB:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            except Exception as e:
                self.logger.warning(f"rouge_score init failed: {e}; using built-in ROUGE")
        if NLTK_LIB:
            try:
                nltk.download('punkt', quiet=True)
                self.smoothie = SmoothingFunction().method4
            except Exception as e:
                self.logger.warning(f"nltk init failed: {e}; using built-in BLEU")
        self._semantic_model = None  # lazily loaded

    # ------------------------------------------------------------------ #
    # Running a system
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_prediction(response: Any) -> str:
        """Normalise any system response (dict or string) to a string answer."""
        if isinstance(response, dict):
            return str(response.get('answer', '')).strip()
        return str(response).strip()

    def evaluate_system(self, system, test_data: List[Dict[str, Any]], system_name: str) -> Dict[str, Any]:
        self.logger.info(f"Evaluating {system_name}...")
        results = {'system_name': system_name, 'predictions': [], 'metrics': {}, 'timing': {}}

        start_time = time.time()
        answer_times = []

        for i, item in enumerate(test_data):
            question = item['question']
            ground_truth = item['answer']

            pred_start = time.time()
            answer_source = 'n/a'
            try:
                if hasattr(system, 'answer_question'):
                    response = system.answer_question(question)
                    prediction = self._extract_prediction(response)
                    if isinstance(response, dict):
                        answer_source = response.get('answer_source', 'n/a')
                else:
                    prediction = "System error: answer_question method not found"
            except Exception as e:
                self.logger.error(f"Error predicting question {i}: {e}")
                prediction = "System error during prediction"
            pred_time = time.time() - pred_start
            answer_times.append(pred_time)

            results['predictions'].append({
                'question_id': item.get('id', i),
                'question': question,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'answer_source': answer_source,
                'question_type': item.get('attribute', 'unknown'),
                'local_or_sum': item.get('local_or_sum', 'unknown'),
                'ex_or_im': item.get('ex_or_im', 'unknown'),
                'prediction_time': pred_time,
            })

            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(test_data)} questions")

        total_time = time.time() - start_time
        results['timing'] = {
            'total_time': total_time,
            'avg_time_per_question': total_time / max(len(test_data), 1),
            'median_time_per_question': sorted(answer_times)[len(answer_times) // 2] if answer_times else 0,
        }
        results['metrics'] = self._calculate_metrics(results['predictions'])
        self._save_results(results, system_name)
        return results

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def _calculate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = {}
        preds = [p['prediction'] for p in predictions]
        truths = [p['ground_truth'] for p in predictions]

        metrics['total_questions'] = len(predictions)
        metrics['valid_predictions'] = sum(
            1 for p in preds if p and not p.startswith('System error'))

        # Track how answers were produced (LLM vs extractive fallback).
        source_counts = Counter(p.get('answer_source', 'n/a') for p in predictions)
        if set(source_counts) != {'n/a'}:
            metrics['answer_sources'] = dict(source_counts)

        pairs = [(p, t) for p, t in zip(preds, truths)
                 if p and t and not p.startswith('System error')]

        metrics.update(self._rouge_scores(pairs))
        metrics.update(self._bleu_scores(pairs))
        metrics.update(self._exact_match_and_f1(pairs))
        metrics.update(self._semantic_similarity(pairs))
        metrics.update(self._bert_scores(pairs))
        metrics.update(self._readability_scores(preds))

        pred_lengths = [len(p.split()) for p in preds if p]
        truth_lengths = [len(t.split()) for t in truths if t]
        metrics['avg_prediction_length'] = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0
        metrics['avg_ground_truth_length'] = sum(truth_lengths) / len(truth_lengths) if truth_lengths else 0

        metrics['by_question_type'] = self._type_metrics(predictions)
        return metrics

    def _rouge_scores(self, pairs) -> Dict[str, float]:
        r1, r2, rl = [], [], []
        for pred, truth in pairs:
            if self.rouge_scorer is not None:
                s = self.rouge_scorer.score(truth, pred)
                r1.append(s['rouge1'].fmeasure)
                r2.append(s['rouge2'].fmeasure)
                rl.append(s['rougeL'].fmeasure)
            else:
                s = _rouge_py(pred, truth)
                r1.append(s['rouge1']); r2.append(s['rouge2']); rl.append(s['rougeL'])
        avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
        return {'rouge1_avg': avg(r1), 'rouge2_avg': avg(r2), 'rougeL_avg': avg(rl)}

    def _bleu_scores(self, pairs) -> Dict[str, float]:
        scores = []
        for pred, truth in pairs:
            try:
                if self.smoothie is not None:
                    # Tokenize consistently (lowercase, punctuation split off) so
                    # tokens like "burn." are not counted as different from "burn".
                    pred_tokens = tokenize(pred)
                    truth_tokens = tokenize(truth)
                    if not pred_tokens or not truth_tokens:
                        continue
                    scores.append(sentence_bleu(
                        [truth_tokens], pred_tokens,
                        smoothing_function=self.smoothie))
                else:
                    scores.append(_bleu_py(pred, truth))
            except Exception:
                continue
        return {'bleu_avg': sum(scores) / len(scores) if scores else 0.0}

    def _exact_match_and_f1(self, pairs) -> Dict[str, float]:
        if not pairs:
            return {'exact_match': 0.0, 'f1': 0.0}
        em = sum(1 for p, t in pairs if normalize_answer(p) == normalize_answer(t))
        f1 = sum(_token_f1(p, t) for p, t in pairs)
        return {'exact_match': em / len(pairs), 'f1': f1 / len(pairs)}

    def _semantic_similarity(self, pairs) -> Dict[str, float]:
        """Embedding cosine similarity; TF-IDF cosine fallback when no model."""
        if not pairs:
            return {'semantic_similarity': 0.0}
        preds = [p for p, _ in pairs]
        truths = [t for _, t in pairs]

        # Try sentence-transformers for a true semantic signal.
        try:
            from sentence_transformers import SentenceTransformer, util
            if self._semantic_model is None:
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            ep = self._semantic_model.encode(preds, convert_to_tensor=True)
            et = self._semantic_model.encode(truths, convert_to_tensor=True)
            sims = util.cos_sim(ep, et).diagonal()
            return {'semantic_similarity': float(sims.mean())}
        except Exception:
            pass

        # TF-IDF cosine fallback.
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer().fit(preds + truths)
            mp, mt = vec.transform(preds), vec.transform(truths)
            sims = [float(cosine_similarity(mp[i], mt[i])[0][0]) for i in range(len(preds))]
            return {'semantic_similarity': sum(sims) / len(sims) if sims else 0.0}
        except Exception as e:
            self.logger.warning(f"Semantic similarity unavailable: {e}")
            return {'semantic_similarity': 0.0}

    def _bert_scores(self, pairs) -> Dict[str, float]:
        if not BERT_LIB or not pairs:
            return {}
        preds = [p for p, _ in pairs]
        truths = [t for _, t in pairs]
        # Prefer the configured (lighter) model; fall back to the lang default
        # if that model id is not recognised by bert-score.
        try:
            P, R, F1 = bert_score(preds, truths, model_type=BERT_SCORE_MODEL, verbose=False)
        except Exception as e:
            self.logger.warning(f"BERTScore with model '{BERT_SCORE_MODEL}' failed ({e}); "
                                f"trying default English model")
            try:
                P, R, F1 = bert_score(preds, truths, lang="en", verbose=False)
            except Exception as e2:
                self.logger.warning(f"BERTScore failed: {e2}")
                return {}
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item(),
        }

    def _readability_scores(self, preds) -> Dict[str, float]:
        valid = [p for p in preds if p and not p.startswith('System error')]
        if not valid:
            return {}
        flesch, fog = [], []
        for pred in valid:
            try:
                if TEXTSTAT_LIB:
                    flesch.append(textstat.flesch_reading_ease(pred))
                    fog.append(textstat.gunning_fog(pred))
                else:
                    f, g = _readability_py(pred)
                    flesch.append(f); fog.append(g)
            except Exception:
                continue
        return {
            'flesch_reading_ease': sum(flesch) / len(flesch) if flesch else 0.0,
            'gunning_fog_index': sum(fog) / len(fog) if fog else 0.0,
        }

    def _type_metrics(self, predictions) -> Dict[str, Any]:
        groups = defaultdict(list)
        for item in predictions:
            groups[item['question_type']].append(item)
            groups[f"local_sum_{item['local_or_sum']}"].append(item)
            groups[f"ex_im_{item['ex_or_im']}"].append(item)

        out = {}
        for name, items in groups.items():
            if not items:
                continue
            valid = sum(1 for it in items
                        if it['prediction'] and not it['prediction'].startswith('System error'))
            avg_f1 = sum(_token_f1(it['prediction'], it['ground_truth']) for it in items) / len(items)
            out[name] = {
                'count': len(items),
                'valid_predictions': valid,
                'success_rate': valid / len(items),
                'f1': avg_f1,
            }
        return out

    # ------------------------------------------------------------------ #
    # Persistence + reporting
    # ------------------------------------------------------------------ #
    def _save_results(self, results, system_name):
        safe = system_name.replace(' ', '_').replace('(', '').replace(')', '')
        json_file = os.path.join(self.output_dir, f"{safe}_results.json")
        with open(json_file, 'w') as f:
            json.dump(self._convert_for_json(results), f, indent=2)
        csv_file = os.path.join(self.output_dir, f"{safe}_predictions.csv")
        pd.DataFrame(results['predictions']).to_csv(csv_file, index=False)
        self.logger.info(f"Saved {system_name} results to {json_file} and {csv_file}")

    def _convert_for_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        if hasattr(obj, 'item'):
            return obj.item()
        return obj

    def generate_comparison_report(self, system_results: List[Tuple[str, Dict[str, Any]]]):
        self.logger.info("Generating comparison report...")
        rows = []
        for name, results in system_results:
            m, t = results['metrics'], results['timing']
            total = m.get('total_questions', 1) or 1
            rows.append({
                'System': name,
                'Total Questions': m.get('total_questions', 0),
                'Valid Predictions': m.get('valid_predictions', 0),
                'Success Rate': m.get('valid_predictions', 0) / total,
                'Exact Match': m.get('exact_match', 0),
                'F1': m.get('f1', 0),
                'ROUGE-1': m.get('rouge1_avg', 0),
                'ROUGE-2': m.get('rouge2_avg', 0),
                'ROUGE-L': m.get('rougeL_avg', 0),
                'BLEU': m.get('bleu_avg', 0),
                'Semantic Sim': m.get('semantic_similarity', 0),
                'BERTScore F1': m.get('bert_score_f1', 0),
                'Avg Time (s)': t.get('avg_time_per_question', 0),
            })
        comparison_df = pd.DataFrame(rows)
        comparison_df.to_csv(os.path.join(self.output_dir, "system_comparison.csv"), index=False)

        if MATPLOTLIB_LIB:
            try:
                self._generate_comparison_plots(comparison_df, system_results)
            except Exception as e:
                self.logger.warning(f"Plot generation failed: {e}")
        self._generate_detailed_report(comparison_df, system_results)
        self.logger.info(f"Comparison report saved to {self.output_dir}")

    def _generate_comparison_plots(self, comparison_df, system_results):
        for style in ('seaborn-v0_8', 'seaborn', 'ggplot', 'default'):
            try:
                plt.style.use(style)
                break
            except Exception:
                continue

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('STARK-QA System Comparison', fontsize=16, fontweight='bold')

        comparison_df.set_index('System')[['ROUGE-1', 'ROUGE-2', 'ROUGE-L']].plot(
            kind='bar', ax=axes[0, 0], title='ROUGE Scores')
        axes[0, 0].set_ylabel('Score'); axes[0, 0].tick_params(axis='x', rotation=45)

        comparison_df.set_index('System')[['F1', 'BLEU', 'Semantic Sim']].plot(
            kind='bar', ax=axes[0, 1], title='Accuracy Metrics')
        axes[0, 1].set_ylabel('Score'); axes[0, 1].tick_params(axis='x', rotation=45)

        comparison_df.set_index('System')[['Avg Time (s)']].plot(
            kind='bar', ax=axes[1, 0], title='Average Response Time', color='orange', legend=False)
        axes[1, 0].set_ylabel('Seconds'); axes[1, 0].tick_params(axis='x', rotation=45)

        type_metrics = system_results[0][1]['metrics'].get('by_question_type', {}) if system_results else {}
        names, f1s = [], []
        for tname, tdata in type_metrics.items():
            if not tname.startswith(('local_sum', 'ex_im')):
                names.append(tname); f1s.append(tdata.get('f1', 0))
        if names:
            axes[1, 1].bar(names, f1s, color='green', alpha=0.7)
            axes[1, 1].set_title(f'F1 by Question Type ({system_results[0][0]})')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No question-type data', ha='center', va='center',
                            transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "system_comparison_plots.png"),
                    dpi=200, bbox_inches='tight')
        plt.close()

    def _generate_detailed_report(self, comparison_df, system_results):
        report_file = os.path.join(self.output_dir, "evaluation_report.md")
        try:
            table = comparison_df.to_markdown(index=False)
        except Exception:
            table = comparison_df.to_string(index=False)

        with open(report_file, 'w') as f:
            f.write("# STARK-QA Evaluation Report\n\n## System Comparison Overview\n\n")
            f.write(table + "\n\n## Detailed Analysis\n\n")
            if not comparison_df.empty:
                best = comparison_df.loc[comparison_df['F1'].idxmax(), 'System']
                f.write(f"**Best Overall System (by F1):** {best}\n\n")
            for name, results in system_results:
                m, t = results['metrics'], results['timing']
                total = m.get('total_questions', 1) or 1
                f.write(f"### {name}\n\n")
                f.write(f"- **Total Questions:** {m.get('total_questions', 0)}\n")
                f.write(f"- **Success Rate:** {m.get('valid_predictions', 0) / total:.3f}\n")
                f.write(f"- **Exact Match:** {m.get('exact_match', 0):.3f}\n")
                f.write(f"- **F1:** {m.get('f1', 0):.3f}\n")
                f.write(f"- **ROUGE-1 / ROUGE-L:** {m.get('rouge1_avg', 0):.3f} / {m.get('rougeL_avg', 0):.3f}\n")
                f.write(f"- **BLEU:** {m.get('bleu_avg', 0):.3f}\n")
                f.write(f"- **Semantic Similarity:** {m.get('semantic_similarity', 0):.3f}\n")
                f.write(f"- **BERTScore F1:** {m.get('bert_score_f1', 0):.3f}\n")
                f.write(f"- **Avg Response Time:** {t.get('avg_time_per_question', 0):.3f}s\n")
                if m.get('answer_sources'):
                    srcs = ", ".join(f"{k}: {v}" for k, v in m['answer_sources'].items())
                    f.write(f"- **Answer sources:** {srcs}\n")
                self._write_type_breakdown(f, m.get('by_question_type', {}))
                f.write("\n")
            f.write("## Conclusions\n\n")
            if not comparison_df.empty and 'STARK-QA' in set(comparison_df['System']):
                row = comparison_df.set_index('System')
                sk = row.loc['STARK-QA']
                lines = [
                    "STARK-QA fuses hybrid dense+sparse RAG retrieval with knowledge-graph "
                    "context and answers with the Mistral LLM (few-shot, FairytaleQA-style "
                    "prompt), falling back to a local extractive answerer only if the API is "
                    "unavailable.",
                ]
                for base in ('Basic-RAG', 'Basic-KAG'):
                    if base in row.index and row.loc[base, 'F1']:
                        lines.append(
                            f"- It improves F1 over {base} by "
                            f"{sk['F1'] / row.loc[base, 'F1']:.1f}x "
                            f"({sk['F1']:.3f} vs {row.loc[base, 'F1']:.3f}).")
                f.write("\n".join(lines) + "\n")
            else:
                f.write("STARK-QA combines RAG and KAG and is compared against the "
                        "Basic-RAG and Basic-KAG baselines.\n")
        self.logger.info(f"Detailed report saved to {report_file}")

    @staticmethod
    def _write_type_breakdown(f, type_metrics: Dict[str, Any]):
        """Write a per-question-type F1 table (skips the local/ex-im groupings)."""
        rows = [(name, data) for name, data in type_metrics.items()
                if not name.startswith(('local_sum', 'ex_im')) and 'f1' in data]
        if not rows:
            return
        rows.sort(key=lambda x: x[1].get('f1', 0), reverse=True)
        f.write("- **F1 by question type:** ")
        f.write("; ".join(f"{name} {data['f1']:.2f} (n={data['count']})"
                          for name, data in rows))
        f.write("\n")
