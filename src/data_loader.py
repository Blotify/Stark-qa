import re
import json
import os
import glob
import logging
from typing import Dict, List, Any

from src.utils import normalize_text_for_index, dedupe_preserve_order


class FairytaleQALoader:
    """Data loader for the FairytaleQA dataset with caching and preprocessing.

    The Hugging Face ``datasets`` dependency is imported lazily, so the pipeline
    can run fully offline whenever a cached split is present (the common case in
    this repo).
    """

    def __init__(self, cache_dir: str = "cache/"):
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

    def load_test_split(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Load the test split of FairytaleQA, using a JSON cache when available."""
        cache_file = os.path.join(self.cache_dir, f"fairytale_qa_test_{limit}.json")

        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached test data from {cache_file}")
            with open(cache_file, 'r') as f:
                return json.load(f)

        # Reuse a larger cached split offline by slicing it down to `limit`.
        larger = self._find_larger_cached_split(limit)
        if larger is not None:
            cache_path, data = larger
            self.logger.info(f"Slicing first {limit} questions from cached split {cache_path}")
            sliced = data[:limit]
            with open(cache_file, 'w') as f:
                json.dump(sliced, f, indent=2)
            return sliced

        # Only import the heavy dependency when we actually need to download.
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "No cached test split found and the 'datasets' package is not "
                "installed. Either install it (`pip install datasets`) or provide "
                f"a cache file at {cache_file}."
            ) from e

        self.logger.info("Downloading FairytaleQA dataset...")
        dataset = load_dataset("WorkInTheDark/FairytaleQA")
        test_data = dataset['test']

        processed_data = []
        for i, item in enumerate(test_data):
            if i >= limit:
                break

            processed_item = {
                'id': i,
                'story_name': item['story_name'],
                'story_section': item['story_section'],
                'question': item['question'],
                'answer': item['answer1'],  # Primary reference answer
                'answer2': item.get('answer2', ''),
                'local_or_sum': item['local-or-sum'],
                'attribute': item['attribute'],
                'ex_or_im': item['ex-or-im'],
                'ex_or_im2': item.get('ex-or-im2', ''),
            }
            processed_data.append(processed_item)

        with open(cache_file, 'w') as f:
            json.dump(processed_data, f, indent=2)

        self.logger.info(f"Loaded {len(processed_data)} test questions")
        return processed_data

    def _find_larger_cached_split(self, limit: int):
        """Return (path, data) for the smallest cached split with >= limit items."""
        candidates = []
        for path in glob.glob(os.path.join(self.cache_dir, "fairytale_qa_test_*.json")):
            match = re.search(r"fairytale_qa_test_(\d+)\.json$", os.path.basename(path))
            if not match:
                continue
            size = int(match.group(1))
            if size >= limit:
                candidates.append((size, path))
        for _, path in sorted(candidates):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                if len(data) >= limit:
                    return path, data
            except Exception:
                continue
        return None

    def get_unique_stories(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Reconstruct each unique story from its (deduplicated) sections.

        Many QA pairs reference the same ``story_section``; the previous
        implementation appended sections with a fragile substring check that
        produced massively duplicated text. Here we collect the distinct
        sections per story (preserving order) and join them once, which keeps the
        indexed text clean and improves retrieval quality.
        """
        sections_by_story: Dict[str, List[str]] = {}
        for item in data:
            story_name = item['story_name']
            section = normalize_text_for_index(item.get('story_section', ''))
            if not section:
                continue
            sections_by_story.setdefault(story_name, []).append(section)

        stories: Dict[str, str] = {}
        for story_name, sections in sections_by_story.items():
            unique_sections = dedupe_preserve_order(sections)
            stories[story_name] = ' '.join(unique_sections).strip()

        self.logger.info(f"Extracted {len(stories)} unique stories")
        return stories

    def get_story_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute simple dataset statistics (uses pandas if available)."""
        try:
            import pandas as pd
        except ImportError:
            self.logger.warning("pandas not available; returning minimal statistics")
            return {'total_questions': len(data)}

        df = pd.DataFrame(data)
        return {
            'total_questions': len(data),
            'unique_stories': df['story_name'].nunique(),
            'attribute_distribution': df['attribute'].value_counts().to_dict(),
            'local_vs_summary': df['local_or_sum'].value_counts().to_dict(),
            'explicit_vs_implicit': df['ex_or_im'].value_counts().to_dict(),
            'avg_question_length': df['question'].str.len().mean(),
            'avg_answer_length': df['answer'].str.len().mean(),
            'avg_story_length': df['story_section'].str.len().mean(),
        }
