#!/usr/bin/env python3
"""Validation script to ensure all essential components import and run."""


def main():
    print("Validating STARK-QA essential components...")

    try:
        from src.data_loader import FairytaleQALoader
        from src.rag.robust_rag import RobustRAG
        from src.kag.simple_kg import SimpleKnowledgeGraph
        from src.models.stark_qa import StarkQASystem
        from src.models.fallback_qa import SimpleFallbackQA
        from src.evaluation.evaluator import ComprehensiveEvaluator
        from src.baselines.basic_rag import BasicRAG
        from src.baselines.basic_kag import BasicKAG
    except ImportError as e:
        print(f"FAILED - import error: {e}")
        return False

    print("All imports successful.")

    # Smoke test: build tiny indices and answer one question end-to-end.
    import tempfile
    try:
        stories = {
            "demo": (
                "there was once an old man and his wife who lived in a little "
                "cottage by the side of a burn . they had two cows and five hens . "
                "the old woman baked a bannock for supper ."
            )
        }
        tmp_cache = tempfile.mkdtemp(prefix="stark_validate_")
        rag = RobustRAG(cache_dir=tmp_cache)
        rag.index_stories({"demo": stories["demo"]})
        kg = SimpleKnowledgeGraph(cache_dir=tmp_cache)
        kg.build_knowledge_graph({"demo": stories["demo"]})

        system = StarkQASystem(rag, kg)
        if system.client is None:
            system = SimpleFallbackQA(rag, kg)

        out = system.answer_question("where did the old man and his wife live ?")
        assert isinstance(out, dict) and out.get('answer'), "empty answer"
        print(f"End-to-end smoke test passed. Sample answer: {out['answer']!r}")
    except Exception as e:
        print(f"FAILED - runtime error: {e}")
        return False

    print("\nReady to run: python main.py --test_size 10")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
