import os
import argparse
import logging

# Load environment variables from a local .env file (e.g. MISTRAL_API_KEY,
# MISTRAL_MODEL) before anything else reads them. No-op if python-dotenv is
# absent or there is no .env file.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.data_loader import FairytaleQALoader
from src.rag.robust_rag import RobustRAG
from src.kag.simple_kg import SimpleKnowledgeGraph
from src.models.stark_qa import StarkQASystem
from src.models.fallback_qa import SimpleFallbackQA
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.baselines.basic_rag import BasicRAG
from src.baselines.basic_kag import BasicKAG

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stark_qa.log'),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='STARK-QA: Story-based RAG-KAG QA System')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test questions')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs('data/', exist_ok=True)
    
    logger.info("Starting STARK-QA pipeline...")
    
    # Load dataset
    logger.info("Loading FairytaleQA dataset...")
    data_loader = FairytaleQALoader(cache_dir=args.cache_dir)
    test_data = data_loader.load_test_split(limit=args.test_size)
    
    # Initialize components
    logger.info("Initializing RAG system...")
    rag_system = RobustRAG(cache_dir=args.cache_dir)
    
    logger.info("Building knowledge graph...")
    kg_builder = SimpleKnowledgeGraph(cache_dir=args.cache_dir)
    
    logger.info("Initializing STARK-QA system...")
    
    # STARK-QA uses the Mistral API when MISTRAL_API_KEY is set, and a strong
    # extractive fallback over the fused RAG+KAG evidence otherwise. The
    # SimpleFallbackQA is an equivalent no-API combined system kept for clarity.
    stark_qa = StarkQASystem(rag_system, kg_builder)
    if stark_qa.client is None:
        logger.info("Running STARK-QA in extractive (no-API) mode")
        stark_qa = SimpleFallbackQA(rag_system, kg_builder)
    stark_qa_name = "STARK-QA"
    
    # Initialize baselines
    logger.info("Initializing baseline systems...")
    basic_rag = BasicRAG(cache_dir=args.cache_dir)
    basic_kag = BasicKAG(cache_dir=args.cache_dir)
    
    # Process stories and build knowledge graph
    logger.info("Processing stories and building knowledge graph...")
    stories = data_loader.get_unique_stories(test_data)
    
    logger.info(f"Processing {len(stories)} unique stories...")
    rag_system.index_stories(stories)
    kg_builder.build_knowledge_graph(stories)
    basic_rag.index_stories(stories)
    basic_kag.build_knowledge_graph(stories)
    
    # Run evaluations
    logger.info("Running evaluations...")
    evaluator = ComprehensiveEvaluator(output_dir=args.output_dir)
    
    # Evaluate STARK-QA
    stark_results = evaluator.evaluate_system(stark_qa, test_data, stark_qa_name)
    
    # Evaluate baselines
    basic_rag_results = evaluator.evaluate_system(basic_rag, test_data, "Basic-RAG")
    basic_kag_results = evaluator.evaluate_system(basic_kag, test_data, "Basic-KAG")
    
    # Generate comparison report
    evaluator.generate_comparison_report([
        (stark_qa_name, stark_results),
        ("Basic-RAG", basic_rag_results),
        ("Basic-KAG", basic_kag_results)
    ])
    
    logger.info(f"Evaluation complete! Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()