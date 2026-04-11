# STARK-QA Project Configuration

# API Configuration
# Use a current model id ("-latest" or a dated id like "mistral-small-2503");
# the modern mistralai SDK rejects the bare "mistral-small" alias.
# Override at runtime with the MISTRAL_MODEL environment variable.
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_TEMPERATURE = 0.0   # deterministic answers
# Allow complete multi-clause / list answers (FairytaleQA answers can be long);
# the prompt still pushes for the shortest complete span.
MISTRAL_MAX_TOKENS = 80

# API Retry Configuration (for rate limiting)
API_RETRY_MAX_ATTEMPTS = 6
API_RETRY_INITIAL_DELAY = 2.0  # seconds
API_RETRY_EXPONENTIAL_BASE = 2  # for exponential backoff
# Proactive throttle: minimum seconds between Mistral requests. Staying just
# under the free-tier rate limit avoids most 429s (which otherwise force a
# lower-quality extractive fallback on some questions).
API_MIN_REQUEST_INTERVAL = 1.1

# RAG Configuration (chunk sizes are measured in words)
RAG_CHUNK_SIZE = 200
RAG_CHUNK_OVERLAP = 40
RAG_TOP_K = 5
RAG_HYBRID_ALPHA = 0.6  # Weight for dense retrieval in the hybrid fusion

# Knowledge Graph Configuration
KG_MAX_ENTITIES = 8       # Max entities considered per question
KG_MAX_CONTEXTS = 8       # Max context pieces returned per question

# Evaluation Configuration
EVAL_METRICS = [
    'rouge1', 'rouge2', 'rougeL', 
    'bleu', 'bert_score', 
    'flesch_reading_ease', 'gunning_fog'
]

# BERTScore model. Defaults to a lighter model to avoid the ~1.4GB roberta-large
# download (the bert-score default for English). Set to "roberta-large" for the
# canonical scorer, or override with the BERT_SCORE_MODEL environment variable.
BERT_SCORE_MODEL = "distilbert-base-uncased"

# Model Configuration
SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
DENSE_EMBEDDING_MODEL = 'all-mpnet-base-v2'
SPACY_MODEL = 'en_core_web_sm'

# File Paths
DEFAULT_CACHE_DIR = "cache/"
DEFAULT_OUTPUT_DIR = "results/"
DEFAULT_DATA_DIR = "data/"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'