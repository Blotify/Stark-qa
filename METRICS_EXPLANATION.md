# Evaluation Metrics Explanation

This document explains what each metric tells you about the performance of the different QA systems.

## 1. BLEU Score (Bilingual Evaluation Understudy)

**What it measures**: How similar the predicted answer is to the ground truth answer at the word level.

**How it works**:

- Compares n-grams (sequences of words) between prediction and ground truth
- Ranges from 0 to 1 (higher is better)
- Considers word order and exact matches

**What it tells you**:

- **High BLEU (0.8-1.0)**: Prediction closely matches the ground truth with similar words and structure
- **Medium BLEU (0.4-0.7)**: Some word overlap but may have different phrasing or missing details
- **Low BLEU (0.0-0.3)**: Very different words/phrasing, or completely wrong answer

**Example**:

- Ground truth: "The old man and his wife lived in a cottage"
- Good prediction: "An old man and his wife lived in a cottage" (BLEU ~0.9)
- Poor prediction: "They lived somewhere" (BLEU ~0.1)

## 2. ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)

**What it measures**: How much of the ground truth information is captured in the prediction.

**Types**:

- **ROUGE-1**: Overlap of individual words
- **ROUGE-2**: Overlap of word pairs (bigrams)
- **ROUGE-L**: Longest common subsequence

**What it tells you**:

- **High ROUGE (0.7-1.0)**: Prediction captures most important information from ground truth
- **Medium ROUGE (0.4-0.6)**: Some key information captured but may miss details
- **Low ROUGE (0.0-0.3)**: Little to no relevant information captured

**Why it's important**: ROUGE focuses on recall (capturing information) while BLEU focuses on precision (exact matching).

## 3. Exact Match (EM)

**What it measures**: Percentage of predictions that exactly match the ground truth (after normalization).

**Normalization includes**:

- Converting to lowercase
- Removing punctuation
- Removing extra whitespace
- Removing articles (a, an, the)

**What it tells you**:

- **High EM (80-100%)**: System consistently produces exactly correct answers
- **Medium EM (40-79%)**: System often gets close but may have minor differences
- **Low EM (0-39%)**: System rarely produces exactly correct answers

**Example**:

- Ground truth: "The king and queen"
- Exact match: "king and queen" (normalized) ✓
- Not exact match: "the king and the queen" (extra article) ✗

## 4. F1 Score

**What it measures**: Harmonic mean of precision and recall at the word level.

**Components**:

- **Precision**: What fraction of predicted words are correct?
- **Recall**: What fraction of ground truth words are captured?
- **F1**: 2 × (Precision × Recall) / (Precision + Recall)

**What it tells you**:

- **High F1 (0.8-1.0)**: Good balance of precision and recall
- **Medium F1 (0.5-0.7)**: Reasonable performance but room for improvement
- **Low F1 (0.0-0.4)**: Poor word-level accuracy

## 5. Semantic Similarity

**What it measures**: How similar the meaning is between prediction and ground truth, regardless of exact words.

**How it works**: Uses sentence embeddings to compare semantic meaning rather than exact words.

**What it tells you**:

- **High Semantic Similarity (0.8-1.0)**: Prediction conveys the same meaning even if worded differently
- **Medium Semantic Similarity (0.5-0.7)**: Related meaning but may miss nuances
- **Low Semantic Similarity (0.0-0.4)**: Different or unrelated meaning

**Example**:

- Ground truth: "The cottage was beside a stream"
- High semantic similarity: "The house was next to a river" (different words, same meaning)
- Low semantic similarity: "It was very cold" (unrelated meaning)

## 6. Response Rate

**What it measures**: Percentage of questions that received a substantive answer (not "I don't know").

**What it tells you**:

- **High Response Rate (90-100%)**: System attempts to answer most questions
- **Medium Response Rate (70-89%)**: System sometimes unable to provide answers
- **Low Response Rate (0-69%)**: System frequently unable to answer

## 7. Average Response Time

**What it measures**: How long (in seconds) it takes to generate an answer.

**What it tells you**:

- **Fast (0-2 seconds)**: Efficient system, good for real-time applications
- **Medium (2-10 seconds)**: Reasonable for most applications
- **Slow (10+ seconds)**: May not be suitable for interactive use

## Interpreting Results by Question Type

### Character Questions ("who")

- High EM/F1: System correctly identifies specific characters
- High BLEU: Uses same character names as ground truth
- High Semantic Similarity: Identifies correct characters even with different phrasing

### Action Questions ("what did X do")

- High ROUGE: Captures the main actions described
- High Semantic Similarity: Understands the action even if worded differently
- High BLEU: Uses similar action words as ground truth

### Location Questions ("where")

- High EM: Identifies exact location names
- High F1: Captures location details accurately
- Low BLEU but High Semantic: May describe location differently but correctly

### Causal Questions ("why")

- High ROUGE: Captures reasoning from the text
- High Semantic Similarity: Understands causal relationships
- Lower EM expected: Causal explanations can be worded many ways

## System Comparison Guidelines

1. **Basic-RAG vs Basic-KAG**:

   - RAG should have better ROUGE (retrieves relevant text)
   - KAG might have better EM for entity questions (structured knowledge)

2. **STARK-QA vs Baselines**:

   - Should have better overall scores by combining RAG and KAG strengths
   - Better semantic similarity due to language model reasoning

3. **Red Flags**:

   - Very low response rate: System failing to engage with questions
   - High BLEU but low semantic similarity: Memorizing without understanding
   - High response time with low accuracy: Inefficient processing