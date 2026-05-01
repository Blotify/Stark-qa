# STARK-QA Evaluation Report

## System Comparison Overview

| System    |   Total Questions |   Valid Predictions |   Success Rate |   Exact Match |        F1 |   ROUGE-1 |   ROUGE-2 |   ROUGE-L |       BLEU |   Semantic Sim |   BERTScore F1 |   Avg Time (s) |
|:----------|------------------:|--------------------:|---------------:|--------------:|----------:|----------:|----------:|----------:|-----------:|---------------:|---------------:|---------------:|
| STARK-QA  |               100 |                 100 |              1 |          0.35 | 0.611027  |  0.628528 |  0.46718  | 0.627097  | 0.422103   |       0.729254 |       0.892872 |     1.72571    |
| Basic-RAG |               100 |                 100 |              1 |          0    | 0.17594   |  0.1959   |  0.127531 | 0.186376  | 0.0845513  |       0.380946 |       0.728666 |     0.00067554 |
| Basic-KAG |               100 |                 100 |              1 |          0.01 | 0.0410034 |  0.057857 |  0.01     | 0.0548877 | 0.00786654 |       0.193308 |       0.656725 |     0.00337144 |

## Detailed Analysis

**Best Overall System (by F1):** STARK-QA

### STARK-QA

- **Total Questions:** 100
- **Success Rate:** 1.000
- **Exact Match:** 0.350
- **F1:** 0.611
- **ROUGE-1 / ROUGE-L:** 0.629 / 0.627
- **BLEU:** 0.422
- **Semantic Similarity:** 0.729
- **BERTScore F1:** 0.893
- **Avg Response Time:** 1.726s
- **Answer sources:** mistral: 100
- **F1 by question type:** feeling 1.00 (n=8); character 0.82 (n=9); causal relationship 0.64 (n=19); action 0.58 (n=37); setting 0.58 (n=11); prediction 0.36 (n=7); outcome resolution 0.35 (n=9)

### Basic-RAG

- **Total Questions:** 100
- **Success Rate:** 1.000
- **Exact Match:** 0.000
- **F1:** 0.176
- **ROUGE-1 / ROUGE-L:** 0.196 / 0.186
- **BLEU:** 0.085
- **Semantic Similarity:** 0.381
- **BERTScore F1:** 0.729
- **Avg Response Time:** 0.001s
- **F1 by question type:** causal relationship 0.26 (n=19); outcome resolution 0.24 (n=9); action 0.20 (n=37); prediction 0.13 (n=7); character 0.10 (n=9); setting 0.10 (n=11); feeling 0.03 (n=8)

### Basic-KAG

- **Total Questions:** 100
- **Success Rate:** 1.000
- **Exact Match:** 0.010
- **F1:** 0.041
- **ROUGE-1 / ROUGE-L:** 0.058 / 0.055
- **BLEU:** 0.008
- **Semantic Similarity:** 0.193
- **BERTScore F1:** 0.657
- **Avg Response Time:** 0.003s
- **F1 by question type:** character 0.11 (n=9); outcome resolution 0.09 (n=9); causal relationship 0.08 (n=19); setting 0.03 (n=11); prediction 0.01 (n=7); action 0.01 (n=37); feeling 0.00 (n=8)

## Conclusions

STARK-QA fuses hybrid dense+sparse RAG retrieval with knowledge-graph context and answers with the Mistral LLM (few-shot, FairytaleQA-style prompt), falling back to a local extractive answerer only if the API is unavailable.
- It improves F1 over Basic-RAG by 3.5x (0.611 vs 0.176).
- It improves F1 over Basic-KAG by 14.9x (0.611 vs 0.041).
