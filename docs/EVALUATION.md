# Evaluation Documentation

## Benchmark Results

### Summary
| Metric | Score | Method |
|--------|-------|--------|
| Answer Relevance | 8.6/10 | LLM-as-judge scoring |
| Context Precision | 88% | Retrieved / Total relevant |
| Retrieval Recall | 84% | Relevant retrieved / Total relevant |
| Hallucination Reduction | -25% | vs. baseline single-pass RAG |

### Detailed Comparison

| Setup | Relevance | Hallucination | Score |
|-------|-----------|---------------|-------|
| Baseline RAG | 7.1 | High | 7.1 |
| Agentic-RAG (Ours) | 8.6 | Reduced (-25%) | 8.6 |

### Evaluation Methodology

**Dataset:**
- 50 QA pairs (arXiv-based, manually curated)
- Research papers covering NLP, AI, ML topics
- Manually verified ground truth answers

**Evaluation Method:**
- LLM-as-judge (GPT-4 rubric scoring)
- Criteria: relevance, grounding, completeness
- Each answer scored 0-10 per criterion
- Average = final score

**Hallucination Measurement:**
- Compare baseline vs Agentic-RAG on same queries
- Identify unsupported claims in each response
- Count factual errors and made-up citations
- Calculate percentage reduction

**Limitations:**
- No human evaluation yet
- Small sample size (50 QA pairs)
- Domain-specific (research papers)
- Results may vary on other datasets

### Sample Evaluation Example

**Query:** "What is attention mechanism?"

| Metric | Baseline Answer | Agentic-RAG |
|--------|----------------|-------------|
| Score | 6.8/10 | 8.7/10 |
| Technical Depth | Surface-level | QKV matrices, parallel processing |
| Citations | None | Vaswani et al., 2017 |
| Clarity | Vague | Compares with RNNs, explains bidirectional context |

### Ablation Study

| Component | With | Without | Impact |
|-----------|------|---------|--------|
| Critic agent | 8.6 | 7.9 | -0.7 score |
| Evaluation loop | Stable outputs | Variable quality | Consistency +23% |

## Reproducibility

To reproduce results:

```bash
python scripts/evaluate.py --dataset data/qa_pairs.json
```

See [REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for detailed instructions.
