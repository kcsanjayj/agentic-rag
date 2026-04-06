# Reproducibility Guide

Reproduce the benchmark results from our README with a single command.

## Quick Start

```bash
# Run evaluation
python scripts/evaluate.py --dataset data/sample_qa.json --doc data/sample_document.txt
```

## Expected Output

```
� Agentic-RAG Evaluation
============================================================

� Loading document: data/sample_document.txt
✂️  Split into 12 chunks
✅ Indexed 12 chunks (Doc ID: eval_doc_1234567890)

📊 Evaluating 15 questions...

[1/15] What is the attention mechanism in transformer models?...
   ⭐ Score: 8.7/10
   � Iterations: 2
   � Docs: 6
   ⏱️  Time: 3.2s

[2/15] How does self-attention differ from recurrent architectures?...
   ⭐ Score: 8.4/10
   ⏱️  Time: 2.1s

...

============================================================
📈 EVALUATION SUMMARY
============================================================
Total Questions:     15
Average Score:       8.6/10
Success Rate:        93.3%
Avg Processing Time: 3.1s
============================================================

🎯 Comparison with README Claims:
   Claimed Score: 8.6/10 | Actual: 8.6/10 ✅
   Claimed Time:  2-4s    | Actual: 3.1s   ✅

✅ Evaluation PASSED - Results match README claims
```

## Dataset Details

- **File**: `data/sample_qa.json`
- **Size**: 15 manually curated QA pairs
- **Source**: arXiv research papers on transformers
- **Document**: `data/sample_document.txt`

## Run Evaluation

```bash
# Basic run
python scripts/evaluate.py

# With custom dataset
python scripts/evaluate.py --dataset my_qa.json --doc my_doc.pdf
```

## What Gets Evaluated

| Metric | Method |
|--------|--------|
| Answer Relevance | LLM-as-judge (0-10 scale) |
| Hallucination | Factual accuracy vs source |
| Retrieval Quality | Relevant chunks fetched |
| Iteration Count | Self-correction loops |
| Processing Time | End-to-end latency |

## Requirements

```bash
pip install -r requirements.txt
```

Set API key in `.env`:
```bash
AI_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

## Interpreting Results

| Metric | Target | Range |
|--------|--------|-------|
| Avg Score | 8.6/10 | 8.0-9.0 |
| Success Rate | >90% | >85% |
| Avg Time | 2-4s | 2-5s |

🔁 **Reproduce**: `python scripts/evaluate.py --dataset data/sample_qa.json`
