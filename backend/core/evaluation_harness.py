"""
Evaluation Dataset Harness - Ground truth Q&A pairs for testing RAG system quality
"""

from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class EvaluationDataset:
    """Ground truth evaluation dataset for measuring RAG system performance"""
    
    def __init__(self, dataset_path: str = "data/evaluation/ground_truth.json"):
        self.dataset_path = dataset_path
        self.qa_pairs = []
        self.results_cache = []
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load existing dataset from disk"""
        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, 'r') as f:
                    data = json.load(f)
                    self.qa_pairs = data.get("qa_pairs", [])
                logger.info(f"Loaded {len(self.qa_pairs)} QA pairs from dataset")
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                self.qa_pairs = []
    
    def add_qa_pair(self, query: str, expected_answer: str, 
                   source_document: str, difficulty: str = "medium",
                   category: str = "general") -> None:
        """Add a new ground truth Q&A pair"""
        qa_pair = {
            "id": f"qa_{len(self.qa_pairs) + 1:03d}",
            "query": query,
            "expected_answer": expected_answer,
            "source_document": source_document,
            "difficulty": difficulty,  # easy, medium, hard
            "category": category,  # summarization, extraction, comparison, etc.
            "created_at": datetime.now().isoformat(),
            "metrics": {}
        }
        self.qa_pairs.append(qa_pair)
        self._save_dataset()
        logger.info(f"Added QA pair: {qa_pair['id']}")
    
    def evaluate_response(self, qa_id: str, actual_answer: str, 
                         retrieved_docs: List[Dict],
                         evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a response against ground truth"""
        qa_pair = next((qa for qa in self.qa_pairs if qa["id"] == qa_id), None)
        if not qa_pair:
            return {"error": f"QA pair {qa_id} not found"}
        
        # Calculate accuracy metrics
        expected_words = set(qa_pair["expected_answer"].lower().split())
        actual_words = set(actual_answer.lower().split())
        
        if expected_words:
            overlap = len(expected_words.intersection(actual_words))
            accuracy = overlap / len(expected_words)
        else:
            accuracy = 0.0
        
        result = {
            "qa_id": qa_id,
            "query": qa_pair["query"],
            "accuracy": round(accuracy, 2),
            "retrieval_count": len(retrieved_docs),
            "evaluation_metrics": evaluation_metrics,
            "passed": accuracy > 0.6 and evaluation_metrics.get("hallucination_rate", 1.0) < 0.5,
            "tested_at": datetime.now().isoformat()
        }
        
        self.results_cache.append(result)
        return result
    
    def run_full_evaluation(self, query_fn) -> Dict[str, Any]:
        """Run evaluation on all QA pairs"""
        logger.info(f"Starting full evaluation on {len(self.qa_pairs)} QA pairs")
        
        results = []
        for qa in self.qa_pairs:
            try:
                # Execute query
                response = query_fn(qa["query"])
                
                # Evaluate
                result = self.evaluate_response(
                    qa["id"],
                    response.get("answer", ""),
                    response.get("sources", []),
                    response.get("evaluation_metrics", {})
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {qa['id']}: {str(e)}")
                results.append({
                    "qa_id": qa["id"],
                    "error": str(e),
                    "passed": False
                })
        
        # Calculate aggregate metrics
        passed_count = sum(1 for r in results if r.get("passed", False))
        avg_accuracy = sum(r.get("accuracy", 0) for r in results) / len(results) if results else 0
        
        summary = {
            "total_qa_pairs": len(self.qa_pairs),
            "tested": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": round(passed_count / len(results), 2) if results else 0,
            "avg_accuracy": round(avg_accuracy, 2),
            "results": results,
            "evaluated_at": datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation complete: {summary['pass_rate']:.0%} pass rate")
        return summary
    
    def get_sample_test_set(self, n: int = 5) -> List[Dict]:
        """Get a sample of test cases"""
        import random
        if len(self.qa_pairs) <= n:
            return self.qa_pairs
        return random.sample(self.qa_pairs, n)
    
    def _save_dataset(self) -> None:
        """Save dataset to disk"""
        try:
            data = {
                "qa_pairs": self.qa_pairs,
                "updated_at": datetime.now().isoformat(),
                "count": len(self.qa_pairs)
            }
            with open(self.dataset_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        if not self.results_cache:
            return "No evaluation results available."
        
        report_lines = [
            "=" * 60,
            "RAG SYSTEM EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total QA Pairs: {len(self.qa_pairs)}",
            f"Tested: {len(self.results_cache)}",
            ""
        ]
        
        for result in self.results_cache:
            report_lines.extend([
                f"QA ID: {result['qa_id']}",
                f"  Accuracy: {result.get('accuracy', 'N/A'):.0%}",
                f"  Hallucination Rate: {result.get('evaluation_metrics', {}).get('hallucination_rate', 'N/A')}",
                f"  Groundedness: {result.get('evaluation_metrics', {}).get('groundedness', 'N/A')}",
                f"  Passed: {'✓' if result.get('passed') else '✗'}",
                ""
            ])
        
        return "\n".join(report_lines)


# Create sample dataset for testing
SAMPLE_QA_PAIRS = [
    {
        "query": "What is this document about?",
        "expected_answer": "This is a sample PDF file designed for testing purposes",
        "difficulty": "easy",
        "category": "summarization"
    },
    {
        "query": "What is the file size of this document?",
        "expected_answer": "approximately 100KB",
        "difficulty": "easy",
        "category": "extraction"
    },
    {
        "query": "What are the main use cases for this document?",
        "expected_answer": "mobile app testing, PDF rendering validation, email attachments",
        "difficulty": "medium",
        "category": "extraction"
    }
]

# Global instance
evaluation_harness = EvaluationDataset()

# Populate with sample data if empty
if not evaluation_harness.qa_pairs:
    for sample in SAMPLE_QA_PAIRS:
        evaluation_harness.add_qa_pair(
            query=sample["query"],
            expected_answer=sample["expected_answer"],
            source_document="sample-100kb.pdf",
            difficulty=sample["difficulty"],
            category=sample["category"]
        )
