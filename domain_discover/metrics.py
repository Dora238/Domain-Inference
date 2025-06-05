"""Evaluation metrics for domain discovery."""

from typing import List, Dict, Any
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from collections import defaultdict


class MetricsCalculator:
    """Calculator for text generation metrics.
    
    Computes consistency (hit rate), diversity, and perplexity metrics
    for generated text samples.
    
    Args:
        device: Target device for perplexity computation
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        
        # Load GPT2-XL for perplexity calculation
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(
            "gpt2-xl"
        ).to(device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.gpt2_model.eval()
    
    def compute_metrics(
        self,
        texts: List[str],
        labels: List[int],
        target_label: int
    ) -> Dict[str, float]:
        """Compute all metrics for a batch of generated texts.
        
        Args:
            texts: List of generated text samples
            labels: List of classifier labels
            target_label: Target domain label
            
        Returns:
            Dictionary containing:
                - consistency: Fraction of samples with target label
                - diversity: 1 - average pairwise BLEU score
                - perplexity: Mean GPT2-XL perplexity
        """
        metrics = {}
        
        # Compute consistency (hit rate)
        metrics["consistency"] = np.mean(
            [1.0 if l == target_label else 0.0 for l in labels]
        )
        
        # Compute diversity using n-gram overlap
        metrics["diversity"] = self._compute_diversity(texts)
        
        # Compute perplexity using GPT2-XL
        metrics["perplexity"] = self._compute_perplexity(texts)
        
        return metrics
    
    def _compute_diversity(self, texts: List[str]) -> float:
        """Compute diversity score based on n-gram overlap.
        
        Uses 1 minus the average ratio of repeated n-grams as a
        simple diversity metric.
        """
        def get_ngrams(text: str, n: int) -> set:
            words = text.split()
            return set(" ".join(words[i:i+n]) 
                      for i in range(len(words)-n+1))
        
        # Compute for different n-gram sizes
        diversity_scores = []
        for n in [2, 3]:  # bi-grams and tri-grams
            all_ngrams = defaultdict(int)
            unique_ngrams = set()
            
            for text in texts:
                text_ngrams = get_ngrams(text.lower(), n)
                for ng in text_ngrams:
                    all_ngrams[ng] += 1
                    unique_ngrams.add(ng)
            
            if len(all_ngrams) > 0:
                # Compute ratio of unique n-grams
                diversity = len(unique_ngrams) / sum(all_ngrams.values())
                diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    @torch.no_grad()
    def _compute_perplexity(self, texts: List[str]) -> float:
        """Compute mean perplexity using GPT2-XL."""
        perplexities = []
        
        for text in texts:
            inputs = self.gpt2_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.gpt2_model(**inputs)
            loss = outputs.loss
            
            # Convert loss to perplexity
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)
        
        return np.mean(perplexities)


if __name__ == "__main__":
    # Simple unit test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = MetricsCalculator(device)
    
    test_texts = [
        "This is a test sentence.",
        "This is another test sentence.",
        "A completely different example."
    ]
    test_labels = [1, 1, 0]
    target_label = 1
    
    metrics = calculator.compute_metrics(
        texts=test_texts,
        labels=test_labels,
        target_label=target_label
    )
    
    print("Test metrics:", metrics)
