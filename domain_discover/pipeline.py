"""End-to-end domain discovery pipeline."""

from typing import Dict, Any, Optional
import torch
from tqdm import tqdm
import json
from pathlib import Path

from wordnet_conditioner import WordNetConditioner
from diffusion import DiffusionTextGenerator
from blackbox import BlackBox
from reinforce import REINFORCEOptimizer
from metrics import MetricsCalculator


def optimise_prompt_vector(
    generator: DiffusionTextGenerator,
    bb: BlackBox,
    prompt_vec: torch.nn.Parameter,
    target_label: int,
    eta: float = 0.85,
    max_steps: int = 8000,
    batch_size: int = 32,
    eval_every: int = 100,
    patience: int = 3,
    output_dir: Optional[Path] = None
) -> torch.Tensor:
    """Optimize prompt vector for controlled text generation.
    
    Args:
        generator: Text generation model
        bb: Black-box classifier
        prompt_vec: Initial prompt vector to optimize
        target_label: Target domain label
        eta: Target consistency threshold (default: 0.85)
        max_steps: Maximum optimization steps
        batch_size: Batch size for REINFORCE
        eval_every: Steps between evaluations
        patience: Number of consecutive successful evals before stopping
        output_dir: Optional directory to save results
        
    Returns:
        Optimized prompt vector
    """
    device = prompt_vec.embedding.device
    
    # Initialize optimizer and metrics calculator
    optimizer = REINFORCEOptimizer(
        prompt=prompt_vec,
        generator=generator,
        blackbox=bb,
        target_label=target_label
    )
    
    metrics_calc = MetricsCalculator(device)
    
    # Training loop
    best_consistency = 0.0
    above_threshold_count = 0
    all_metrics = []
    
    pbar = tqdm(range(max_steps))
    for step in pbar:
        # Optimization step
        reward, loss = optimizer.step(batch_size=batch_size)
        
        # Evaluation
        if (step + 1) % eval_every == 0:
            # Generate evaluation samples
            with torch.no_grad():
                eval_texts = generator.sample(
                    prompt_vec(),
                    num_samples=100,  # Fixed K=100 for evaluation
                    max_len=100
                )
            eval_labels = bb.predict(eval_texts)
            
            # Compute metrics
            metrics = metrics_calc.compute_metrics(
                texts=eval_texts,
                labels=eval_labels,
                target_label=target_label
            )
            all_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                "consistency": f"{metrics['consistency']:.3f}",
                "perplexity": f"{metrics['perplexity']:.1f}"
            })
            
            # Check early stopping
            if metrics["consistency"] >= eta:
                above_threshold_count += 1
                if above_threshold_count >= patience:
                    print(f"\\nReached target consistency for {patience} consecutive evaluations. Stopping.")
                    break
            else:
                above_threshold_count = 0
            
            # Save best prompt vector
            if metrics["consistency"] > best_consistency:
                best_consistency = metrics["consistency"]
                if output_dir:
                    torch.save(
                        prompt_vec.state_dict(),
                        output_dir / "best_prompt.pt"
                    )
    
    # Save training history
    if output_dir:
        with open(output_dir / "metrics_history.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
    
    return prompt_vec()


if __name__ == "__main__":
    # Simple integration test
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Setup test components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Mock classes for testing
    class MockGenerator(DiffusionTextGenerator):
        def sample(self, *args, **kwargs):
            return ["test text"] * kwargs["num_samples"]
    
    class MockBlackBox(BlackBox):
        def predict(self, texts):
            return [1] * len(texts)
    
    # Create test instances
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    
    generator = MockGenerator(model, tokenizer, device)
    blackbox = MockBlackBox("dummy_endpoint")
    prompt_vec = WordNetConditioner(hidden_dim=model.config.hidden_size).to(device)
    
    # Test optimization
    optimise_prompt_vector(
        generator=generator,
        bb=blackbox,
        prompt_vec=prompt_vec,
        target_label=1,
        max_steps=5  # Small number for testing
    )
