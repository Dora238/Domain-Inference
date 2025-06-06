"""Command-line interface for domain discovery pipeline."""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt_vector import PromptVector
from .diffusion import DiffusionTextGenerator
from .blackbox import BlackBox
from .pipeline import optimise_prompt_vector


def main():
    parser = argparse.ArgumentParser(description="Domain Discovery Pipeline")
    
    parser.add_argument(
        "--target_label",
        type=int,
        required=True,
        help="Target domain label"
    )
    
    parser.add_argument(
        "--blackbox_endpoint",
        type=str,
        required=True,
        help="URL endpoint for black-box classifier"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/diffuseq-large",
        help="HuggingFace model name for text generation"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=8000,
        help="Maximum optimization steps"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for REINFORCE training"
    )
    
    parser.add_argument(
        "--eta",
        type=float,
        default=0.85,
        help="Target consistency threshold"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize components
    generator = DiffusionTextGenerator(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    blackbox = BlackBox(endpoint=args.blackbox_endpoint)
    
    prompt_vec = PromptVector(
        hidden_dim=model.config.hidden_size,
        init_method='wordnet',
        embedding_model=blackbox,
        max_words=5000,
        min_words_per_category=20
    ).to(device)
    
    # Run optimization
    print("Starting prompt vector optimization...")
    optimise_prompt_vector(
        generator=generator,
        bb=blackbox,
        prompt_vec=prompt_vec,
        target_label=args.target_label,
        eta=args.eta,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        output_dir=output_dir
    )
    
    print(f"\\nOptimization complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
