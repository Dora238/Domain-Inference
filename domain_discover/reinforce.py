"""REINFORCE optimization for prompt vector learning."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

from .prompt_vector import PromptVector
from .diffusion import DiffusionTextGenerator
from .blackbox import BlackBox


class REINFORCEOptimizer:
    """REINFORCE optimizer for prompt vector learning.
    
    Implements REINFORCE with baseline for optimizing a prompt vector
    using only hard-label rewards from a black-box classifier.
    
    Args:
        prompt: Learnable prompt vector module
        generator: Text generation model
        blackbox: Black-box classifier
        target_label: Target class label
        learning_rate: Learning rate for Adam optimizer
        baseline_ema: EMA coefficient for reward baseline
        exploration_std: Standard deviation for exploration noise
    """
    
    def __init__(
        self,
        prompt: PromptVector,
        generator: DiffusionTextGenerator,
        blackbox: BlackBox,
        target_label: int,
        learning_rate: float = 1e-4,
        baseline_ema: float = 0.9,
        exploration_std: float = 0.1
    ):
        self.prompt = prompt
        self.generator = generator
        self.blackbox = blackbox
        self.target_label = target_label
        
        self.optimizer = torch.optim.Adam(
            prompt.parameters(),
            lr=learning_rate
        )
        
        self.baseline = 0.0  # Moving average of rewards
        self.baseline_ema = baseline_ema
        self.exploration_std = exploration_std
    
    def step(
        self,
        batch_size: int = 32,
        max_len: int = 100,
        temperature: float = 1.0
    ) -> Tuple[float, float]:
        """Perform one step of REINFORCE optimization.
        
        Args:
            batch_size: Number of samples for gradient estimation
            max_len: Maximum sequence length for generation
            temperature: Sampling temperature
            
        Returns:
            Tuple of (mean reward, loss value)
        """
        self.optimizer.zero_grad()
        
        # Get current prompt vector
        base_prompt = self.prompt()
        
        # Add exploration noise
        noise = torch.randn_like(base_prompt) * self.exploration_std
        noisy_prompt = base_prompt + noise
        
        # Generate samples and get rewards
        texts = self.generator.sample(
            noisy_prompt,
            num_samples=batch_size,
            max_len=max_len,
            temperature=temperature
        )
        labels = self.blackbox.predict(texts)
        rewards = torch.tensor(
            [1.0 if l == self.target_label else 0.0 for l in labels],
            device=base_prompt.device
        )
        
        # Compute advantage
        mean_reward = rewards.mean().item()
        self.baseline = (
            self.baseline_ema * self.baseline +
            (1 - self.baseline_ema) * mean_reward
        )
        advantage = rewards - self.baseline
        
        # Compute REINFORCE loss with baseline
        noise_dist = Normal(0, self.exploration_std)
        log_probs = noise_dist.log_prob(noise).sum(dim=-1)
        loss = -(log_probs * advantage).mean()
        
        # Update prompt vector
        loss.backward()
        self.optimizer.step()
        
        return mean_reward, loss.item()


if __name__ == "__main__":
    # Simple unit test setup
    import torch.nn.functional as F
    
    # Mock classes for testing
    class MockGenerator:
        def sample(self, *args, **kwargs):
            return ["test text"] * kwargs["num_samples"]
    
    class MockBlackBox:
        def predict(self, texts):
            return [1] * len(texts)  # Always return target label
    
    # Create test instances
    prompt = PromptVector(hidden_dim=4)
    generator = MockGenerator()
    blackbox = MockBlackBox()
    
    # Test optimizer
    optim = REINFORCEOptimizer(
        prompt=prompt,
        generator=generator,
        blackbox=blackbox,
        target_label=1
    )
    
    reward, loss = optim.step(batch_size=2)
    print(f"Test step - Reward: {reward:.3f}, Loss: {loss:.3f}")
