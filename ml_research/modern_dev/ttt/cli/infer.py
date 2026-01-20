#!/usr/bin/env python3
"""
TTT Inference CLI

Command-line interface for running inference with Test-Time Training language models.

Usage:
    python -m ttt.cli.infer --checkpoint model.pt --prompt "Once upon a time"
    python -m ttt.cli.infer --checkpoint model.pt --prompt-file prompts.txt --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load TTT model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Loaded TTT model.
    """
    import torch
    from ..src.model import TTTLanguageModel

    logger.info(f"Loading model from {checkpoint_path}")
    model = TTTLanguageModel.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {num_params:,} parameters")
    logger.info(f"  TTT type: {model.config.ttt_type}")
    logger.info(f"  Hidden dim: {model.config.hidden_dim}")

    return model


def load_tokenizer(tokenizer_name: str = "gpt2"):
    """Load tokenizer.

    Args:
        tokenizer_name: Name or path of tokenizer.

    Returns:
        Tokenizer instance.

    Note:
        Placeholder - actual implementation would use HuggingFace tokenizers.
    """
    logger.warning("Using placeholder tokenizer. Implement actual tokenizer loading.")

    class DummyTokenizer:
        """Dummy tokenizer for placeholder."""

        def __init__(self, vocab_size: int = 50257):
            self.vocab_size = vocab_size
            self.eos_token_id = 50256
            self.pad_token_id = 50256

        def encode(self, text: str) -> List[int]:
            # Placeholder: Return random token IDs
            import random
            return [random.randint(0, self.vocab_size - 1) for _ in text.split()]

        def decode(self, token_ids: List[int]) -> str:
            # Placeholder: Return placeholder text
            return f"[Generated {len(token_ids)} tokens]"

    return DummyTokenizer()


def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Generate text from prompt.

    Args:
        model: TTT model.
        tokenizer: Tokenizer instance.
        prompt: Input prompt text.
        config: Generation configuration.
        device: Device to run on.

    Returns:
        Dictionary containing generated text and metadata.
    """
    import torch

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)

    # Reset hidden states for new sequence
    model.reset_hidden_states()

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=config.get("max_new_tokens", 100),
            temperature=config.get("temperature", 1.0),
            top_k=config.get("top_k"),
            top_p=config.get("top_p"),
        )

    # Decode
    generated_ids = output_ids[0].tolist()
    prompt_len = input_ids.size(1)
    new_ids = generated_ids[prompt_len:]
    generated_text = tokenizer.decode(new_ids)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": prompt + generated_text,
        "num_prompt_tokens": prompt_len,
        "num_generated_tokens": len(new_ids),
    }


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: str,
) -> Dict[str, float]:
    """Compute perplexity on text.

    Args:
        model: TTT model.
        tokenizer: Tokenizer instance.
        text: Input text.
        device: Device to run on.

    Returns:
        Dictionary containing perplexity metrics.
    """
    import torch

    # Tokenize
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor([input_ids], device=device)

    # Reset hidden states
    model.reset_hidden_states()

    # Compute loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs["loss"].item()

    perplexity = torch.exp(torch.tensor(loss)).item()

    return {
        "loss": loss,
        "perplexity": perplexity,
        "num_tokens": input_ids.size(1),
    }


def streaming_generate(
    model,
    tokenizer,
    prompt: str,
    config: Dict[str, Any],
    device: str,
):
    """Generate text with streaming output.

    Args:
        model: TTT model.
        tokenizer: Tokenizer instance.
        prompt: Input prompt text.
        config: Generation configuration.
        device: Device to run on.

    Yields:
        Generated tokens one at a time.
    """
    import torch
    import torch.nn.functional as F

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], device=device)

    # Reset hidden states
    model.reset_hidden_states()

    # Process prompt first
    with torch.no_grad():
        outputs = model(input_ids, return_hidden_states=True)
        hidden_states = outputs["hidden_states"]

    # Generate tokens one by one
    max_new_tokens = config.get("max_new_tokens", 100)
    temperature = config.get("temperature", 1.0)

    current_ids = input_ids

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                current_ids[:, -1:],
                hidden_states=hidden_states,
                return_hidden_states=True,
            )

        logits = outputs["logits"][:, -1, :]
        hidden_states = outputs["hidden_states"]

        # Apply temperature and sample
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Yield the token
        yield tokenizer.decode([next_token.item()])

        # Update for next iteration
        current_ids = torch.cat([current_ids, next_token], dim=1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break


def analyze_hidden_states(
    model,
    tokenizer,
    text: str,
    config: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Analyze hidden state dynamics during processing.

    Args:
        model: TTT model.
        tokenizer: Tokenizer instance.
        text: Input text.
        config: Analysis configuration.
        device: Device to run on.

    Returns:
        Dictionary containing hidden state analysis.
    """
    import torch

    # Tokenize
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor([input_ids], device=device)

    # Reset hidden states
    model.reset_hidden_states()

    # Track hidden state changes
    hidden_norms = []

    # Process token by token to track evolution
    with torch.no_grad():
        hidden_states = None
        for i in range(input_ids.size(1)):
            outputs = model(
                input_ids[:, i:i+1],
                hidden_states=hidden_states,
                return_hidden_states=True,
            )
            hidden_states = outputs["hidden_states"]

            # Compute norm of hidden states
            if hidden_states and hidden_states[0] is not None:
                # For TTT-Linear, hidden state is (batch, heads, d, d)
                norm = hidden_states[0].norm().item()
                hidden_norms.append(norm)

    return {
        "num_tokens": input_ids.size(1),
        "hidden_norms": hidden_norms,
        "mean_norm": sum(hidden_norms) / len(hidden_norms) if hidden_norms else 0,
        "max_norm": max(hidden_norms) if hidden_norms else 0,
        "min_norm": min(hidden_norms) if hidden_norms else 0,
    }


def batch_inference(
    model,
    tokenizer,
    prompts: List[str],
    mode: str,
    config: Dict[str, Any],
    device: str,
) -> List[Dict[str, Any]]:
    """Run batch inference on multiple prompts.

    Args:
        model: TTT model.
        tokenizer: Tokenizer instance.
        prompts: List of prompts.
        mode: Inference mode.
        config: Configuration.
        device: Device to run on.

    Returns:
        List of results.
    """
    results = []

    for i, prompt in enumerate(prompts):
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")

        if mode == "generate":
            result = generate_text(model, tokenizer, prompt, config, device)
        elif mode == "perplexity":
            result = compute_perplexity(model, tokenizer, prompt, device)
        elif mode == "analyze":
            result = analyze_hidden_states(model, tokenizer, prompt, config, device)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        result["prompt_index"] = i
        results.append(result)

    return results


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup device
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model and tokenizer
    model = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.tokenizer)

    # Build config
    config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        logger.error("No prompt specified. Use --prompt or --prompt-file")
        sys.exit(1)

    # Run inference
    if args.stream and args.mode == "generate" and len(prompts) == 1:
        # Streaming generation
        print(prompts[0], end="", flush=True)
        for token in streaming_generate(model, tokenizer, prompts[0], config, device):
            print(token, end="", flush=True)
        print()
    else:
        # Batch inference
        results = batch_inference(model, tokenizer, prompts, args.mode, config, device)

        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        else:
            print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with TTT language model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer name or path",
    )

    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt text",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "perplexity", "analyze"],
        help="Inference mode",
    )

    # Generation configuration
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output tokens (single prompt only)",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
