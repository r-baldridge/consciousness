#!/usr/bin/env python3
"""
RWKV Inference CLI

Command-line interface for generating text with RWKV models.

Usage:
    python -m rwkv.cli.infer --checkpoint path/to/model.pt --prompt "Hello"
    python -m rwkv.cli.infer --checkpoint path/to/model.pt --interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate text with an RWKV model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model configuration",
    )

    # Input
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generation",
    )
    input_group.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="File containing input prompt",
    )
    input_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    gen_group.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling parameter (0 to disable)",
    )
    gen_group.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter",
    )
    gen_group.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    gen_group.add_argument(
        "--no_recurrent",
        action="store_true",
        help="Disable recurrent mode (use full sequence forward)",
    )
    gen_group.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling",
    )

    # Output
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to write generated text",
    )
    output_group.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token",
    )

    # Hardware
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    hw_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for inference",
    )
    hw_group.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster inference",
    )

    # Tokenizer
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="rwkv_world",
        help="Tokenizer to use",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: Optional[str], device: str, dtype: str):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Optional path to configuration file.
        device: Device to load model on.
        dtype: Data type for model weights.

    Returns:
        Loaded model.
    """
    try:
        import torch
        from ..src.model import RWKVConfig, RWKV
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please install PyTorch: pip install torch")
        sys.exit(1)

    logger.info(f"Loading model from {checkpoint_path}...")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        config = RWKVConfig(**config_dict) if isinstance(config_dict, dict) else config_dict
    elif config_path:
        import yaml
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = RWKVConfig(**config_dict.get("model", config_dict))
    else:
        logger.warning("No config found, using defaults")
        config = RWKVConfig()

    model = RWKV(config)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    logger.info(f"Model loaded with {model.num_parameters():,} parameters")

    return model


def load_tokenizer(tokenizer_name: str):
    """Load tokenizer.

    Args:
        tokenizer_name: Name or path of tokenizer.

    Returns:
        Tokenizer object.
    """
    if tokenizer_name == "rwkv_world":
        # RWKV World tokenizer - placeholder, use HF if available
        logger.warning("RWKV World tokenizer not available, using GPT-2")
        tokenizer_name = "gpt2"

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        logger.error("transformers library required for tokenization")
        logger.error("Install with: pip install transformers")
        sys.exit(1)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
    no_sample: bool = False,
    use_recurrent: bool = True,
    stream: bool = False,
) -> str:
    """Generate text from prompt.

    Args:
        model: RWKV model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum new tokens.
        temperature: Sampling temperature.
        top_k: Top-k parameter.
        top_p: Top-p parameter.
        no_sample: Use greedy decoding.
        use_recurrent: Use recurrent mode.
        stream: Stream output.

    Returns:
        Generated text.
    """
    import torch

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(next(model.parameters()).device)

    with torch.no_grad():
        if no_sample:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.01,
                top_k=1,
                use_recurrent=use_recurrent,
            )
        else:
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k and top_k > 0 else None,
                top_p=top_p,
                use_recurrent=use_recurrent,
            )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def interactive_mode(model, tokenizer, args: argparse.Namespace):
    """Run interactive generation mode.

    Args:
        model: RWKV model.
        tokenizer: Tokenizer.
        args: Command-line arguments.
    """
    print("\n" + "=" * 60)
    print("RWKV Interactive Generation")
    print("Type 'quit' or 'exit' to end, 'clear' to reset")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        prompt = prompt.strip()

        if not prompt:
            continue

        if prompt.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if prompt.lower() == "clear":
            print("\n" * 50)
            continue

        print("\nGenerating...\n")

        try:
            output = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p,
                no_sample=args.no_sample,
                use_recurrent=not args.no_recurrent,
                stream=args.stream,
            )
            print(output)
            print()
        except Exception as e:
            logger.error(f"Generation failed: {e}")


def main():
    """Main inference function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RWKV Inference")
    logger.info("=" * 60)

    if not args.interactive and not args.prompt and not args.prompt_file:
        logger.error("Must provide --prompt, --prompt_file, or --interactive")
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        logger.info("This is a placeholder CLI - create a checkpoint first by training")
        sys.exit(1)

    model = load_model(
        args.checkpoint,
        args.config,
        args.device,
        args.dtype,
    )

    if args.compile:
        import torch
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    tokenizer = load_tokenizer(args.tokenizer)

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    else:
        if args.prompt_file:
            with open(args.prompt_file, "r") as f:
                prompt = f.read().strip()
        else:
            prompt = args.prompt

        logger.info(f"Prompt: {prompt[:100]}...")
        logger.info("Generating...")

        output = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            no_sample=args.no_sample,
            use_recurrent=not args.no_recurrent,
            stream=args.stream,
        )

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(output)
            logger.info(f"Output written to {args.output_file}")
        else:
            print("\n" + "=" * 60)
            print("Generated Text:")
            print("=" * 60)
            print(output)
            print("=" * 60)


if __name__ == "__main__":
    main()
