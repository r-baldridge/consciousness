#!/usr/bin/env python3
"""
Hyena Inference CLI

Command-line interface for running inference with trained Hyena models.

Usage:
    python -m hyena.cli.infer --checkpoint path/to/checkpoint --prompt "Hello"
    python -m hyena.cli.infer --checkpoint path/to/checkpoint --interactive
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Hyena model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model loading
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
        help="Path to config (if not in checkpoint)",
    )

    # Input modes
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate",
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
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="Number of sequences to return per prompt",
    )

    # Output
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to write outputs to",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run on",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="bf16",
        help="Model dtype",
    )

    # Debug
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: Optional[str] = None, device: str = "cuda"):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config_path: Optional path to config file.
        device: Device to load model to.

    Returns:
        Loaded model and tokenizer.
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Placeholder implementation
    # Would load checkpoint and create model
    # import torch
    # from hyena.src.model import HyenaModel, HyenaConfig
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # config = HyenaConfig(**checkpoint["config"])
    # model = HyenaModel(config)
    # model.load_state_dict(checkpoint["model_state_dict"])
    # model.to(device)
    # model.eval()

    model = None
    tokenizer = None

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    num_return_sequences: int = 1,
    stream: bool = False,
) -> List[str]:
    """Generate text from a prompt.

    Args:
        model: Loaded Hyena model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Nucleus sampling.
        num_return_sequences: Number of sequences to return.
        stream: Whether to stream output.

    Returns:
        List of generated texts.
    """
    logger.info(f"Generating from prompt: {prompt[:50]}...")

    # Placeholder implementation
    # Would tokenize, generate, and decode
    # input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    # output_ids = model.generate(
    #     input_ids,
    #     max_new_tokens=max_new_tokens,
    #     temperature=temperature,
    #     top_k=top_k,
    #     top_p=top_p,
    # )
    # generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    generated_texts = [f"[Generated text for: {prompt}]"]

    return generated_texts


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
):
    """Stream generate text token by token.

    Args:
        model: Loaded Hyena model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        top_p: Nucleus sampling.

    Yields:
        Generated tokens one at a time.
    """
    # Placeholder implementation
    # Would implement streaming generation
    yield "[Streaming generation placeholder]"


def interactive_mode(
    model,
    tokenizer,
    args: argparse.Namespace,
):
    """Run interactive inference mode.

    Args:
        model: Loaded model.
        tokenizer: Tokenizer.
        args: Command-line arguments.
    """
    print("\n" + "=" * 60)
    print("Hyena Interactive Mode")
    print("=" * 60)
    print("Type your prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to exit.")
    print("Type 'config' to view/change generation settings.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()

            if not prompt:
                continue

            if prompt.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if prompt.lower() == "config":
                print(f"\nCurrent settings:")
                print(f"  max_new_tokens: {args.max_new_tokens}")
                print(f"  temperature: {args.temperature}")
                print(f"  top_k: {args.top_k}")
                print(f"  top_p: {args.top_p}")
                print()
                continue

            # Generate
            if args.stream:
                print("\n", end="")
                for token in stream_generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                ):
                    print(token, end="", flush=True)
                print("\n")
            else:
                outputs = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_return_sequences=args.num_return_sequences,
                )
                print("\n" + "-" * 40)
                for i, output in enumerate(outputs):
                    if args.num_return_sequences > 1:
                        print(f"[{i+1}] {output}")
                    else:
                        print(output)
                print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main inference entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load model
    model, tokenizer = load_model(
        args.checkpoint,
        args.config,
        args.device,
    )

    # Determine mode
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    elif args.prompt:
        # Single prompt
        outputs = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            stream=args.stream,
        )

        for output in outputs:
            print(output)

        if args.output_file:
            with open(args.output_file, "w") as f:
                for output in outputs:
                    f.write(output + "\n")
            logger.info(f"Outputs written to {args.output_file}")

    elif args.prompt_file:
        # Multiple prompts from file
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        all_outputs = []
        for prompt in prompts:
            outputs = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences,
            )
            all_outputs.extend(outputs)

            for output in outputs:
                print(output)
                print("-" * 40)

        if args.output_file:
            with open(args.output_file, "w") as f:
                for output in all_outputs:
                    f.write(output + "\n")
            logger.info(f"Outputs written to {args.output_file}")
    else:
        logger.error("Must specify --prompt, --prompt-file, or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
