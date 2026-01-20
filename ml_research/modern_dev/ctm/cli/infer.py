#!/usr/bin/env python3
"""
CTM Inference CLI

Command-line interface for running inference with Continuous Thought Machine models.

Usage:
    python -m ctm.cli.infer --checkpoint model.pt --input image.jpg
    python -m ctm.cli.infer --checkpoint model.pt --input-dir images/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load CTM model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on.

    Returns:
        Loaded CTM model.
    """
    import torch
    from ..src.model import CTM

    logger.info(f"Loading model from {checkpoint_path}")
    model = CTM.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {num_params:,} parameters")

    return model


def preprocess_input(
    input_path: str,
    config: Optional[Dict[str, Any]] = None,
):
    """Preprocess input for inference.

    Args:
        input_path: Path to input file.
        config: Optional preprocessing configuration.

    Returns:
        Preprocessed tensor.

    Note:
        This is a placeholder. Actual implementation would handle
        different input types (images, text, etc.).
    """
    import torch

    # Placeholder: Return dummy tensor
    # In real implementation, load and preprocess actual input
    logger.warning("Using placeholder preprocessing. Implement actual input handling.")

    input_dim = config.get("input_dim", 768) if config else 768
    return torch.randn(1, input_dim)


def run_inference(
    model,
    inputs,
    config: Dict[str, Any],
    device: str,
    return_intermediates: bool = False,
) -> Dict[str, Any]:
    """Run inference on inputs.

    Args:
        model: CTM model.
        inputs: Preprocessed input tensor.
        config: Inference configuration.
        device: Device to run on.
        return_intermediates: Whether to return intermediate states.

    Returns:
        Dictionary containing predictions and metadata.
    """
    import torch

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(
            inputs,
            num_steps=config.get("num_steps"),
            return_intermediates=return_intermediates,
        )

    # Convert tensors to lists for JSON serialization
    result = {
        "logits": outputs["output"].cpu().tolist(),
        "num_steps_used": outputs["num_steps_used"],
    }

    # Add predictions
    if config.get("task_type") == "classification":
        probs = torch.softmax(outputs["output"], dim=-1)
        pred_class = probs.argmax(dim=-1).item()
        pred_prob = probs.max(dim=-1).values.item()
        result["prediction"] = {
            "class": pred_class,
            "confidence": pred_prob,
            "top_k": get_top_k_predictions(probs, k=config.get("top_k", 5)),
        }

    # Add sync features if requested
    if config.get("return_sync_features", False):
        result["sync_features"] = outputs["sync_features"].cpu().tolist()

    # Add intermediates if requested
    if return_intermediates and "intermediates" in outputs:
        result["intermediates"] = [
            {
                "step": s["step"],
                "activation_norm": s["activations"].norm().item(),
            }
            for s in outputs["intermediates"]
        ]

    return result


def get_top_k_predictions(
    probs,
    k: int = 5,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Get top-k predictions with probabilities.

    Args:
        probs: Probability tensor (1, num_classes).
        k: Number of top predictions to return.
        class_names: Optional list of class names.

    Returns:
        List of top-k predictions.
    """
    import torch

    probs = probs.squeeze(0)  # (num_classes,)
    top_probs, top_indices = probs.topk(k)

    predictions = []
    for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
        pred = {
            "rank": i + 1,
            "class_id": idx,
            "probability": prob,
        }
        if class_names and idx < len(class_names):
            pred["class_name"] = class_names[idx]
        predictions.append(pred)

    return predictions


def batch_inference(
    model,
    input_paths: List[str],
    config: Dict[str, Any],
    device: str,
) -> List[Dict[str, Any]]:
    """Run batch inference on multiple inputs.

    Args:
        model: CTM model.
        input_paths: List of input file paths.
        config: Inference configuration.
        device: Device to run on.

    Returns:
        List of results for each input.
    """
    results = []
    for input_path in input_paths:
        logger.info(f"Processing: {input_path}")
        inputs = preprocess_input(input_path, config)
        result = run_inference(model, inputs, config, device)
        result["input_path"] = str(input_path)
        results.append(result)
    return results


def visualize_thinking(
    result: Dict[str, Any],
    output_path: Optional[str] = None,
) -> None:
    """Visualize CTM thinking process.

    Shows how synchronization patterns evolve over internal time steps.

    Args:
        result: Inference result with intermediates.
        output_path: Optional path to save visualization.
    """
    if "intermediates" not in result:
        logger.warning("No intermediates available for visualization")
        return

    # Placeholder: Print text-based visualization
    logger.info("Thinking process visualization:")
    for step_info in result["intermediates"]:
        step = step_info["step"]
        norm = step_info["activation_norm"]
        bar = "#" * int(norm * 10)
        print(f"  Step {step:2d}: {bar} ({norm:.3f})")

    logger.info(f"Total steps used: {result['num_steps_used']}")

    # In real implementation, create matplotlib visualization
    if output_path:
        logger.info(f"Visualization would be saved to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup device
    import torch
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load model
    model = load_model(args.checkpoint, device)

    # Build inference config
    config = {
        "num_steps": args.num_steps,
        "task_type": args.task_type,
        "top_k": args.top_k,
        "return_sync_features": args.return_sync,
        "input_dim": args.input_dim,
    }

    # Determine inputs
    if args.input:
        input_paths = [args.input]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        input_paths = list(input_dir.glob("*"))
        input_paths = [p for p in input_paths if p.is_file()]
    else:
        logger.error("No input specified. Use --input or --input-dir")
        sys.exit(1)

    # Run inference
    logger.info(f"Running inference on {len(input_paths)} input(s)")
    results = batch_inference(
        model,
        [str(p) for p in input_paths],
        config,
        device,
    )

    # Visualize thinking process if requested
    if args.visualize and len(results) == 1:
        # Re-run with intermediates for visualization
        inputs = preprocess_input(str(input_paths[0]), config)
        viz_result = run_inference(
            model, inputs, config, device,
            return_intermediates=True,
        )
        visualize_thinking(viz_result, args.viz_output)

    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        # Print to stdout
        print(json.dumps(results, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run inference with Continuous Thought Machine (CTM) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Input arguments
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to single input file",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory of input files",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=768,
        help="Input dimension (for placeholder preprocessing)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    # Inference configuration
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of internal time steps (default: use model config)",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "regression", "embedding"],
        help="Task type for output formatting",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to return (for classification)",
    )
    parser.add_argument(
        "--return-sync",
        action="store_true",
        help="Include synchronization features in output",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize thinking process",
    )
    parser.add_argument(
        "--viz-output",
        type=str,
        default=None,
        help="Path to save visualization",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
