"""
TRM Inference CLI

Run inference with trained Tiny Recursive Model.

Usage:
    python -m trm.cli.infer --model checkpoints/best_model.pt --input puzzle.txt
    python -m trm.cli.infer --model checkpoints/best_model.pt --interactive
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F

from ..src.model import TRM, TRMConfig


def load_puzzle_from_file(path: str, grid_size: int) -> torch.Tensor:
    """Load puzzle from text file.

    Format: Space/comma separated digits, one row per line.
    Use 0 for empty cells.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    puzzle = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Parse row
        row = []
        for char in line.replace(",", " ").split():
            if char.isdigit():
                row.append(int(char))
            elif char == ".":
                row.append(0)
        if row:
            puzzle.append(row)

    # Validate
    if len(puzzle) != grid_size:
        raise ValueError(f"Expected {grid_size} rows, got {len(puzzle)}")
    for i, row in enumerate(puzzle):
        if len(row) != grid_size:
            raise ValueError(f"Row {i} has {len(row)} elements, expected {grid_size}")

    return torch.tensor(puzzle, dtype=torch.long)


def format_grid(grid: torch.Tensor, grid_size: int) -> str:
    """Format grid for display."""
    lines = []
    for i in range(grid_size):
        row = grid[i].tolist() if grid.dim() == 2 else grid[i * grid_size:(i + 1) * grid_size].tolist()
        if grid_size == 9:
            # Sudoku formatting with box separators
            formatted = []
            for j, val in enumerate(row):
                if j > 0 and j % 3 == 0:
                    formatted.append("|")
                formatted.append(str(val) if val != 0 else ".")
            line = " ".join(formatted)
            if i > 0 and i % 3 == 0:
                lines.append("-" * len(line))
            lines.append(line)
        else:
            # Generic grid formatting
            lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines)


def run_interactive(model: TRM, device: torch.device):
    """Run interactive inference mode."""
    grid_size = model.config.grid_size

    print("\n=== TRM Interactive Mode ===")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Enter puzzle row by row (use 0 or . for empty cells)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            print("\nEnter puzzle:")
            puzzle = []
            for i in range(grid_size):
                line = input(f"  Row {i + 1}: ").strip()
                if line.lower() == "quit":
                    return

                row = []
                for char in line.replace(",", " ").split():
                    if char.isdigit():
                        row.append(int(char))
                    elif char == ".":
                        row.append(0)

                if len(row) != grid_size:
                    print(f"  Error: Expected {grid_size} values, got {len(row)}")
                    break
                puzzle.append(row)

            if len(puzzle) != grid_size:
                continue

            # Convert to tensor
            puzzle_tensor = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0).to(device)

            # Solve
            print("\nSolving...")
            result = model.solve(puzzle_tensor, return_trajectory=True)

            # Display result
            solution = result["solution"].squeeze(0)
            if solution.dim() == 1:
                solution = solution.view(grid_size, grid_size)

            print(f"\nSolution (found in {result['steps']} steps, "
                  f"confidence: {result['confidence'].item():.1%}):\n")
            print(format_grid(solution, grid_size))

            # Show trajectory if requested
            show_traj = input("\nShow trajectory? (y/n): ").strip().lower()
            if show_traj == "y":
                for step_info in result["trajectory"]:
                    step_pred = step_info["prediction"].squeeze(0)
                    if step_pred.dim() == 1:
                        step_pred = step_pred.view(grid_size, grid_size)
                    print(f"\n--- Step {step_info['step'] + 1} "
                          f"(q_hat: {step_info['q_hat'].item():.3f}) ---")
                    print(format_grid(step_pred, grid_size))

        except KeyboardInterrupt:
            print("\n\nExiting...")
            return
        except Exception as e:
            print(f"Error: {e}")


def run_batch(
    model: TRM,
    input_path: str,
    output_path: Optional[str],
    device: torch.device,
):
    """Run batch inference on file."""
    # Load puzzle
    puzzle = load_puzzle_from_file(input_path, model.config.grid_size)
    puzzle = puzzle.unsqueeze(0).to(device)

    # Solve
    result = model.solve(puzzle, return_trajectory=True)

    # Format output
    solution = result["solution"].squeeze(0)
    grid_size = model.config.grid_size
    if solution.dim() == 1:
        solution = solution.view(grid_size, grid_size)

    print(f"\nInput puzzle:")
    print(format_grid(puzzle.squeeze(0).cpu(), grid_size))
    print(f"\nSolution (steps: {result['steps']}, "
          f"confidence: {result['confidence'].item():.1%}):")
    print(format_grid(solution.cpu(), grid_size))

    # Save output if requested
    if output_path:
        output = {
            "input": puzzle.squeeze(0).cpu().tolist(),
            "solution": solution.cpu().tolist(),
            "steps": result["steps"],
            "confidence": result["confidence"].item(),
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nOutput saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TRM Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, help="Input puzzle file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-steps", type=int, default=16, help="Maximum inference steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    device = torch.device(args.device)
    model = TRM.from_pretrained(args.model).to(device)
    model.eval()

    print(f"Model loaded: {model.num_parameters():,} parameters")
    print(f"Grid size: {model.config.grid_size}")
    print(f"Effective depth: {model.config.effective_depth}")

    # Run inference
    if args.interactive:
        run_interactive(model, device)
    elif args.input:
        run_batch(model, args.input, args.output, device)
    else:
        print("Error: Specify --input or --interactive")
        return 1


if __name__ == "__main__":
    main()
