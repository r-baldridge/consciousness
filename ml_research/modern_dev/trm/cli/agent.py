"""
TRM-Powered CLI Agent for Reasoning Tasks

A command-line agent that uses the Tiny Recursive Model for iterative
reasoning on puzzles and structured problems.

Usage:
    # Interactive mode
    python -m trm.cli.agent --interactive --task sudoku

    # Solve from file
    python -m trm.cli.agent --input puzzle.json --task sudoku

    # With trained model
    python -m trm.cli.agent --model checkpoints/best.pt --interactive

Example puzzle.json:
    {
        "puzzle": [[5,3,0,0,7,0,0,0,0],
                   [6,0,0,1,9,5,0,0,0],
                   ...],
        "type": "sudoku"
    }
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import torch

from ..src.model import TRM, TRMConfig


class TRMAgent:
    """
    CLI Agent powered by Tiny Recursive Model.

    Uses recursive neural reasoning for iterative problem solving.
    The agent "thinks" through multiple refinement steps before
    producing a final answer.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        task: str = "reasoning",
        device: str = "cpu",
        verbose: bool = False,
        max_steps: int = 16,
    ):
        """
        Initialize the TRM Agent.

        Args:
            model_path: Path to trained model checkpoint (optional)
            task: Task type (sudoku, maze, arc, reasoning)
            device: Device to run on (cpu, cuda, mps)
            verbose: Print detailed reasoning trajectory
            max_steps: Maximum reasoning steps
        """
        self.device = torch.device(device)
        self.verbose = verbose
        self.task = task
        self.max_steps = max_steps

        # Load or create model
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}...")
            self.model = TRM.from_pretrained(model_path).to(self.device)
        else:
            if model_path:
                print(f"Model not found at {model_path}, using untrained model")
            config = self._get_config_for_task(task)
            self.model = TRM(config).to(self.device)

        self.model.eval()

        if verbose:
            self._print_model_info()

    def _get_config_for_task(self, task: str) -> TRMConfig:
        """Get appropriate config for task type."""
        configs = {
            "sudoku": TRMConfig.for_sudoku,
            "maze": TRMConfig.for_maze,
            "arc": TRMConfig.for_arc_agi,
            "reasoning": TRMConfig.for_arc_agi,
        }
        return configs.get(task, TRMConfig)()

    def _print_model_info(self):
        """Print model information."""
        config = self.model.config
        print(f"\n{'='*50}")
        print(f"TRM Agent Initialized")
        print(f"{'='*50}")
        print(f"  Task:            {self.task}")
        print(f"  Parameters:      {self.model.num_parameters():,}")
        print(f"  Effective Depth: {config.effective_depth} layers")
        print(f"  Grid Size:       {config.grid_size}x{config.grid_size}")
        print(f"  T Cycles:        {config.T_cycles}")
        print(f"  n Cycles:        {config.n_cycles}")
        print(f"  Device:          {self.device}")
        print(f"{'='*50}\n")

    def solve(
        self,
        puzzle: torch.Tensor,
        show_trajectory: bool = False,
    ) -> Dict[str, Any]:
        """
        Solve a reasoning task with recursive refinement.

        Args:
            puzzle: Input puzzle tensor (batch, grid_size, grid_size) or (batch, seq_len)
            show_trajectory: Whether to track and return intermediate steps

        Returns:
            Dictionary with solution, confidence, steps, and trajectory
        """
        # Ensure batch dimension
        if puzzle.dim() == 2:
            puzzle = puzzle.unsqueeze(0)

        with torch.no_grad():
            result = self.model.solve(
                puzzle.to(self.device),
                max_steps=self.max_steps,
                return_trajectory=show_trajectory or self.verbose,
            )

        output = {
            "solution": result["solution"].cpu(),
            "confidence": float(result["confidence"].mean().item()),
            "steps": result["steps"],
            "converged": float(result["confidence"].mean().item()) > 0.0,
            "logits": result["logits"].cpu(),
        }

        if show_trajectory or self.verbose:
            output["trajectory"] = result.get("trajectory", [])
            if self.verbose:
                self._print_trajectory(output["trajectory"])

        return output

    def _print_trajectory(self, trajectory: List[Dict]):
        """Print reasoning trajectory for debugging."""
        print("\n┌─ Reasoning Trajectory ─────────────────────┐")
        for step in trajectory:
            conf = step['q_hat'].mean().item()
            bar = "█" * int(conf * 20 + 10) + "░" * (30 - int(conf * 20 + 10))
            print(f"│ Step {step['step']+1:2d}: [{bar}] q={conf:+.3f} │")
        print("└─────────────────────────────────────────────┘")

    def solve_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve puzzle from dictionary format.

        Args:
            data: Dict with 'puzzle' key containing nested list

        Returns:
            Solution dictionary
        """
        puzzle = torch.tensor(data["puzzle"], dtype=torch.long)
        result = self.solve(puzzle, show_trajectory=self.verbose)

        # Reshape solution to grid if needed
        grid_size = self.model.config.grid_size
        solution = result["solution"].squeeze(0)
        if solution.dim() == 1:
            solution = solution.view(grid_size, grid_size)

        return {
            "solution": solution.tolist(),
            "confidence": result["confidence"],
            "steps": result["steps"],
            "converged": result["converged"],
        }

    def format_grid(self, grid: torch.Tensor) -> str:
        """Format grid for display."""
        grid_size = self.model.config.grid_size
        if grid.dim() == 1:
            grid = grid.view(grid_size, grid_size)

        lines = []
        for i in range(grid_size):
            row = grid[i].tolist()
            if grid_size == 9:
                # Sudoku formatting
                formatted = []
                for j, val in enumerate(row):
                    if j > 0 and j % 3 == 0:
                        formatted.append("│")
                    formatted.append(str(val) if val != 0 else ".")
                line = " ".join(formatted)
                if i > 0 and i % 3 == 0:
                    lines.append("──────┼───────┼──────")
                lines.append(line)
            else:
                lines.append(" ".join(str(v) for v in row))

        return "\n".join(lines)

    def interactive_mode(self):
        """Run agent in interactive mode."""
        print(f"\n🧠 TRM Agent - Interactive Mode")
        print(f"   Task: {self.task}")
        print(f"   Type 'help' for commands, 'quit' to exit\n")

        commands = {
            "help": self._cmd_help,
            "info": self._cmd_info,
            "solve": self._cmd_solve,
            "demo": self._cmd_demo,
            "quit": None,
            "exit": None,
        }

        while True:
            try:
                cmd_line = input("agent> ").strip()
                if not cmd_line:
                    continue

                parts = cmd_line.split(maxsplit=1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if cmd in ("quit", "exit"):
                    print("Goodbye!")
                    break
                elif cmd in commands and commands[cmd]:
                    commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")

    def _cmd_help(self, args: str):
        """Print help message."""
        print("""
┌─ TRM Agent Commands ────────────────────────────┐
│                                                 │
│  help          Show this help message           │
│  info          Show model information           │
│  solve <file>  Solve puzzle from JSON file      │
│  demo          Run demo puzzle                  │
│  quit          Exit the agent                   │
│                                                 │
│ Puzzle JSON format:                             │
│   {"puzzle": [[5,3,0,...], ...]}                │
│                                                 │
└─────────────────────────────────────────────────┘
        """)

    def _cmd_info(self, args: str):
        """Print model information."""
        self._print_model_info()

    def _cmd_solve(self, args: str):
        """Solve puzzle from file."""
        if not args:
            print("Usage: solve <filename.json>")
            return

        filepath = Path(args.strip())
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            print(f"\nLoading puzzle from {filepath}...")
            print("\nInput:")
            puzzle = torch.tensor(data["puzzle"], dtype=torch.long)
            print(self.format_grid(puzzle))

            print("\nSolving...", end="", flush=True)
            result = self.solve_from_dict(data)

            print(f" done in {result['steps']} steps!\n")
            print("Solution:")
            print(self.format_grid(torch.tensor(result["solution"])))
            print(f"\nConfidence: {result['confidence']:.1%}")
            print(f"Converged: {'Yes' if result['converged'] else 'No'}")

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        except Exception as e:
            print(f"Error solving puzzle: {e}")

    def _cmd_demo(self, args: str):
        """Run a demo puzzle."""
        if self.task == "sudoku":
            # Easy Sudoku demo
            demo_puzzle = [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
            print("\nDemo: Easy Sudoku")
        else:
            # Generic grid demo
            grid_size = self.model.config.grid_size
            demo_puzzle = torch.randint(
                0, self.model.config.vocab_size,
                (grid_size, grid_size)
            ).tolist()
            print(f"\nDemo: Random {grid_size}x{grid_size} grid")

        print("\nInput:")
        puzzle = torch.tensor(demo_puzzle, dtype=torch.long)
        print(self.format_grid(puzzle))

        print("\nSolving...", end="", flush=True)
        result = self.solve_from_dict({"puzzle": demo_puzzle})

        print(f" done in {result['steps']} steps!\n")
        print("Solution (note: untrained model produces random output):")
        print(self.format_grid(torch.tensor(result["solution"])))
        print(f"\nConfidence: {result['confidence']:.1%}")


def create_agent(
    task: str = "reasoning",
    model_path: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = False,
) -> TRMAgent:
    """
    Factory function to create a TRM agent.

    Args:
        task: Task type (sudoku, maze, arc, reasoning)
        model_path: Path to trained model
        device: Device to run on
        verbose: Enable verbose output

    Returns:
        Configured TRMAgent instance
    """
    return TRMAgent(
        model_path=model_path,
        task=task,
        device=device,
        verbose=verbose,
    )


def main():
    """Main entry point for CLI agent."""
    parser = argparse.ArgumentParser(
        description="TRM-Powered CLI Agent for Reasoning Tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode for Sudoku
  python -m trm.cli.agent --interactive --task sudoku

  # Solve puzzle from file
  python -m trm.cli.agent --input puzzle.json --output solution.json

  # Use trained model with verbose output
  python -m trm.cli.agent --model best.pt --interactive --verbose

  # Quick solve (non-interactive)
  python -m trm.cli.agent --input puzzle.json --task sudoku
        """
    )

    parser.add_argument(
        "--task", "-t",
        choices=["sudoku", "maze", "arc", "reasoning"],
        default="reasoning",
        help="Task type (default: reasoning)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input puzzle file (JSON format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for solution (JSON format)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="Maximum reasoning steps (default: 16)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with reasoning trajectory"
    )

    args = parser.parse_args()

    # Create agent
    agent = TRMAgent(
        model_path=args.model,
        task=args.task,
        device=args.device,
        verbose=args.verbose,
        max_steps=args.max_steps,
    )

    # Run in appropriate mode
    if args.interactive:
        agent.interactive_mode()
    elif args.input:
        # Batch mode: solve from file
        try:
            with open(args.input) as f:
                data = json.load(f)

            result = agent.solve_from_dict(data)

            # Output results
            output_data = {
                "input": data.get("puzzle"),
                "solution": result["solution"],
                "confidence": result["confidence"],
                "steps": result["steps"],
                "converged": result["converged"],
            }

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"Solution written to {args.output}")
            else:
                print(json.dumps(output_data, indent=2))

        except FileNotFoundError:
            print(f"Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {args.input}: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        print("\nTip: Use --interactive or --input to run the agent")


if __name__ == "__main__":
    main()
