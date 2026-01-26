"""
Batch Processing Example

Process multiple code files in parallel using the ML Research pipeline.

Features demonstrated:
- Batch file processing
- Parallel execution
- Progress tracking
- Error handling for multiple files
- Result aggregation and reporting

Usage:
    python -m modern_dev.examples.batch_processing
    python -m modern_dev.examples.batch_processing --input ./src --output ./fixed
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modern_dev.orchestrator.router import MLOrchestrator, Request, Response


@dataclass
class FileResult:
    """Result of processing a single file."""
    file_path: Path
    success: bool
    changes_count: int = 0
    architecture_used: str = ""
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    output: Any = None


@dataclass
class BatchResult:
    """Result of processing a batch of files."""
    total_files: int
    successful: int
    failed: int
    total_changes: int
    total_time_ms: float
    results: List[FileResult] = field(default_factory=list)


def read_file(path: Path) -> str:
    """Read file contents safely."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise IOError(f"Failed to read {path}: {e}")


def write_file(path: Path, content: str) -> None:
    """Write file contents safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def count_changes(original: str, fixed: str) -> int:
    """Count number of line changes between original and fixed code."""
    orig_lines = original.splitlines()
    fixed_lines = fixed.splitlines()

    changes = 0
    max_lines = max(len(orig_lines), len(fixed_lines))

    for i in range(max_lines):
        orig_line = orig_lines[i] if i < len(orig_lines) else ""
        fixed_line = fixed_lines[i] if i < len(fixed_lines) else ""
        if orig_line != fixed_line:
            changes += 1

    return changes


def process_single_file(
    orchestrator: MLOrchestrator,
    file_path: Path,
    output_dir: Optional[Path] = None,
    line_range: Optional[tuple] = None,
) -> FileResult:
    """
    Process a single file through the repair pipeline.

    Args:
        orchestrator: MLOrchestrator instance
        file_path: Path to the file to process
        output_dir: Optional output directory for fixed files
        line_range: Optional (start, end) line range to process

    Returns:
        FileResult with processing outcome
    """
    start_time = time.time()

    try:
        # Read file
        content = read_file(file_path)

        # Optionally extract line range
        if line_range:
            lines = content.splitlines()
            start, end = line_range
            content = "\n".join(lines[start:end])

        # Process through orchestrator
        response = orchestrator.process(Request(
            task_type="code_repair",
            input_data={
                "buggy_code": content,
                "file_path": str(file_path),
            },
        ))

        execution_time = (time.time() - start_time) * 1000

        if response.success:
            # Get fixed code from output
            fixed_code = response.output.get("fixed_code", content) if isinstance(response.output, dict) else content

            # Count changes
            changes = count_changes(content, fixed_code)

            # Write output if directory specified
            if output_dir and changes > 0:
                output_path = output_dir / file_path.name
                write_file(output_path, fixed_code)

            return FileResult(
                file_path=file_path,
                success=True,
                changes_count=changes,
                architecture_used=response.architecture_used,
                execution_time_ms=execution_time,
                output=response.output,
            )
        else:
            return FileResult(
                file_path=file_path,
                success=False,
                architecture_used=response.architecture_used,
                execution_time_ms=execution_time,
                error=str(response.metadata.get("error", "Unknown error")),
            )

    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return FileResult(
            file_path=file_path,
            success=False,
            execution_time_ms=execution_time,
            error=str(e),
        )


def process_batch(
    files: List[Path],
    output_dir: Optional[Path] = None,
    max_workers: int = 4,
    show_progress: bool = True,
) -> BatchResult:
    """
    Process multiple files in parallel.

    Args:
        files: List of file paths to process
        output_dir: Optional output directory for fixed files
        max_workers: Maximum parallel workers
        show_progress: Whether to show progress updates

    Returns:
        BatchResult with aggregated results
    """
    start_time = time.time()

    # Create orchestrator once (shared across threads)
    orchestrator = MLOrchestrator()

    results: List[FileResult] = []
    successful = 0
    failed = 0
    total_changes = 0

    if show_progress:
        print(f"Processing {len(files)} files with {max_workers} workers...")

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_single_file,
                orchestrator,
                f,
                output_dir,
            ): f
            for f in files
        }

        # Collect results as they complete
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]

            try:
                result = future.result()
                results.append(result)

                if result.success:
                    successful += 1
                    total_changes += result.changes_count

                    if show_progress:
                        status = f"[OK] {result.changes_count} changes"
                        print(f"  [{i}/{len(files)}] {file_path.name}: {status}")
                else:
                    failed += 1

                    if show_progress:
                        print(f"  [{i}/{len(files)}] {file_path.name}: [FAIL] {result.error}")

            except Exception as e:
                failed += 1
                results.append(FileResult(
                    file_path=file_path,
                    success=False,
                    error=str(e),
                ))

                if show_progress:
                    print(f"  [{i}/{len(files)}] {file_path.name}: [ERROR] {e}")

    total_time = (time.time() - start_time) * 1000

    return BatchResult(
        total_files=len(files),
        successful=successful,
        failed=failed,
        total_changes=total_changes,
        total_time_ms=total_time,
        results=results,
    )


def find_python_files(directory: Path, recursive: bool = True) -> List[Path]:
    """Find all Python files in a directory."""
    pattern = "**/*.py" if recursive else "*.py"
    return sorted(directory.glob(pattern))


def print_report(batch_result: BatchResult) -> None:
    """Print a summary report of batch processing."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING REPORT")
    print("=" * 60)

    print(f"\nSummary:")
    print(f"  Total files:     {batch_result.total_files}")
    print(f"  Successful:      {batch_result.successful}")
    print(f"  Failed:          {batch_result.failed}")
    print(f"  Total changes:   {batch_result.total_changes}")
    print(f"  Total time:      {batch_result.total_time_ms:.2f}ms")

    if batch_result.total_files > 0:
        avg_time = batch_result.total_time_ms / batch_result.total_files
        print(f"  Avg time/file:   {avg_time:.2f}ms")

    # Detailed results
    if batch_result.results:
        print("\nDetailed Results:")
        print("-" * 60)

        for result in batch_result.results:
            status = "OK" if result.success else "FAIL"
            print(f"  {result.file_path.name}:")
            print(f"    Status:       {status}")
            print(f"    Architecture: {result.architecture_used or 'N/A'}")
            print(f"    Time:         {result.execution_time_ms:.2f}ms")

            if result.success:
                print(f"    Changes:      {result.changes_count}")
            else:
                print(f"    Error:        {result.error}")

            print()

    # Architecture usage breakdown
    arch_usage: Dict[str, int] = {}
    for result in batch_result.results:
        if result.architecture_used:
            arch_usage[result.architecture_used] = arch_usage.get(result.architecture_used, 0) + 1

    if arch_usage:
        print("Architecture Usage:")
        for arch, count in sorted(arch_usage.items(), key=lambda x: -x[1]):
            pct = count / batch_result.total_files * 100
            print(f"  {arch}: {count} ({pct:.1f}%)")

    print("=" * 60)


def demo_batch_processing() -> None:
    """Demo batch processing with synthetic files."""
    print("=" * 60)
    print("Demo: Batch Processing with Synthetic Files")
    print("=" * 60)

    # Create synthetic buggy files
    synthetic_files = [
        ("file1.py", "def add(a, b): return a + c"),
        ("file2.py", "def greet(name): retrun 'Hello, ' + name"),
        ("file3.py", "def divide(a, b): return a / b"),
        ("file4.py", "def square(x): return x ** 22"),  # Wrong exponent
        ("file5.py", "def concat(a, b): return a + b  # Missing type check"),
    ]

    # Create temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Write synthetic files
        for name, content in synthetic_files:
            (tmp_path / name).write_text(content)

        # Find files
        files = find_python_files(tmp_path)
        print(f"\nFound {len(files)} files to process\n")

        # Process batch
        result = process_batch(
            files,
            output_dir=tmp_path / "fixed",
            max_workers=2,
            show_progress=True,
        )

        # Print report
        print_report(result)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Batch process code files for repair"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input directory containing Python files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for fixed files"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process directories recursively"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic files"
    )

    args = parser.parse_args()

    if args.demo or not args.input:
        # Run demo mode
        demo_batch_processing()
    else:
        # Process specified directory
        if not args.input.exists():
            print(f"Error: Input path does not exist: {args.input}")
            sys.exit(1)

        if args.input.is_file():
            files = [args.input]
        else:
            files = find_python_files(args.input, args.recursive)

        if not files:
            print(f"No Python files found in: {args.input}")
            sys.exit(1)

        print(f"Found {len(files)} Python files")

        result = process_batch(
            files,
            output_dir=args.output,
            max_workers=args.workers,
            show_progress=True,
        )

        print_report(result)

        # Exit with error code if any failures
        sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
