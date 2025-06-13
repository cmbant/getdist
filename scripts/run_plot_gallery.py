#!/usr/bin/env python3
"""
Script to run the plot gallery notebook and export it to HTML.

This script:
1. Executes the docs/plot_gallery.ipynb notebook
2. Exports it to HTML format
3. Replaces the existing docs/plot_gallery.html file

Requirements:
- jupyter or nbconvert
- All dependencies needed to run the notebook
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_repo_root():
    """Find the repository root directory."""
    current_dir = Path(__file__).parent
    while current_dir != current_dir.parent:
        if (current_dir / ".git").exists() or (current_dir / "pyproject.toml").exists():
            return current_dir
        current_dir = current_dir.parent
    raise RuntimeError("Could not find repository root")


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        # Check if nbconvert is available
        result = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print("Error: nbconvert is not available. Please install it with:")
            print("  pip install nbconvert")
            return False

        print(f"Found nbconvert: {result.stdout.strip()}")
        return True

    except FileNotFoundError:
        print("Error: Python is not available in PATH")
        return False


def run_and_export_notebook(notebook_path, output_path, execute=True):
    """
    Run the notebook and export it to HTML.

    Args:
        notebook_path: Path to the input notebook
        output_path: Path for the output HTML file
        execute: Whether to execute the notebook (default: True)
    """
    print(f"Processing notebook: {notebook_path}")
    print(f"Output will be saved to: {output_path}")

    # Build the nbconvert command
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "html",
        "--output",
        str(output_path),
    ]

    if execute:
        cmd.extend(["--execute", "--ExecutePreprocessor.timeout=600"])
        print("Executing notebook (this may take a while)...")
    else:
        print("Converting notebook without execution...")

    cmd.append(str(notebook_path))

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=notebook_path.parent)

        print("Conversion successful!")
        if result.stdout:
            print("Output:", result.stdout)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run plot gallery notebook and export to HTML")
    parser.add_argument(
        "--no-execute", action="store_true", help="Don't execute the notebook, just convert existing output"
    )
    parser.add_argument("--output", help="Output HTML file path (default: docs/plot_gallery.html)")

    args = parser.parse_args()

    # Find repository root and set paths
    try:
        repo_root = find_repo_root()
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    notebook_path = repo_root / "docs" / "plot_gallery.ipynb"
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = repo_root / "docs" / "plot_gallery.html"

    # Check if notebook exists
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return 1

    # Check dependencies
    if not check_dependencies():
        return 1

    # Run and export the notebook
    success = run_and_export_notebook(notebook_path, output_path, execute=not args.no_execute)

    if success:
        print(f"\nSuccess! HTML file created at: {output_path}")
        return 0
    else:
        print("\nFailed to convert notebook")
        return 1


if __name__ == "__main__":
    sys.exit(main())
