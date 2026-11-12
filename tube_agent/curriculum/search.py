"""Command-line entry point for distributed prefix search."""
import argparse
from pathlib import Path

from tube_agent.posttrain import Forest, node_key, run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    run_search(args.run.resolve(), args.model)


if __name__ == "__main__":
    main()
