import argparse
import sys
from loguru import logger
from animax import get_version


def main() -> None:
    """
    anima CLI entry point. Supports 'version' command to print the current version.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="anima: Your personal anima foundry"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # version command
    subparsers.add_parser("version", help="Show anima version")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "version":
        version: str = get_version()
        print(f"anima version: {version}")
        logger.info(f"Displayed anima version: {version}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error(f"CLI failed: {exc}")
        sys.exit(1)