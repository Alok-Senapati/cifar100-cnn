"""Launch the local CIFAR-100 model explorer."""

from __future__ import annotations

import argparse

from cifar100_cnn.explorer_server import serve_local_explorer


def main() -> None:
    """Parse local server options and start the browser interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="local HTTP port (default: 8765)")
    arguments = parser.parse_args()
    serve_local_explorer(port=arguments.port)


if __name__ == "__main__":
    main()
