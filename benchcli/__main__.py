"""Allow `python -m benchcli` as a compatibility entry point."""
from .cli import app

if __name__ == "__main__":
    app()
