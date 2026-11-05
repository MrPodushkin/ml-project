"""Create a user-private runtime env file outside Git; never print credentials."""

import os
import secrets
from pathlib import Path


def main():
    folder = Path.home() / ".config" / "ml-project"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder / "lab4.env"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write("LAB4_API_TOKEN=" + secrets.token_urlsafe(32) + "\n")
            stream.write("LAB4_METRICS_TOKEN=" + secrets.token_urlsafe(32) + "\n")
    if path.stat().st_mode & 0o077:
        raise SystemExit("Runtime secret file permissions must be 600")
    print("Runtime secrets ready: ~/.config/ml-project/lab4.env (values hidden)")


if __name__ == "__main__":
    main()
