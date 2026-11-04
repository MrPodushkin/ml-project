"""Install one pinned official Trivy binary after verifying its SHA256."""

import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from ci.common import read_json
from src.data_utils import ROOT


def main():
    config = read_json(ROOT / "config/tools.json")
    target = Path.home() / ".local/bin/trivy"
    if (
        target.is_file()
        and hashlib.sha256(target.read_bytes()).hexdigest() == config["trivy_binary_sha256"]
    ):
        print(f"Verified cached Trivy {config['trivy_version']} binary SHA256")
        return
    with urllib.request.urlopen(config["trivy_linux_amd64_url"], timeout=90) as response:
        data = response.read(150 * 1024 * 1024 + 1)
    if (
        len(data) > 150 * 1024 * 1024
        or hashlib.sha256(data).hexdigest() != config["trivy_linux_amd64_sha256"]
    ):
        raise ValueError("Trivy download size or SHA256 mismatch")
    with tempfile.TemporaryDirectory() as folder:
        archive = Path(folder) / "trivy.tar.gz"
        archive.write_bytes(data)
        with tarfile.open(archive) as source:
            entry = source.getmember("trivy")
            if not entry.isfile() or entry.size > 200 * 1024 * 1024:
                raise ValueError("Invalid Trivy archive entry")
            with source.extractfile(entry) as stream:
                binary = stream.read()
    if hashlib.sha256(binary).hexdigest() != config["trivy_binary_sha256"]:
        raise ValueError("Trivy binary SHA256 mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(binary)
    target.chmod(0o755)
    print(
        f"Installed official Trivy {config['trivy_version']} at {target}; archive SHA256 verified"
    )


if __name__ == "__main__":
    main()
