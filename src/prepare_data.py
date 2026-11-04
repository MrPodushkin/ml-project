"""Extract only the expected file, with size and integrity checks; never run dataset code."""

import argparse
import hashlib
import zipfile
from pathlib import Path

from src.data_utils import DATA_HASH_PATH, DATA_PATH


def prepare(archive):
    archive = Path(archive)
    if archive.stat().st_size > 1024 * 1024:
        raise ValueError("Архив превышает 1 MiB")
    with zipfile.ZipFile(archive) as source:
        entry = source.getinfo("SMSSpamCollection")
        if entry.file_size > 2 * 1024 * 1024:
            raise ValueError("Декодированные данные превышают 2 MiB")
        with source.open(entry) as stream:
            data = stream.read(2 * 1024 * 1024 + 1)
    if (
        len(data) > 2 * 1024 * 1024
        or hashlib.sha256(data).hexdigest() != DATA_HASH_PATH.read_text().split()[0]
    ):
        raise ValueError("Размер или SHA256 исходного файла не соответствует ЛР1")
    data.decode("utf-8")
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_bytes(data)
    print("Dataset extracted and SHA256 verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", default="data/sms_spam.zip")
    prepare(parser.parse_args().archive)
