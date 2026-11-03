"""Load only the locally trained artifact after checking its manifest."""
from pathlib import Path
import joblib
from src.data_utils import ROOT, sha256_file

MODEL_PATH = ROOT / 'models/spam_pipeline.joblib'
MODEL_HASH_PATH = ROOT / 'model.sha256'

def load_model(model_path=MODEL_PATH, checksum_path=MODEL_HASH_PATH):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f'Файл модели не найден: {path}')
    expected, name = Path(checksum_path).read_text(encoding='utf-8').strip().split(maxsplit=1)
    if Path(name).name != path.name or sha256_file(path) != expected:
        raise ValueError('Контрольная сумма модели не совпадает с манифестом')
    # A checksum detects changes; it does not make an untrusted pickle safe.
    # This path is a locally trained, private artifact, never a user upload.
    return joblib.load(path)
