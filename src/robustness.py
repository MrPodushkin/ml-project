"""Paired perturbation experiment on 20 genuinely held-out SMS; no model substitutes."""

import json
import re

import pandas as pd

from src.data_utils import ROOT, SEED, normalized_keys
from src.evaluate_model import load_experiment
from src.predict_sms import InputValidationError, predict_sms
from src.train import save_json, score

REPLACEMENTS = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}


def case_spaces(text):
    return "  " + text.swapcase() + "  "


def small_typo(text):
    # Exactly one character replacement, preferably inside a word.
    for match in re.finditer(r"\b[A-Za-z]{4,}\b", text):
        for index in range(match.start() + 1, match.end() - 1):
            replacement = REPLACEMENTS.get(text[index].lower())
            if replacement:
                return text[:index] + replacement + text[index + 1 :]
    for index, char in enumerate(text):
        if char.lower() in REPLACEMENTS:
            return text[:index] + REPLACEMENTS[char.lower()] + text[index + 1 :]
    return text + " "


def leet_stress(text):
    # Additional strong stress case; not represented as a small natural typo.
    return "".join(REPLACEMENTS.get(char.lower(), char) for char in text)


def run():
    train, test, model, baseline = load_experiment()
    # Selection uses true labels only, never prediction success or robustness results.
    sample = pd.concat(
        [
            test[test.label == "spam"].sample(n=10, random_state=SEED),
            test[test.label == "ham"].sample(n=10, random_state=SEED),
        ]
    )
    overlap = set(normalized_keys(sample.message)) & set(normalized_keys(train.message))
    if overlap:
        raise ValueError("Выборка эксперимента пересекается с train")
    before = [predict_sms(str(t), model) for t in sample.message]
    experiments = {}
    records = []
    for name, change in [
        ("case_spaces", case_spaces),
        ("single_typo", small_typo),
        ("leet_stress", leet_stress),
    ]:
        changed_texts = [change(str(t)) for t in sample.message]
        after = [predict_sms(t, model) for t in changed_texts]
        changed = sum(a != b for a, b in zip(before, after))
        experiments[name] = {
            "changed_predictions": changed,
            "changed_fraction": changed / len(sample),
            "before": score(sample.label, before),
            "after": score(sample.label, after),
        }
        for row_id, original, modified, label, pred0, pred1 in zip(
            sample.index, sample.message, changed_texts, sample.label, before, after
        ):
            records.append(
                {
                    "row_id": int(row_id),
                    "experiment": name,
                    "original": original,
                    "modified": modified,
                    "label": label,
                    "prediction_before": pred0,
                    "prediction_after": pred1,
                }
            )
    atypical = "🙂🙂🙂"
    atypical_prediction = predict_sms(atypical, model)
    vector = model.named_steps["tfidf"].transform([atypical])
    try:
        predict_sms("A" * 1001, model)
    except InputValidationError as exc:
        oversized = str(exc)
    report = {
        "sample_size": len(sample),
        "sample_ids": [int(i) for i in sample.index],
        "selection": "10 spam + 10 ham from the canonical held-out test; selection fixed before prediction",
        "seed": SEED,
        "sample_train_overlap": len(overlap),
        "model_file": baseline["model_file"],
        "model_sha256": baseline["model_sha256"],
        "dataset_sha256": baseline["dataset_sha256"],
        "split_fingerprints": baseline["split_fingerprints"],
        "experiments": experiments,
        "atypical_input": {
            "case": "emoji-only synthetic SMS",
            "prediction": atypical_prediction,
            "tfidf_nonzero_features": int(vector.nnz),
            "interpretation": "Форма допустима, но словарных признаков нет; ответ не доказывает применимость модели. OOD не утверждается.",
        },
        "oversized_input_rejected": oversized,
        "conclusion": "Результаты описывают только эти 20 объектов и выбранные изменения. Регистр и пробелы нормализуются TF-IDF; общая устойчивость не доказана.",
        "proposed_measure": "Проверить нормализацию типичных замен букв на отдельной validation/CV внутри train; не настраивать защиту по test. Для входов без словарных признаков рассмотреть отказ с ручной обработкой.",
    }
    pd.DataFrame(records).to_csv(ROOT / "metrics/robustness_private.csv", index=False)
    save_json(ROOT / "metrics/robustness.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run()
