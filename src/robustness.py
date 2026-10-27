"""Paired test on 20 held-out messages, preserving the original labels."""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import f1_score
from src.predict import load_model, predict_sms, InputValidationError


def slight_change(text):
    # Preserves core message content while simulating a different case and extra spaces.
    return '  ' + text.swapcase() + '  '


def run():
    holdout = pd.read_csv('metrics/holdout_private.csv', keep_default_na=False)
    spam = holdout[holdout.label == 'spam'].sample(n=10, random_state=42)
    ham = holdout[holdout.label == 'ham'].sample(n=10, random_state=42)
    sample = pd.concat([spam, ham], ignore_index=True)
    model = load_model()
    original = []
    modified = []
    for text in sample.message:
        original.append(predict_sms(text, model))
        modified.append(predict_sms(slight_change(text), model))
    report = {
        'number_of_test_objects': int(len(sample)),
        'change': 'swapcase and add leading/trailing whitespace',
        'changed_predictions': sum(a != b for a, b in zip(original, modified)),
        'changed_fraction': sum(a != b for a, b in zip(original, modified)) / len(sample),
        'f1_spam_before': float(f1_score(sample.label, original, pos_label='spam')),
        'f1_spam_after': float(f1_score(sample.label, modified, pos_label='spam')),
        'atypical_input': '🙂🙂🙂',
        'atypical_prediction': predict_sms('🙂🙂🙂', model),
        'atypical_interpretation': 'Вход допустим по схеме, но нет оснований считать его обычным англоязычным SMS или уверенно объявлять OOD.'
    }
    # Explicit controlled rejection as a separate stress case.
    try:
        predict_sms('A' * 1001, model)
    except InputValidationError as exc:
        report['oversized_input_rejected'] = str(exc)
    Path('metrics/robustness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    run()
