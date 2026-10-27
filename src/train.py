"""Train the reproducible TF-IDF + LogisticRegression baseline on CPU."""
from pathlib import Path
import json
import platform
import time
import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.data_utils import load_and_check, sha256_file
from src.predict import predict_sms

SEED = 42
TEST_SIZE = 0.20
METRIC_DIR = Path('metrics')
MODEL_FILE = Path('models/spam_pipeline.joblib')


def score(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=['ham', 'spam'], zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_spam': float(p[1]), 'recall_spam': float(r[1]),
        'f1_spam': float(f1[1]),
        'per_class': {
            'ham': {'precision': float(p[0]), 'recall': float(r[0]), 'f1': float(f1[0])},
            'spam': {'precision': float(p[1]), 'recall': float(r[1]), 'f1': float(f1[1])}
        }
    }


def train_and_evaluate():
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    df, data_report = load_and_check()
    Path('metrics/data_check.json').write_text(json.dumps(data_report, indent=2, ensure_ascii=False), encoding='utf-8')
    x_train, x_test, y_train, y_test = train_test_split(
        df['message'].astype(str), df['label'].astype(str), test_size=TEST_SIZE,
        random_state=SEED, stratify=df['label'])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced'))
    ])
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    seconds = time.perf_counter() - start
    predicted = pipeline.predict(x_test)
    majority = DummyClassifier(strategy='most_frequent')
    majority.fit(x_train.to_frame(), y_train)
    baseline_pred = majority.predict(x_test.to_frame())
    metrics = {
        'primary_metric': 'f1_spam',
        'primary_metric_value': score(y_test, predicted)['f1_spam'],
        'test_size': len(y_test), 'train_size': len(y_train),
        'dataset_sha256': data_report['dataset_sha256'],
        'dataset_rows_after_cleaning': len(df),
        'random_state': SEED, 'test_fraction': TEST_SIZE,
        'split': 'stratified random train/test; duplicate SMS removed before splitting',
        'model_name': 'TF-IDF(1-2grams)+LogisticRegression',
        'pipeline_steps': ['tfidf', 'classifier'],
        'vectorizer_parameters': {'ngram_range': [1, 2], 'min_df': 2, 'max_features': 30000},
        'classifier_parameters': {'max_iter': 1000, 'random_state': SEED, 'class_weight': 'balanced'},
        'train_seconds': round(seconds, 4),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__},
        'model_metrics': score(y_test, predicted),
        'most_frequent_baseline': score(y_test, baseline_pred),
        'classification_report': classification_report(y_test, predicted, output_dict=True, zero_division=0),
        'confusion_matrix_labels': ['ham', 'spam'],
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=['ham', 'spam']).tolist()
    }
    METRIC_DIR.joinpath('baseline.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    # These holdout texts are kept LOCALLY, ignored by Git, and never used to fit.
    holdout = pd.DataFrame({'message': x_test.to_numpy(), 'label': y_test.to_numpy(), 'prediction': predicted})
    holdout.to_csv(METRIC_DIR / 'holdout_private.csv', index=False)
    failures = holdout[holdout.label != holdout.prediction].head(5)
    # Avoid accidentally publishing text: keep wrong examples only in private local file.
    failures.to_csv(METRIC_DIR / 'errors_private.csv', index=False)
    joblib.dump(pipeline, MODEL_FILE)
    METRIC_DIR.joinpath('model.sha256').write_text(f'{sha256_file(MODEL_FILE)}  {MODEL_FILE}\n', encoding='utf-8')
    # Verify load and validated prediction.
    recovered = joblib.load(MODEL_FILE)
    sanity = predict_sms('Congratulations! You won a free prize', recovered)
    print('Загруженная модель отвечает:', sanity)
    print('Проверка данных:', data_report)
    print('Метрики:', json.dumps(metrics['model_metrics'], ensure_ascii=False, indent=2))
    print('Простой baseline:', json.dumps(metrics['most_frequent_baseline'], ensure_ascii=False, indent=2))
    print('Ошибочные предсказания (количество):', int((holdout.label != holdout.prediction).sum()))
    return metrics


if __name__ == '__main__':
    train_and_evaluate()
