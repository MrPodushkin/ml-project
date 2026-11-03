"""Canonical CPU baseline: normalized deduplication, stratified split, one model."""
import json
import platform
import time
import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from src.data_utils import ROOT, SEED, TEST_FRACTION, load_and_check, sha256_file, split_dataset, split_fingerprints
from src.model_utils import MODEL_PATH, MODEL_HASH_PATH, load_model
from src.predict_sms import predict_sms

METRIC_DIR = ROOT / 'metrics'

def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def score(y_true,y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true,y_pred,labels=['ham','spam'],zero_division=0)
    return {'accuracy':float(accuracy_score(y_true,y_pred)), 'precision_spam':float(p[1]),
            'recall_spam':float(r[1]),'f1_spam':float(f[1]),
            'per_class':{label:{'precision':float(p[i]),'recall':float(r[i]),'f1':float(f[i])}
                         for i,label in enumerate(['ham','spam'])}}

def plot_confusion(matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(matrix,display_labels=['ham','spam']).plot(cmap='Blues',values_format='d')
    plt.title('SMS baseline: held-out test')
    plt.tight_layout()
    plt.savefig(METRIC_DIR/'confusion_matrix.png',dpi=150)
    plt.close()

def train_and_evaluate():
    METRIC_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    df,report = load_and_check()
    train,test = split_dataset(df)
    model = Pipeline([
        ('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000)),
        ('classifier',LogisticRegression(max_iter=1000,random_state=SEED,class_weight='balanced'))])
    started = time.perf_counter()
    model.fit(train.message.astype(str),train.label.astype(str))
    seconds = time.perf_counter()-started
    predicted = model.predict(test.message.astype(str))
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(train[['message']],train.label)
    dummy_pred = dummy.predict(test[['message']])
    joblib.dump(model,MODEL_PATH)
    model_hash = sha256_file(MODEL_PATH)
    MODEL_HASH_PATH.write_text(f'{model_hash}  models/spam_pipeline.joblib\n',encoding='utf-8')
    matrix = confusion_matrix(test.label,predicted,labels=['ham','spam'])
    metrics = {'primary_metric':'f1_spam','primary_metric_value':score(test.label,predicted)['f1_spam'],
        'test_size':len(test),'train_size':len(train),'dataset_sha256':report['dataset_sha256'],
        'dataset_rows_after_cleaning':len(df),'random_state':SEED,'test_fraction':TEST_FRACTION,
        'split':'stratified random 80/20; normalized duplicates removed BEFORE splitting',
        'split_fingerprints':split_fingerprints(train,test),'normalized_train_test_overlap':0,
        'model_file':'models/spam_pipeline.joblib','model_sha256':model_hash,
        'model_name':'TF-IDF(1-2grams)+LogisticRegression','pipeline_steps':['tfidf','classifier'],
        'vectorizer_parameters':{'ngram_range':[1,2],'min_df':2,'max_features':30000},
        'classifier_parameters':{'max_iter':1000,'random_state':SEED,'class_weight':'balanced'},
        'train_seconds':round(seconds,4),'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'joblib':joblib.__version__},
        'model_metrics':score(test.label,predicted),'most_frequent_baseline':score(test.label,dummy_pred),
        'classification_report':classification_report(test.label,predicted,output_dict=True,zero_division=0),
        'confusion_matrix_labels':['ham','spam'],'confusion_matrix':matrix.tolist()}
    holdout = test.copy()
    holdout['prediction'] = predicted
    holdout.to_csv(METRIC_DIR/'holdout_private.csv',index=True,index_label='row_id')
    holdout[holdout.label!=holdout.prediction].to_csv(METRIC_DIR/'errors_private.csv',index=True,index_label='row_id')
    save_json(METRIC_DIR/'data_check.json',report)
    save_json(METRIC_DIR/'baseline.json',metrics)
    plot_confusion(matrix)
    recovered = load_model()
    print('Reload sanity:',predict_sms('Congratulations! You won a free prize',recovered))
    print(json.dumps({'train_size':len(train),'test_size':len(test),'metrics':metrics['model_metrics'],'dummy':metrics['most_frequent_baseline']},indent=2))
    return metrics

if __name__=='__main__':
    train_and_evaluate()
