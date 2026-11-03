"""Verify and evaluate the same artifact and exact split recorded by training."""
import json
from sklearn.metrics import confusion_matrix
from src.data_utils import ROOT, load_and_check, split_dataset, split_fingerprints, sha256_file
from src.model_utils import MODEL_PATH, load_model
from src.train import score, plot_confusion, save_json

def load_experiment():
    baseline = json.loads((ROOT/'metrics/baseline.json').read_text(encoding='utf-8'))
    df,report = load_and_check()
    train,test = split_dataset(df)
    if baseline['dataset_sha256'] != report['dataset_sha256'] or baseline['split_fingerprints'] != split_fingerprints(train,test):
        raise ValueError('Данные или split не соответствуют обучению')
    if baseline['model_sha256'] != sha256_file(MODEL_PATH):
        raise ValueError('Модель не соответствует baseline')
    return train,test,load_model(),baseline

def evaluate():
    train,test,model,baseline = load_experiment()
    predicted = model.predict(test.message.astype(str))
    metrics = score(test.label,predicted)
    if metrics != baseline['model_metrics']:
        raise ValueError('Метрики сохранённой модели отличаются от baseline')
    matrix = confusion_matrix(test.label,predicted,labels=['ham','spam'])
    plot_confusion(matrix)
    result = {'model_file':baseline['model_file'],'model_sha256':baseline['model_sha256'],
              'dataset_sha256':baseline['dataset_sha256'],'test_size':len(test),'metrics':metrics,
              'matches_baseline':True,'confusion_matrix':matrix.tolist()}
    save_json(ROOT/'metrics/evaluation.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    evaluate()
