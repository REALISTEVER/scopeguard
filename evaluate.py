"""ScopeGuard reproducibility repair; preserves original CP1 data and baseline.

No tuning occurs on the historical test set. Results are diagnostic, not a fresh
unseen evaluation: the test examples were previously inspected in checkpoints.
"""
import argparse, hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from original.baseline import classify, LABELS

ROOT = Path(__file__).resolve().parent

def metrics(y, pred):
    return {'cases':len(y),'accuracy':accuracy_score(y,pred),
            'macro_f1':f1_score(y,pred,labels=LABELS,average='macro',zero_division=0),
            'clarification_recall':recall_score(y,pred,labels=[LABELS[2]],average=None,zero_division=0)[0],
            'confusion_matrix':confusion_matrix(y,pred,labels=LABELS).tolist()}

def features():
    return FeatureUnion([
        ('word',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),
        ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True))])

def load_data():
    path=ROOT/'data/original.csv'
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['csv_sha256']:
        raise ValueError('Original CSV changed; do not silently modify the historical split.')
    df=pd.read_csv(path)
    assert len(df)==90 and df.id.is_unique
    assert not df.duplicated(['project_scope','new_request']).any()
    train=df[df.split=='train'].copy();test=df[df.split=='test'].copy()
    assert test.id.tolist()==manifest['test_ids'] and len(train)==72
    assert set(train.id).isdisjoint(test.id)
    assert set(zip(train.project_scope,train.new_request)).isdisjoint(zip(test.project_scope,test.new_request))
    return train,test,manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--embeddings',action='store_true')
    parser.add_argument('--output',default='results');args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    train,test,manifest=load_data()
    text=lambda d:np.array([f'scope: {s} request: {r}' for s,r in zip(d.project_scope,d.new_request)])
    xt,xh=text(train),text(test);yt,yh=train.label.to_numpy(),test.label.to_numpy()
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    models={
        'TF-IDF + logistic regression':Pipeline([('features',features()),('model',LogisticRegression(C=4.0,max_iter=3000,class_weight='balanced',random_state=42))]),
        'TF-IDF + linear SVM':Pipeline([('features',features()),('model',LinearSVC(C=1.0,class_weight='balanced',random_state=42,max_iter=10000))])}
    results=[];predictions=[];cv_results=[];embedding_info={'status':'not_requested'}
    def record(name,pred):
        result={'model':name,**metrics(yh,pred)};results.append(result)
        for (_,r),v in zip(test.iterrows(),pred):
            predictions.append({'model':name,'id':r.id,'domain':r.project_domain,'scope':r.project_scope,'request':r.new_request,'label':r.label,'prediction':v,'correct':r.label==v})
    record('Original Jaccard baseline',[classify(s,r)[0] for s,r in zip(test.project_scope,test.new_request)])
    for name,model in models.items():
        # TF-IDF vocabularies are refit inside each training-only fold.
        pred=cross_val_predict(model,xt,yt,cv=cv)
        cv_results.append({'model':name,**metrics(yt,pred)})
        model.fit(xt,yt);record(name,model.predict(xh))
    if args.embeddings:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import model_info
            torch.set_num_threads(2)
            model_id='sentence-transformers/all-MiniLM-L6-v2'
            revision_file=ROOT/'data/embedding_revision.txt'
            revision=revision_file.read_text().strip() if revision_file.exists() else model_info(model_id).sha
            revision_file.write_text(revision+'\n')
            encoder=SentenceTransformer(model_id,revision=revision,device='cpu',trust_remote_code=False)
            def encode(d):
                a=encoder.encode(d.project_scope.tolist(),normalize_embeddings=True,show_progress_bar=False)
                b=encoder.encode(d.new_request.tolist(),normalize_embeddings=True,show_progress_bar=False)
                return np.hstack([a,b,np.abs(a-b),a*b])
            et,eh=encode(train),encode(test)
            model=LogisticRegression(C=1.0,max_iter=3000,class_weight='balanced',random_state=42)
            name='MiniLM pair features + logistic regression'
            cv_results.append({'model':name,**metrics(yt,cross_val_predict(model,et,yt,cv=cv))})
            model.fit(et,yt);record(name,model.predict(eh))
            embedding_info={'status':'completed','model':model_id,'revision':revision,'frozen':True,'fine_tuning':False,'pair_features':'scope, request, absolute difference, elementwise product','dimension':int(et.shape[1]),'max_seq_length':encoder.max_seq_length}
        except Exception as e:
            embedding_info={'status':'blocked','error':f'{type(e).__name__}: {e}'}
            print('Embedding run blocked:',embedding_info['error'],file=sys.stderr)
    pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix'} for r in results]).to_csv(out/'metrics.csv',index=False)
    p=pd.DataFrame(predictions);p.to_csv(out/'predictions.csv',index=False);p[~p.correct].to_csv(out/'failures.csv',index=False)
    versions={}
    for package in ['numpy','pandas','scikit-learn','scipy','sentence-transformers','transformers','torch','huggingface-hub']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    payload={'dataset':manifest,'historical_test':results,'training_only_3fold_oof':cv_results,'embedding':embedding_info,
        'environment':{'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'packages':versions},
        'limitations':['Historical test was seen before; this is a diagnostic rerun, not a new blind test.','Only 6 test examples per class.','No independent human label review or added data is claimed.','No hyperparameter search was conducted in this repair.','Pretrained model pretraining-data overlap cannot be ruled out.']}
    (out/'run.json').write_text(json.dumps(payload,indent=2))
    print(pd.read_csv(out/'metrics.csv').to_string(index=False))
    print('Embedding status:',embedding_info['status'])

if __name__=='__main__':main()
