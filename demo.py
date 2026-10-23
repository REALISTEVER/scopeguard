"""Interactive comparison demo. Decisions require human review; nothing is approved."""
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from evaluate import ROOT, load_data, features
from original.baseline import classify

def main():
    p=argparse.ArgumentParser();p.add_argument('--scope');p.add_argument('--request');p.add_argument('--embeddings',action='store_true');args=p.parse_args()
    scope=args.scope or input('Project scope: ').strip()
    request=args.request or input('New request: ').strip()
    if not scope.strip() or not request.strip():p.error('Both scope and request must contain text.')
    train,_,_=load_data()
    model=Pipeline([('features',features()),('model',LogisticRegression(C=4.0,max_iter=3000,class_weight='balanced',random_state=42))])
    model.fit([f'scope: {s} request: {r}' for s,r in zip(train.project_scope,train.new_request)],train.label)
    print('Original Jaccard baseline:',classify(scope,request)[0])
    print('TF-IDF logistic regression:',model.predict([f'scope: {scope} request: {request}'])[0])
    if args.embeddings:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        revision=(ROOT/'data/embedding_revision.txt').read_text().strip()
        encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',revision=revision,device='cpu',trust_remote_code=False)
        def encode(scopes,requests):
            a=encoder.encode(scopes,normalize_embeddings=True,show_progress_bar=False)
            b=encoder.encode(requests,normalize_embeddings=True,show_progress_bar=False)
            return np.hstack([a,b,np.abs(a-b),a*b])
        clf=LogisticRegression(C=1.0,max_iter=3000,class_weight='balanced',random_state=42)
        clf.fit(encode(train.project_scope.tolist(),train.new_request.tolist()),train.label)
        print('MiniLM pair features + logistic regression:',clf.predict(encode([scope],[request]))[0])
    print('REVIEW REQUIRED: experimental predictions only. No scope change is approved or saved.')

if __name__=='__main__':main()
