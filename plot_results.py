"""Draw labeled diagnostic confusion matrices from actual evaluation outputs."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import ROOT
from original.baseline import LABELS

r=json.loads((ROOT/'results/run.json').read_text())['historical_test']
fig,axes=plt.subplots(2,2,figsize=(12,10))
for ax,row in zip(axes.flat,r):
    matrix=row['confusion_matrix'];ax.imshow(matrix,cmap='Blues',vmin=0,vmax=6)
    ax.set_xticks(range(3),['IN','OUT','CLARIFY']);ax.set_yticks(range(3),['IN','OUT','CLARIFY'])
    ax.set_xlabel('Predicted class');ax.set_ylabel('Actual class')
    ax.set_title(row['model'],fontsize=11)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(matrix[i][j]),ha='center',va='center',color='white' if matrix[i][j]>3 else 'black')
for ax in axes.flat[len(r):]:ax.set_visible(False)
fig.suptitle('Historical Checkpoint 1 test: 18 examples, 6 per class\nFixed 72-row training set; no hyperparameter search in this repair',fontsize=13)
fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(ROOT/'results/confusion_matrices.png',dpi=160)
