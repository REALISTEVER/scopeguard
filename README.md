# ScopeGuard
Arthur Cox III | COMP365 | North Carolina A&T State University

## Quick demonstration
Open ScopeGuard_Quick_Demo.ipynb in Google Colab and run its single code cell.
The notebook embeds the original dataset and demo code. It uses NumPy, pandas,
SciPy and scikit-learn available in standard Colab runtimes, with no model downloads.
Change the scope and request strings at the top to try other inputs.
This demonstration runs Jaccard and TF-IDF logistic regression on CPU.

## Reproduce the evaluation
The recorded environment is Python 3.12.14 on Linux, using CPU execution.
Use a fresh Python 3.12 virtual environment and run from this directory:

```sh
python -m pip install -r requirements.txt
python check_integrity.py
python evaluate.py --output reproduced_results
python demo.py
```

For the MiniLM comparison:

```sh
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-embeddings.txt
python evaluate.py --embeddings --output reproduced_results
python demo.py --embeddings
```

The first embedding run downloads the pretrained model. Its revision is recorded
in data/embedding_revision.txt. The encoder is frozen and uses two CPU threads.
The Python 3.13 Colab embedding run encountered torchvision/timm dependency errors;
the supplied simplified notebook avoids those dependencies. Saved four-model
results were generated in the recorded Python 3.12 environment.

## Files
- data/original.csv: 90 synthetic examples with the original 72/18 split.
- data/manifest.json: dataset hash and historical test IDs.
- data/label_review.csv: original labels and unfilled independent-review fields.
- results/metrics.csv: historical-test accuracy, macro F1 and clarification recall.
- results/predictions.csv: individual test predictions.
- results/failures.csv: misclassified examples and their inputs.
- results/confusion_matrices.png: class-level results.
- results/run.json: model settings, environment and training-only cross-validation.
- results/verification.json: checks of the recorded results.
- original/: historical checkpoint artifacts and the original baseline.

## Evaluation and limitations
All four models were compared on the original 18 historical test examples, six
per class. The test cases were previously inspected, so this is a diagnostic
comparison rather than a fresh blind evaluation. No independent label review,
expanded dataset, or user study was performed. The evaluated command-line models
are separate from the older web interface. Predictions require human review.

The earlier Checkpoint 2 reconstruction changed the data and baseline settings.
Its 88.9% result is not directly comparable with the original historical test.
The original checkpoint artifacts are retained for provenance.

## AI assistance and outside tools
ChatGPT assisted with earlier checkpoint materials, code inspection, evaluation
changes, execution, verification, documentation, and manuscript drafting.
Results are saved program outputs. The project uses scikit-learn, NumPy, pandas,
SciPy and optional PyTorch, Transformers, and Sentence Transformers.
MiniLM is a third-party pretrained model:
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Recorded demo
https://youtu.be/GnEoDjPpTUI
