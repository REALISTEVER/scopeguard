"""ScopeGuard+ Practical Checkpoint 1 baseline.

Reads scopeguard_checkpoint1_data.csv and evaluates a deterministic
keyword/Jaccard baseline on the held-out test split.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

DATA_PATH = Path(__file__).with_name("scopeguard_checkpoint1_data.csv")
PREDICTIONS_PATH = Path(__file__).with_name("test_predictions.csv")
RESULTS_PATH = Path(__file__).with_name("baseline_results.txt")

LABELS = ["IN SCOPE", "OUT OF SCOPE", "NEEDS CLARIFICATION"]
STOPWORDS = set(
    "a an the and or to of for in on with that this is are be by from as at into each current new "
    "build application app tool system users user add let show make more using use".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, tokenize, and remove a small fixed stop-word list."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def classify(project_scope: str, new_request: str) -> tuple[str, float]:
    """Classify using Jaccard similarity only; no learned model is used."""
    scope_tokens = set(tokenize(project_scope))
    request_tokens = set(tokenize(new_request))
    union = scope_tokens | request_tokens
    score = len(scope_tokens & request_tokens) / len(union) if union else 0.0

    # Fixed thresholds selected before evaluating the held-out test rows.
    if score >= 0.08:
        return "IN SCOPE", score
    if score >= 0.025:
        return "NEEDS CLARIFICATION", score
    return "OUT OF SCOPE", score


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = load_rows(DATA_PATH)
    test_rows = [row for row in rows if row["split"] == "test"]

    true_labels: list[str] = []
    predictions: list[str] = []
    output_rows: list[dict[str, str]] = []

    for row in test_rows:
        predicted, score = classify(row["project_scope"], row["new_request"])
        true_labels.append(row["label"])
        predictions.append(predicted)
        output_rows.append(
            {
                "id": row["id"],
                "project_domain": row["project_domain"],
                "new_request": row["new_request"],
                "true_label": row["label"],
                "prediction": predicted,
                "jaccard_score": f"{score:.4f}",
            }
        )

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average="macro")
    clarification_recall = recall_score(
        true_labels,
        predictions,
        labels=["NEEDS CLARIFICATION"],
        average=None,
        zero_division=0,
    )[0]
    matrix = confusion_matrix(true_labels, predictions, labels=LABELS)

    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)

    report = (
        "ScopeGuard+ Checkpoint 1 - Keyword/Jaccard Baseline\n"
        f"Dataset: {len(rows)} synthetic labeled pairs (30 per class)\n"
        f"Held-out test set: {len(test_rows)} examples (6 per class)\n\n"
        f"Accuracy: {accuracy:.3f}\n"
        f"Macro-F1: {macro_f1:.3f}\n"
        f"NEEDS CLARIFICATION recall: {clarification_recall:.3f}\n\n"
        "Confusion matrix (rows=true, columns=predicted)\n"
        f"Labels: {LABELS}\n"
        + "\n".join(str(list(row)) for row in matrix)
        + "\n\nClassification report\n"
        + classification_report(true_labels, predictions, labels=LABELS, zero_division=0)
    )
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
