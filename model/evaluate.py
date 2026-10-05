"""
Evaluate the trained LightGBM model against the spatial hold-out test set.
Saves metrics to a timestamped file in model/artifacts/.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.predict import find_latest_model, load_model, prepare_features
from model.train import TARGET_COL, load_dataset, spatial_split

ARTIFACTS_DIR = Path("model/artifacts")


def evaluate() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on spatial test split.")
    parser.add_argument("--model", type=Path, default=None, help="Path to model pkl")
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/labeled_dataset.csv"))
    args = parser.parse_args()

    model_path = args.model or find_latest_model(ARTIFACTS_DIR)
    clf, le, feature_cols = load_model(model_path)

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    # Recreate the exact same spatial split (seed=42 is default)
    df = load_dataset(args.dataset).dropna(subset=[TARGET_COL])
    _, test_df = spatial_split(df)
    
    if len(test_df) == 0:
        print("No test data available for evaluation.")
        sys.exit(1)

    X_test = prepare_features(test_df, feature_cols)
    y_true = le.transform(test_df[TARGET_COL])
    y_pred = clf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    # Find weakest recall
    recalls = {cls: report_dict[cls]["recall"] for cls in le.classes_ if cls in report_dict}
    weakest_class = min(recalls.keys(), key=lambda k: recalls[k]) if recalls else "N/A"
    weakest_recall = recalls.get(weakest_class, 0)

    # Format output
    output = []
    output.append(f"Model Evaluation: {model_path.name}")
    output.append(f"Timestamp: {datetime.now().isoformat()}")
    output.append(f"Test Set Size: {len(test_df)} rows")
    output.append(f"Overall Accuracy: {acc:.4f}\n")
    
    output.append("=== PER-CLASS METRICS ===")
    output.append(report_str)
    
    output.append("=== CONFUSION MATRIX (Rows=Actual, Cols=Predicted) ===")
    output.append(cm_df.to_string() + "\n")
    
    output.append("=== INSIGHTS ===")
    output.append(f"⚠️ Weakest Recall: '{weakest_class}' ({weakest_recall:.2%} found)")
    if weakest_recall < 0.5:
        output.append(f"  -> The model is struggling to identify {weakest_class}. Consider reviewing the heuristic rules in Phase 3 or providing more labeled data for this class.")
    else:
        output.append("  -> All classes have reasonable recall.")

    out_text = "\n".join(output)
    
    # Print to console
    print(out_text)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = ARTIFACTS_DIR / f"metrics_{timestamp}.txt"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(out_text)
    print(f"\nSaved metrics to {metrics_path}")


if __name__ == "__main__":
    evaluate()
