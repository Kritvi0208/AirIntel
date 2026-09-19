"""
AirIntel - Production Classification Training & MLflow Experiment Tracking
Author: Ritvika (IIT BHU)
Tracks CatBoost 6-tier EPA severity classifier with metrics, confusion matrix, and parameters.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "airintel_ml_final.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "models", "deployment", "deployment_pipeline.pkl")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

def run_classification_tracking():
    print("=" * 65)
    print("  AIRINTEL MLOPS: CLASSIFICATION EXPERIMENT TRACKING")
    print("=" * 65)
    
    # 1. Initialize MLflow SQLite backend store
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")
    experiment_name = "AirIntel_Severity_Classification"
    mlflow.set_experiment(experiment_name)
    
    # 2. Load deployment bundle and dataset
    print(f"[1/4] Loading deployment pipeline from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    cls_pipeline = bundle['cls_pipeline']
    label_encoder = bundle['label_encoder']
    features = bundle['selected_features']
    class_names = list(label_encoder.classes_)
    
    print(f"[2/4] Loading test dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Chronological holdout evaluation
    test_size = int(len(df) * 0.20)
    df_test = df.iloc[-test_size:].copy()
    
    X_test = df_test[features]
    y_test_raw = df_test['AQI_Category'].values
    y_test = label_encoder.transform(y_test_raw)
    
    print(f"[3/4] Evaluating CatBoost Classifier on {len(df_test):,} holdout samples...")
    y_pred_raw = cls_pipeline.predict(X_test)
    y_pred = np.array(y_pred_raw).ravel()
    
    acc = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    
    print(f"      Accuracy    : {acc * 100:.2f}%")
    print(f"      F1-Weighted : {f1_weighted:.4f}")
    print(f"      F1-Macro    : {f1_macro:.4f}")
    
    cb_model = cls_pipeline.named_steps['model']
    cb_params = cb_model.get_params()
    params = {
        "model_type": "CatBoost_Classifier",
        "iterations": cb_params.get("iterations", 150),
        "depth": cb_params.get("depth", 8),
        "learning_rate": cb_params.get("learning_rate", 0.2),
        "l2_leaf_reg": cb_params.get("l2_leaf_reg", 7),
        "class_count": len(class_names),
        "target_variable": "AQI_Category",
        "author": "Ritvika (IIT BHU)"
    }
    
    # 4. Log run into MLflow
    print("[4/4] Logging experiment run and artifacts into MLflow...")
    with mlflow.start_run(run_name="CatBoost_Production_Benchmark") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro
        })
        
        # Plot confusion matrix with fixed 6-tier labels
        labels_idx = np.arange(len(class_names))
        cm = confusion_matrix(y_test, y_pred, labels=labels_idx)
        # Avoid division by zero for rare tiers
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        
        fig, ax = plt.subplots(figsize=(8, 7), dpi=120)
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_xticks(labels_idx)
        ax.set_yticks(labels_idx)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_title("AirIntel CatBoost: Normalized Confusion Matrix", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Severity Tier")
        ax.set_xlabel("Predicted Severity Tier")
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Print numbers inside cells
        thresh = cm_norm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black",
                        fontsize=8)
                        
        plt.tight_layout()
        plot_path = os.path.join(BASE_DIR, "reports", "figures", "mlflow_confusion_matrix.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
        
        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
        
        # Log model artifact with cloudpickle serialization
        mlflow.sklearn.log_model(
            sk_model=cls_pipeline,
            name="catboost_classifier_pipeline",
            serialization_format="cloudpickle"
        )
        mlflow.log_artifact(MODEL_PATH, artifact_path="deployment_bundle")
        
        print(f"      Run successfully logged! Run ID: {run.info.run_id}")
        print("=" * 65)

if __name__ == "__main__":
    run_classification_tracking()
