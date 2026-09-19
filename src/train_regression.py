"""
AirIntel - Production Regression Training & MLflow Experiment Tracking
Author: Ritvika (IIT BHU)
Tracks LightGBM AQI continuous predictor with parameters, metrics, and residual artifacts.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configure paths
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "airintel_ml_final.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "models", "deployment", "deployment_pipeline.pkl")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

def run_regression_tracking():
    print("=" * 65)
    print("  AIRINTEL MLOPS: REGRESSION EXPERIMENT TRACKING")
    print("=" * 65)
    
    # 1. Initialize MLflow SQLite backend store
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")
    experiment_name = "AirIntel_AQI_Regression"
    mlflow.set_experiment(experiment_name)
    
    # 2. Load deployment bundle and dataset
    print(f"[1/4] Loading deployment pipeline from {MODEL_PATH}...")
    bundle = joblib.load(MODEL_PATH)
    reg_pipeline = bundle['reg_pipeline']
    features = bundle['selected_features']
    
    print(f"[2/4] Loading test dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Use chronological 20% holdout for evaluation
    test_size = int(len(df) * 0.20)
    df_test = df.iloc[-test_size:].copy()
    
    X_test = df_test[features]
    y_test = df_test['US_AQI'].values
    
    print(f"[3/4] Evaluating LightGBM Regressor on {len(df_test):,} holdout samples...")
    y_pred = reg_pipeline.predict(X_test)
    y_pred = np.clip(y_pred, 0, 500)
    
    # Compute metrics
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    pearson_corr = float(np.corrcoef(y_test, y_pred)[0, 1])
    
    print(f"      R² Score    : {r2:.4f}")
    print(f"      MAE (AQI)   : {mae:.2f}")
    print(f"      RMSE (AQI)  : {rmse:.2f}")
    print(f"      Pearson Corr: {pearson_corr:.4f}")
    
    # Extract model hyperparameters
    lgb_model = reg_pipeline.named_steps['model']
    params = {
        "model_type": "LightGBM_Regressor",
        "n_estimators": lgb_model.n_estimators,
        "learning_rate": lgb_model.learning_rate,
        "max_depth": lgb_model.max_depth,
        "num_leaves": lgb_model.num_leaves,
        "subsample": lgb_model.subsample,
        "feature_count": len(features),
        "target_variable": "US_AQI",
        "author": "Ritvika (IIT BHU)"
    }
    
    # 4. Log run into MLflow
    print("[4/4] Logging experiment run and artifacts into MLflow...")
    with mlflow.start_run(run_name="LightGBM_Production_Benchmark") as run:
        # Log params & metrics
        mlflow.log_params(params)
        mlflow.log_metrics({
            "r2_score": r2,
            "mae": mae,
            "rmse": rmse,
            "pearson_correlation": pearson_corr
        })
        
        # Generate & log actual vs predicted residual figure
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
        # Sample for fast plotting
        plot_idx = np.random.RandomState(42).choice(len(y_test), size=min(2000, len(y_test)), replace=False)
        ax.scatter(y_test[plot_idx], y_pred[plot_idx], alpha=0.35, color="#0284C7", edgecolors="none", s=18)
        ax.plot([0, 500], [0, 500], "r--", lw=1.8, label="Ideal 1:1 Parity")
        ax.set_title("AirIntel LightGBM: Actual vs Predicted AQI (Holdout)", fontsize=12, fontweight="bold")
        ax.set_xlabel("True Observed US_AQI", fontsize=10)
        ax.set_ylabel("Predicted US_AQI", fontsize=10)
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        plt.tight_layout()
        
        plot_path = os.path.join(BASE_DIR, "reports", "figures", "mlflow_regression_residual.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path)
        plt.close(fig)
        
        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
        
        # Log model artifact with cloudpickle serialization
        mlflow.sklearn.log_model(
            sk_model=reg_pipeline,
            name="lightgbm_regressor_pipeline",
            serialization_format="cloudpickle"
        )
        mlflow.log_artifact(MODEL_PATH, artifact_path="deployment_bundle")
        
        print(f"      Run successfully logged! Run ID: {run.info.run_id}")
        print("=" * 65)

if __name__ == "__main__":
    run_regression_tracking()
