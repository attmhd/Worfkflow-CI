import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from mlflow.models import infer_signature



# Define paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "diabetes_dataset_processing.csv")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "test_confusion_matrix.png")

def load_data():
    """Load and preprocess dataset, return train and test split"""
    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Handle class imbalance using SMOTE on train only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    return X_train_resampled, y_train_resampled, X_test, y_test

def create_grid_search():
    """Create grid search with hyperparameter grid"""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    
    rf = RandomForestClassifier(random_state=42)
    return GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        scoring="accuracy", 
        verbose=2
    )

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix plot"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    # Load data (train and test split)
    X_train_resampled, y_train_resampled, X_test, y_test = load_data()
    
    # Create and run grid search
    grid_search = create_grid_search()
    
    mlflow.autolog(log_models=False)
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Log model and dataset with signature
    signature = infer_signature(X_test, grid_search.best_estimator_.predict(X_test))
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        "tuned_model",
        signature=signature,
        input_example=X_test.iloc[:1]
    )
    mlflow.log_artifact(DATASET_PATH, artifact_path="data")
    
    # Generate and log confusion matrix on test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    conf_matrix_path = plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
    mlflow.log_artifact(conf_matrix_path, artifact_path="confusion_matrix")

    # Log classification report metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("test_precision", report["weighted avg"]["precision"])
    mlflow.log_metric("test_recall", report["weighted avg"]["recall"])
    mlflow.log_metric("test_f1_score", report["weighted avg"]["f1-score"])
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    main()
