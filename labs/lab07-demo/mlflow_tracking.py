import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

mlflow.set_experiment("iris-classifier")

with mlflow.start_run(run_name="rf-v1"):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Log parameters
    params = {"n_estimators": 100, "max_depth": 5}
    mlflow.log_params(params)
    
    # Train
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    
    # Log metrics
    preds = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="macro"))
    
    # Log model
    mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    registered_model_name="IrisClassifier",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
