# Load a specific version from the registry
import mlflow.pyfunc
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    

# Load by version number
model_v1 = mlflow.pyfunc.load_model("models:/IrisClassifier/1")

# Load by stage (Staging, Production)
# prod_model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")

# Make predictions
predictions = model_v1.predict(X_test)
print("Predictions:", predictions)