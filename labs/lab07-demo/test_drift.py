import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from evidently import Report
from evidently.presets import DataDriftPreset


# 1. Load the Iris dataset
iris = load_iris(as_frame=True)

X = iris.data
feature_cols = X.columns.tolist()

# 2. Split into reference and test datasets
reference_data, test_data = train_test_split(
    X,
    test_size=0.3,
    random_state=42
)

# 3. Simulate production data
# Shift the feature distributions to simulate data drift.
current_data = test_data.copy()

current_data["sepal length (cm)"] += 1.5
current_data["petal length (cm)"] += 1.0

# 4. Save datasets for inspection
reference_data.to_csv("training_data.csv", index=False)
current_data.to_csv("production_logs_last_7d.csv", index=False)

print("Reference data shape:", reference_data.shape)
print("Production data shape:", current_data.shape)

# 5. Generate the drift report
report = Report(
    metrics=[DataDriftPreset()]
)

snapshot=report.run(
    reference_data=reference_data[feature_cols],
    current_data=current_data[feature_cols]
)

# 6. Save the HTML report
snapshot.save_html("drift_report.html")

print("\nDrift report saved to drift_report.html")

# 7. Inspect the report programmatically
result = snapshot.as_dict()

print("\nReport results:")
print(result)

# 8. Find the dataset-level drift result
drift_detected = None

for metric in result["metrics"]:
    metric_result = metric.get("result", {})

    if "dataset_drift" in metric_result:
        drift_detected = metric_result["dataset_drift"]
        break

if drift_detected is None:
    print(
        "\nCould not find dataset_drift in the report structure. "
        "Inspect the printed results for your Evidently version."
    )
elif drift_detected:
    print("\nWARNING: Data drift detected!")
    print("Investigate the changes before considering retraining.")
else:
    print("\nNo dataset-level drift detected.")