from mlflow.tracking import MlflowClient
import mlflow

'''' mlflow, version 2.3.2 '''

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()

# client = MlflowClient.tracking_uri('http://127.0.0.1:5000')
# print(f'Tracking URI: {mlflow.get_tracking_uri()}')
# print(f'List Experiments: {mlflow.list_experiments()}')

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
