import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("California_Housing_Optimization")
experiments_config = [
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 7, "learning_rate": 0.01},
]
for i, config in enumerate(experiments_config, start=1):
    run_name = f"Run {i}"
    max_depth = config["max_depth"]
    learning_rate = config["learning_rate"]
    with mlflow.start_run(run_name=run_name):
        model = xgb.XGBRegressor( max_depth=max_depth, learning_rate=learning_rate,random_state=42  )
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.xgboost.log_model(model,artifact_path="xgboost-model" )
        print(
            f"{run_name} Complete | "
            f"Depth: {max_depth} | "
            f"Learning Rate: {learning_rate} | "
            f"RMSE: {rmse:.4f} | "
            f"MAE: {mae:.4f} | "
            f"R2: {r2:.4f}"
        )