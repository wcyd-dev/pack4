
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load the dataset
df = pd.read_csv('synthetic_loe_sales_data.csv')

# Select the required features
df_selected = df[['Marketing_Spend', 'Price', 'Inventory_Level', 'Economic_Conditions', 'Adherence_Rate', 'Sales']]

# Check for missing values
if df_selected.isnull().sum().any():
    print("Data contains missing values! Please handle them before proceeding.")
else:
    print("No missing values in the dataset.")

# Split the data into train and test sets
X = df_selected[['Marketing_Spend', 'Price', 'Inventory_Level', 'Economic_Conditions', 'Adherence_Rate']]
y = df_selected['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")

# Save the trained model
model_filename = 'sales_predictor_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
