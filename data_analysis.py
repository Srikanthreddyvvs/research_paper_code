import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# Enable logging to debug steps
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Load the dataset
logging.debug("Step 1: Load the dataset")
try:
    data = pd.read_csv(r'C:/Users/vsake/OneDrive/Documents/dwdm/final_merged_data.csv')
    logging.debug(f"Dataset loaded successfully with shape: {data.shape}")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    exit()

# Step 2: Data cleaning and preprocessing
logging.debug("Step 2: Clean and preprocess the dataset")
try:
    # Function to clean strings and convert to float
    def clean_and_convert(column, remove_str):
        return column.str.replace('\xa0', '').str.replace(remove_str, '').astype(float)

    # Remove non-numeric units in 'Temperature', 'Rainfall', 'Humidity' and handle non-breaking spaces
    data['Temperature'] = clean_and_convert(data['Temperature'], '°C')
    data['Rainfall'] = clean_and_convert(data['Rainfall'], 'mm')
    data['Humidity'] = clean_and_convert(data['Humidity'], '%')

    # Check if cleaning was successful
    logging.debug(f"Cleaned columns: {data[['Temperature', 'Rainfall', 'Humidity']].head()}")

    # Handle missing values (fill with median for simplicity)
    data.fillna(data.median(numeric_only=True), inplace=True)

    # One-hot encode categorical columns
    categorical_cols = ['District', 'Crop', 'Soil Type', 'Season']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_categorical = encoder.fit_transform(data[categorical_cols])
    
    # Create DataFrame for encoded categorical columns
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Concatenate encoded columns with original dataframe
    data = pd.concat([data.drop(categorical_cols, axis=1), encoded_df], axis=1)
    logging.debug(f"Data after encoding: {data.head()}")

except Exception as e:
    logging.error(f"Error in data preprocessing: {e}")
    exit()

# Step 3: Clustering the data for insights (KMeans)
logging.debug("Step 3: Perform clustering")
try:
    # Selecting features for clustering (ignoring target variable like Price)
    features = ['Water Consumption (liters/hectare)', 'Water Availability (liters/hectare)', 
                'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)', 'Potassium Content (kg/ha)',
                'Average Soil Moisture', 'Temperature', 'Rainfall', 'Humidity']

    X_cluster = data[features]
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
    
    logging.debug(f"Clustering results: {data[['Cluster']].value_counts()}")

except Exception as e:
    logging.error(f"Error in clustering: {e}")
    exit()

# Step 4: Train Gradient Boosting model for water consumption prediction
logging.debug("Step 4: Train Gradient Boosting model")
try:
    target = 'Water Consumption (liters/hectare)'
    
    # Define features (dropping the target and irrelevant columns)
    X = data.drop([target, 'Price (INR/quintal)'], axis=1)  # Assuming Price is not a predictor
    y = data[target]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Gradient Boosting Regressor
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    logging.debug(f"Training MSE: {train_mse}")
    logging.debug(f"Test MSE: {test_mse}")

except Exception as e:
    logging.error(f"Error in model training: {e}")
    exit()

# Step 5: Analysis of Feature Importance
logging.debug("Step 5: Feature Importance")
try:
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    logging.debug(f"Top features:\n{importance_df.head(10)}")

except Exception as e:
    logging.error(f"Error in feature importance analysis: {e}")
    exit()

# Step 6: Save the results
logging.debug("Step 6: Save the results to CSV")
try:
    data.to_csv('final_analysis_results.csv', index=False)
    logging.debug("Results saved to final_analysis_results.csv")
except Exception as e:
    logging.error(f"Error saving results: {e}")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the final dataset with cluster labels
df = pd.read_csv('final_analysis_results.csv')

# Compute the mean values for each cluster
cluster_means = df.groupby('Cluster').mean()

print("Cluster Means:")
print(cluster_means)

# Visualize the distribution of key features by cluster
plt.figure(figsize=(12, 8))

# Plot temperature distribution by cluster
sns.boxplot(x='Cluster', y='Temperature', data=df)
plt.title('Temperature Distribution by Cluster')
plt.show()

# Plot water consumption distribution by cluster
sns.boxplot(x='Cluster', y='Water Consumption (liters/hectare)', data=df)
plt.title('Water Consumption Distribution by Cluster')
plt.show()

# Plot soil moisture distribution by cluster
sns.boxplot(x='Cluster', y='Average Soil Moisture', data=df)
plt.title('Soil Moisture Distribution by Cluster')
plt.show()