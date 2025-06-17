import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=== Enhanced Used Car Price Prediction Model ===\n")

# Load Dataset with error handling
try:
    df = pd.read_csv("used_cars_india.csv")
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print("Error: 'used_cars_india.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Standardize column names to match your requirements
expected_columns = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']

# Check if columns exist and standardize names
column_mapping = {}
for col in df.columns:
    col_lower = col.lower().replace('_', '').replace(' ', '')
    if 'brand' in col_lower:
        column_mapping[col] = 'brand'
    elif 'model' in col_lower:
        column_mapping[col] = 'name'  # Use model as name
    elif 'year' in col_lower:
        column_mapping[col] = 'year'
    elif 'selling' in col_lower and 'price' in col_lower:
        column_mapping[col] = 'selling_price'
    elif 'price' in col_lower and 'inr' in col_lower:
        column_mapping[col] = 'selling_price'
    elif ('km' in col_lower and 'driven' in col_lower) or col_lower == 'kmdriven':
        column_mapping[col] = 'km_driven'
    elif 'fuel' in col_lower:
        column_mapping[col] = 'fuel'
    elif 'seller' in col_lower:
        column_mapping[col] = 'seller_type'
    elif 'transmission' in col_lower:
        column_mapping[col] = 'transmission'
    elif 'owner' in col_lower:
        column_mapping[col] = 'owner'
    elif 'engine' in col_lower:
        column_mapping[col] = 'engine_cc'
    elif 'mileage' in col_lower and 'kmpl' in col_lower:
        column_mapping[col] = 'mileage_kmpl'

# Rename columns
df = df.rename(columns=column_mapping)
print(f"Standardized columns: {list(df.columns)}")

# Check if we have all required columns
available_columns = list(df.columns)
missing_cols = [col for col in expected_columns if col not in available_columns]
if missing_cols:
    print(f"Warning: Missing columns {missing_cols}")
    print("Available columns:", available_columns)
    
    # Handle missing columns with alternatives or defaults
    if 'name' not in available_columns and 'brand' in available_columns:
        print("Using 'brand' column as 'name' substitute")
    if 'seller_type' not in available_columns:
        print("'seller_type' not available - will use default value for predictions")

# Data Overview
print(f"\nDataset Overview:")
print(f"Total records: {len(df)}")
print(f"Missing values:\n{df.isnull().sum()}")

# Handle missing values and reset index to avoid duplicate index issues
df_clean = df.dropna().copy()
df_clean = df_clean.reset_index(drop=True)  # FIX: Reset index to avoid duplicates
print(f"Records after removing missing values: {len(df_clean)}")

# Debug: Check for duplicate columns
print(f"Checking for duplicate columns...")
duplicate_cols = df_clean.columns[df_clean.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"Found duplicate columns: {duplicate_cols}")
    # Remove duplicate columns
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
    print(f"After removing duplicates: {list(df_clean.columns)}")
else:
    print("No duplicate columns found")

# Debug: Check index for duplicates
print(f"Checking for duplicate indices...")
if df_clean.index.duplicated().any():
    print(f"Found {df_clean.index.duplicated().sum()} duplicate indices")
    df_clean = df_clean.reset_index(drop=True)
    print("Reset index to remove duplicates")
else:
    print("No duplicate indices found")

# Feature Engineering
print(f"\nPerforming Feature Engineering...")

# Extract brand from car name or use existing brand column
if 'name' in df_clean.columns:
    df_clean['brand'] = df_clean['name'].str.split().str[0]
    print(f"Extracted brands from name: {df_clean['brand'].unique()[:10]}...")
elif 'brand' in df_clean.columns:
    print(f"Using existing brand column: {df_clean['brand'].unique()[:10]}...")
else:
    print("No brand information available")

# Calculate car age
current_year = 2024
if 'year' in df_clean.columns:
    df_clean['car_age'] = current_year - df_clean['year']
    df_clean['car_age'] = df_clean['car_age'].clip(lower=0)  # No negative ages

# Price per kilometer (value retention) - FIXED
if 'km_driven' in df_clean.columns and 'selling_price' in df_clean.columns:
    df_clean['price_per_km'] = df_clean['selling_price'] / (df_clean['km_driven'] + 1)

# Depreciation indicator (higher km = more depreciation)
if 'km_driven' in df_clean.columns:
    df_clean['high_mileage'] = (df_clean['km_driven'] > df_clean['km_driven'].median()).astype(int)

# Owner type numeric conversion
if 'owner' in df_clean.columns:
    # Convert owner descriptions to numeric values
    owner_map = {
        'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 
        'Fourth & Above Owner': 4, 'Test Drive Car': 0
    }
    df_clean['owner_numeric'] = df_clean['owner'].map(owner_map).fillna(1)

# Remove outliers using IQR method
def remove_outliers(df, column):
    if column not in df.columns:
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    # Reset index after filtering to avoid duplicate indices
    return filtered_df.reset_index(drop=True)

# Remove outliers from price and kilometers
if 'selling_price' in df_clean.columns:
    df_clean = remove_outliers(df_clean, 'selling_price')
if 'km_driven' in df_clean.columns:
    df_clean = remove_outliers(df_clean, 'km_driven')

print(f"Records after outlier removal: {len(df_clean)}")

# Data Preprocessing - Label Encoding
print(f"\nEncoding categorical variables...")

categorical_columns = ['brand', 'fuel', 'transmission', 'owner']
# Add seller_type only if it exists
if 'seller_type' in df_clean.columns:
    categorical_columns.append('seller_type')
label_encoders = {}
original_mappings = {}

for col in categorical_columns:
    if col in df_clean.columns:
        le = LabelEncoder()
        # Store original values for prediction
        unique_values = df_clean[col].astype(str).unique()
        original_mappings[col] = dict(zip(unique_values, range(len(unique_values))))
        # Encode
        df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
        print(f"{col} - Unique values: {len(unique_values)}")

# Define features and target
feature_columns = []

# Add numeric features that exist
numeric_features = ['year', 'km_driven', 'car_age', 'price_per_km', 'high_mileage', 'owner_numeric']
for col in numeric_features:
    if col in df_clean.columns:
        feature_columns.append(col)

# Add encoded categorical features
for col in categorical_columns:
    if f'{col}_encoded' in df_clean.columns:
        feature_columns.append(f'{col}_encoded')

# Ensure we have features and target
if 'selling_price' not in df_clean.columns:
    print("Error: 'selling_price' column not found!")
    exit()

X = df_clean[feature_columns]
y = df_clean['selling_price']

print(f"\nFeatures used: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target variable: selling_price")

# Basic statistics
print(f"\nPrice Statistics:")
print(f"Mean price: ₹{y.mean():,.0f}")
print(f"Median price: ₹{y.median():,.0f}")
print(f"Min price: ₹{y.min():,.0f}")
print(f"Max price: ₹{y.max():,.0f}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Model Training - Multiple Models
print(f"\nTraining multiple models...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled features for Linear Regression, original for tree-based models
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        X_train_model = X_train_scaled
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        X_train_model = X_train
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_rmse': cv_rmse,
        'predictions': y_pred
    }
    
    print(f"MAE: ₹{mae:,.0f}")
    print(f"RMSE: ₹{rmse:,.0f}")
    print(f"R² Score: {r2:.3f}")
    print(f"CV RMSE: ₹{cv_rmse:,.0f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_results = results[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"R² Score: {best_results['r2']:.3f}")
print(f"RMSE: ₹{best_results['rmse']:,.0f}")

# Model Comparison Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Model Performance Comparison
model_names = list(results.keys())
mae_scores = [results[name]['mae'] for name in model_names]
rmse_scores = [results[name]['rmse'] for name in model_names]
r2_scores = [results[name]['r2'] for name in model_names]

# MAE Comparison
axes[0,0].bar(model_names, mae_scores, color=['skyblue', 'lightgreen', 'salmon'])
axes[0,0].set_title('Mean Absolute Error Comparison')
axes[0,0].set_ylabel('MAE (₹)')
axes[0,0].tick_params(axis='x', rotation=45)
for i, v in enumerate(mae_scores):
    axes[0,0].text(i, v + max(mae_scores)*0.01, f'₹{v:,.0f}', ha='center', va='bottom', fontsize=8)

# R² Comparison
axes[0,1].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'salmon'])
axes[0,1].set_title('R² Score Comparison')
axes[0,1].set_ylabel('R² Score')
axes[0,1].tick_params(axis='x', rotation=45)
for i, v in enumerate(r2_scores):
    axes[0,1].text(i, v + max(r2_scores)*0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

# Actual vs Predicted (Best Model)
axes[1,0].scatter(y_test, best_results['predictions'], alpha=0.6, color='blue')
axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1,0].set_xlabel('Actual Price (₹)')
axes[1,0].set_ylabel('Predicted Price (₹)')
axes[1,0].set_title(f'Actual vs Predicted - {best_model_name}')
axes[1,0].grid(True, alpha=0.3)

# Residuals Plot
residuals = y_test - best_results['predictions']
axes[1,1].scatter(best_results['predictions'], residuals, alpha=0.6, color='green')
axes[1,1].axhline(y=0, color='r', linestyle='--')
axes[1,1].set_xlabel('Predicted Price (₹)')
axes[1,1].set_ylabel('Residuals (₹)')
axes[1,1].set_title('Residuals Plot')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature', palette='viridis')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    print(f"\nFeature Importance Rankings:")
    for i, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")

# Enhanced Visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price distribution
axes[0,0].hist(df_clean['selling_price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribution of Selling Price')
axes[0,0].set_xlabel('Selling Price (₹)')
axes[0,0].set_ylabel('Count')
axes[0,0].grid(True, alpha=0.3)

# Price vs Age (if car_age exists)
if 'car_age' in df_clean.columns:
    axes[0,1].scatter(df_clean['car_age'], df_clean['selling_price'], alpha=0.5, color='green')
    axes[0,1].set_title('Price vs Car Age')
    axes[0,1].set_xlabel('Car Age (Years)')
    axes[0,1].set_ylabel('Selling Price (₹)')
    axes[0,1].grid(True, alpha=0.3)

# Price vs KM Driven
if 'km_driven' in df_clean.columns:
    axes[1,0].scatter(df_clean['km_driven'], df_clean['selling_price'], alpha=0.5, color='orange')
    axes[1,0].set_title('Price vs Kilometers Driven')
    axes[1,0].set_xlabel('KM Driven')
    axes[1,0].set_ylabel('Selling Price (₹)')
    axes[1,0].grid(True, alpha=0.3)

# Average price by fuel type
if 'fuel' in df_clean.columns:
    avg_fuel_price = df_clean.groupby('fuel')['selling_price'].mean().sort_values(ascending=True)
    axes[1,1].barh(range(len(avg_fuel_price)), avg_fuel_price.values, color='coral')
    axes[1,1].set_yticks(range(len(avg_fuel_price)))
    axes[1,1].set_yticklabels(avg_fuel_price.index)
    axes[1,1].set_title('Average Price by Fuel Type')
    axes[1,1].set_xlabel('Average Price (₹)')
    axes[1,1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# Enhanced Prediction Function
def predict_car_price(sample_input):
    """Enhanced prediction function with your specific parameters"""
    try:
        sample = sample_input.copy()
        
        # Extract brand from name
        if 'name' in sample:
            sample['brand'] = sample['name'].split()[0]
        
        # Add engineered features
        if 'year' in sample:
            sample['car_age'] = current_year - sample['year']
        
        if 'km_driven' in sample:
            # Estimate price per km (will be updated after prediction)
            sample['price_per_km'] = 1.0  # Default value
            sample['high_mileage'] = 1 if sample['km_driven'] > 50000 else 0
        
        # Convert owner to numeric
        if 'owner' in sample:
            owner_map = {
                'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 
                'Fourth & Above Owner': 4, 'Test Drive Car': 0
            }
            sample['owner_numeric'] = owner_map.get(sample['owner'], 1)
        
        # Encode categorical values
        categorical_features = ['brand', 'fuel', 'transmission', 'owner']
        if 'seller_type' in df_clean.columns:
            categorical_features.append('seller_type')
        for col in categorical_features:
            if col in sample and col in label_encoders:
                try:
                    sample[f'{col}_encoded'] = label_encoders[col].transform([str(sample[col])])[0]
                except ValueError:
                    print(f"Warning: '{sample[col]}' not found in {col} training data. Using most common value.")
                    sample[f'{col}_encoded'] = 0
        
        # Create feature vector in the same order as training
        sample_features = []
        for feature in feature_columns:
            if feature in sample:
                sample_features.append(sample[feature])
            else:
                print(f"Warning: Feature '{feature}' not provided. Using default value 0.")
                sample_features.append(0)
        
        sample_df = np.array(sample_features).reshape(1, -1)
        
        # Use best model for prediction
        if best_model_name == 'Linear Regression':
            sample_df = scaler.transform(sample_df)
        
        predicted_price = best_model.predict(sample_df)[0]
        return max(0, predicted_price)  # Ensure non-negative price
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Sample Prediction with your exact parameters
print(f"\n" + "="*60)
print(f"SAMPLE PREDICTION")
print(f"="*60)

sample_car = {
    'name': 'Maruti Swift',  # This will be used to extract brand
    'year': 2018,
    'km_driven': 30000,
    'fuel': 'Petrol',  # Change to match your dataset
    'transmission': 'Manual',
    'owner': 'First Owner'
    # Note: seller_type not included as it's not in your dataset
}

predicted_price = predict_car_price(sample_car)

if predicted_price:
    print(f"\n🚗 Sample Car Details:")
    for key, value in sample_car.items():
        print(f"   {key}: {value}")
    
    print(f"\n💰 Predicted Selling Price: ₹{predicted_price:,.0f}")
    if predicted_price >= 100000:
        print(f"   (Approximately ₹{predicted_price/100000:.1f} lakhs)")
else:
    print("❌ Prediction failed. Please check input data.")

print(f"\n" + "="*60)
print(f"MODEL SUMMARY")
print(f"="*60)
print(f"Dataset: {len(df_clean)} records")
print(f"Features: {len(feature_columns)} features")
print(f"Best Model: {best_model_name}")
print(f"Accuracy (R²): {best_results['r2']:.3f}")
print(f"Average Error: ₹{best_results['mae']:,.0f}")
print(f"Training Records: {len(X_train)}")
print(f"Test Records: {len(X_test)}")
print(f"="*60)

# Show available categories for prediction
print(f"\nAvailable Categories for Prediction:")
categories_to_show = ['brand', 'fuel', 'transmission', 'owner']
if 'seller_type' in df_clean.columns:
    categories_to_show.append('seller_type')
    
for col in categories_to_show:
    if col in df_clean.columns:
        unique_vals = df_clean[col].unique()[:60]  # Show first 10
        print(f"{col}: {list(unique_vals)}")
        if len(df_clean[col].unique()) > 60:
            print(f"   ... and {len(df_clean[col].unique())-60} more")
 # Make important variables available to Flask app
best_model_name = best_model_name
best_results = best_results