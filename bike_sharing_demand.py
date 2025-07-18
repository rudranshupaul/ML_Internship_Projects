# ============================================================================
# Bike Sharing Demand Prediction
# Internship Project 2 - Machine Learning
# Student: RUDRANSHU PAUL
# Dataset: UCI/Kaggle Daily Bike Sharing Dataset(Boom Bikes dataset)
# Date: July 1, 2025
# ============================================================================

# ============================================================================
# STEP 1: IMPORTING LIBRARIES & DATA UNDERSTANDING
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("✅ Libraries imported successfully!")

# 1.2 Load Dataset
dataset_path = '/home/personal/Downloads/archive (4)/bike_sharing_data.csv'
try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Successfully loaded '{dataset_path}'")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# 1.3 Rename columns for compatibility
rename_dict = {
    'dteday': 'datetime',
    'weathersit': 'weather',
    'hum': 'humidity',
    'cnt': 'count',
    'yr': 'year',
    'mnth': 'month',
    'weekday': 'weekday'
}
df.rename(columns=rename_dict, inplace=True)
print("✅ Columns renamed for compatibility.")
print("Columns now:", df.columns.tolist())

# 1.4 Dataset Overview
print("\n" + "="*70)
print("🔍 DATASET OVERVIEW")
print("="*70)
print(f"📊 Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
print("\n📋 Column Names:")
print(df.columns.tolist())

print("\n👀 First 5 Rows:")
print(df.head())
print("\n👀 Last 5 Rows:")
print(df.tail())

print("\n" + "="*70)
print("🧩 DATA TYPES & MISSING VALUES")
print("="*70)
print(df.info())
print("\n🔎 Missing Values Per Column:")
print(df.isnull().sum())

print("\n" + "="*70)
print("📈 STATISTICAL SUMMARY (NUMERICAL FEATURES)")
print("="*70)
print(df.describe())

print("\n" + "="*70)
print("🔢 UNIQUE VALUES & SAMPLE VALUES")
print("="*70)
for col in df.columns:
    print(f"{col:15}: {df[col].nunique()} unique | Sample: {df[col].unique()[:5]}")

# 1.9 Overview Table (Plotly)
overview = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values,
    'Missing Values': df.isnull().sum().values,
    'Unique Values': [df[col].nunique() for col in df.columns]
})
try:
    pio.renderers.default = 'browser'
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(overview.columns), fill_color='#1f77b4', font=dict(color='white', size=14), align='left'),
        cells=dict(values=[overview[col] for col in overview.columns], fill_color='lavender', align='left'))
    ])
    fig.update_layout(title="Dataset Structure Overview", height=500)
    fig.show()
except Exception:
    print("Plotly Table could not be displayed in this environment.")
    print(overview)

print("\n✅ STEP 1 COMPLETE: Data loaded, inspected, and summarized.")

# ============================================================================
# STEP 2: DATA VISUALIZATION (INTERACTIVE & PROFESSIONAL)
# ============================================================================

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues

print("\n" + "="*70)
print("📊 STEP 2: DATA VISUALIZATION")
print("="*70)

plt.figure(figsize=(10,5))
sns.histplot(df['count'], bins=40, kde=True, color='royalblue')
plt.title('Distribution of Bike Demand (count)', fontsize=14, weight='bold')
plt.xlabel('Total Rentals (count)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(x='season', y='count', data=df, palette='Set2')
plt.title('Bike Demand by Season', fontsize=14, weight='bold')
plt.xlabel('Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)')
plt.ylabel('Total Rentals (count)')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_by_season.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
sns.barplot(x='month', y='count', data=df, palette='viridis', ci=None)
plt.title('Average Bike Demand by Month', fontsize=14, weight='bold')
plt.xlabel('Month')
plt.ylabel('Average Rentals (count)')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_by_month.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(x='weather', y='count', data=df, palette='coolwarm')
plt.title('Bike Demand by Weather Condition', fontsize=14, weight='bold')
plt.xlabel('Weather (1=Clear, 2=Mist, 3=Light Rain/Snow)')
plt.ylabel('Total Rentals (count)')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_by_weather.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(x='temp', y='count', data=df, hue='season', palette='Set1', alpha=0.7)
plt.title('Bike Demand vs Temperature', fontsize=14, weight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Total Rentals (count)')
plt.legend(title='Season')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_vs_temperature.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,5))
sns.scatterplot(x='humidity', y='count', data=df, hue='weather', palette='Set2', alpha=0.7)
plt.title('Bike Demand vs Humidity', fontsize=14, weight='bold')
plt.xlabel('Humidity (%)')
plt.ylabel('Total Rentals (count)')
plt.legend(title='Weather')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_vs_humidity.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
df['day_type'] = df.apply(lambda x: 'Holiday' if x['holiday'] == 1 
                         else ('Working Day' if x['workingday'] == 1 else 'Weekend'), axis=1)
sns.boxplot(x='day_type', y='count', data=df, palette='pastel')
plt.title('Bike Demand by Day Type', fontsize=14, weight='bold')
plt.xlabel('Day Type')
plt.ylabel('Total Rentals (count)')
plt.grid(True, alpha=0.2)
plt.savefig('bike_demand_by_day_type.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12,8))
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'instant' in numeric_cols:
    numeric_cols.remove('instant')
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, 
            square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Heatmap of Numeric Features', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10,5))
plt.scatter(df['casual'], df['count'], alpha=0.5, label='Total vs Casual', color='blue')
plt.scatter(df['registered'], df['count'], alpha=0.5, label='Total vs Registered', color='red')
plt.xlabel('User Count')
plt.ylabel('Total Rentals (count)')
plt.title('Casual vs Registered Users Contribution', fontsize=14, weight='bold')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig('casual_vs_registered.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ STEP 2 COMPLETE: Data visualizations generated and saved as PNG files.")
print("📁 Plot files saved in the same directory as your script.")

# ============================================================================
# STEP 3: COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA) - CORRECTED
# ============================================================================

print("\n" + "="*70)
print("🔍 STEP 3: COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
print("="*70)

# 3.1 Target Variable Analysis
print("\n" + "-"*50)
print("🎯 TARGET VARIABLE ANALYSIS")
print("-"*50)

print(f"Target Variable: 'count' (Total Bike Rentals)")
print(f"Mean: {df['count'].mean():.2f}")
print(f"Median: {df['count'].median():.2f}")
print(f"Standard Deviation: {df['count'].std():.2f}")
print(f"Min: {df['count'].min()}")
print(f"Max: {df['count'].max()}")
print(f"Range: {df['count'].max() - df['count'].min()}")

q1 = df['count'].quantile(0.25)
q3 = df['count'].quantile(0.75)
iqr = q3 - q1
print(f"Q1 (25th percentile): {q1:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")
print(f"IQR (Interquartile Range): {iqr:.2f}")

outlier_threshold_low = q1 - 1.5 * iqr
outlier_threshold_high = q3 + 1.5 * iqr
outliers = df[(df['count'] < outlier_threshold_low) | (df['count'] > outlier_threshold_high)]
print(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# 3.2 Temporal Analysis
print("\n" + "-"*50)
print("📅 TEMPORAL ANALYSIS")
print("-"*50)

# Parse datetime with dayfirst=True for 'dd-mm-yyyy' format
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df['day_of_year'] = df['datetime'].dt.dayofyear
df['month_name'] = df['datetime'].dt.month_name()

monthly_stats = df.groupby('month')['count'].agg(['mean', 'std', 'min', 'max']).round(2)
print("Monthly Statistics:")
print(monthly_stats)

seasonal_stats = df.groupby('season')['count'].agg(['mean', 'std', 'min', 'max']).round(2)
print("\nSeasonal Statistics (1=Spring, 2=Summer, 3=Fall, 4=Winter):")
print(seasonal_stats)

weekday_stats = df.groupby('weekday')['count'].agg(['mean', 'std', 'min', 'max']).round(2)
print("\nWeekday Statistics (0=Sunday, 1=Monday, ..., 6=Saturday):")
print(weekday_stats)

# 3.3 Weather Impact Analysis
print("\n" + "-"*50)
print("🌤️ WEATHER IMPACT ANALYSIS")
print("-"*50)

weather_stats = df.groupby('weather')['count'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
print("Weather Condition Statistics:")
print("1=Clear/Partly Cloudy, 2=Mist/Cloudy, 3=Light Rain/Snow")
print(weather_stats)

temp_corr = df['temp'].corr(df['count'])
atemp_corr = df['atemp'].corr(df['count'])
humidity_corr = df['humidity'].corr(df['count'])
windspeed_corr = df['windspeed'].corr(df['count'])

print(f"\nWeather Correlations with Bike Demand:")
print(f"Temperature correlation: {temp_corr:.3f}")
print(f"Feels-like temperature correlation: {atemp_corr:.3f}")
print(f"Humidity correlation: {humidity_corr:.3f}")
print(f"Windspeed correlation: {windspeed_corr:.3f}")

# 3.4 User Type Analysis
print("\n" + "-"*50)
print("👥 USER TYPE ANALYSIS")
print("-"*50)

print("Casual vs Registered Users Analysis:")
print(f"Average casual users: {df['casual'].mean():.2f}")
print(f"Average registered users: {df['registered'].mean():.2f}")
print(f"Total average users: {df['count'].mean():.2f}")

casual_percentage = (df['casual'].sum() / df['count'].sum()) * 100
registered_percentage = (df['registered'].sum() / df['count'].sum()) * 100
print(f"\nUser Distribution:")
print(f"Casual users: {casual_percentage:.1f}%")
print(f"Registered users: {registered_percentage:.1f}%")

casual_registered_corr = df['casual'].corr(df['registered'])
print(f"Correlation between casual and registered users: {casual_registered_corr:.3f}")

# 3.5 Holiday vs Working Day Analysis
print("\n" + "-"*50)
print("🏖️ HOLIDAY VS WORKING DAY ANALYSIS")
print("-"*50)

df['day_category'] = df.apply(lambda x: 'Holiday' if x['holiday'] == 1 
                             else ('Working Day' if x['workingday'] == 1 else 'Weekend'), axis=1)

day_type_stats = df.groupby('day_category')['count'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
print("Day Type Statistics:")
print(day_type_stats)

holiday_avg = df[df['holiday'] == 1]['count'].mean()
non_holiday_avg = df[df['holiday'] == 0]['count'].mean()
print(f"\nHoliday Impact:")
print(f"Average rentals on holidays: {holiday_avg:.2f}")
print(f"Average rentals on non-holidays: {non_holiday_avg:.2f}")
print(f"Holiday vs Non-holiday difference: {holiday_avg - non_holiday_avg:.2f}")

# 3.6 Feature Relationships and Patterns
print("\n" + "-"*50)
print("🔗 FEATURE RELATIONSHIPS AND PATTERNS")
print("-"*50)

numeric_features = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 
                   'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
correlation_matrix = df[numeric_features].corr()

print("Top 5 positive correlations with bike demand:")
count_correlations = correlation_matrix['count'].drop('count').sort_values(ascending=False)
print(count_correlations.head())

print("\nTop 5 negative correlations with bike demand:")
print(count_correlations.tail())

# 3.7 Data Quality Assessment - CORRECTED
print("\n" + "-"*50)
print("✅ DATA QUALITY ASSESSMENT")
print("-"*50)

print("Data Quality Summary:")
print(f"Total records: {len(df)}")
print(f"Duplicate records: {df.duplicated().sum()}")
print(f"Missing values: {df.isnull().sum().sum()}")

print(f"\nLogical Consistency Checks:")
print(f"Records where casual + registered != count: {len(df[df['casual'] + df['registered'] != df['count']])}")

# FIX: Only check numeric columns for negative values
numeric_cols = df.select_dtypes(include=[np.number]).columns
negative_mask = (df[numeric_cols] < 0).any(axis=1)
print(f"Records with negative values: {negative_mask.sum()}")

print(f"\nValue Range Checks:")
print(f"Temperature range: {df['temp'].min():.2f} to {df['temp'].max():.2f}")
print(f"Humidity range: {df['humidity'].min():.2f} to {df['humidity'].max():.2f}")
print(f"Windspeed range: {df['windspeed'].min():.2f} to {df['windspeed'].max():.2f}")

# 3.8 Business Insights and Key Findings
print("\n" + "-"*50)
print("💡 BUSINESS INSIGHTS AND KEY FINDINGS")
print("-"*50)

peak_season = seasonal_stats['mean'].idxmax()
peak_month = monthly_stats['mean'].idxmax()
peak_weather = weather_stats['mean'].idxmax()

print("Key Business Insights:")
print(f"1. Peak demand season: Season {peak_season} (1=Spring, 2=Summer, 3=Fall, 4=Winter)")
print(f"2. Peak demand month: Month {peak_month}")
print(f"3. Best weather for bike rentals: Weather condition {peak_weather}")
print(f"4. Registered users dominate with {registered_percentage:.1f}% of total rentals")
print(f"5. Strong positive correlation with temperature: {temp_corr:.3f}")
print(f"6. Strong negative correlation with humidity: {humidity_corr:.3f}")

low_demand_threshold = df['count'].quantile(0.33)
high_demand_threshold = df['count'].quantile(0.67)

low_demand_days = len(df[df['count'] <= low_demand_threshold])
medium_demand_days = len(df[(df['count'] > low_demand_threshold) & (df['count'] <= high_demand_threshold)])
high_demand_days = len(df[df['count'] > high_demand_threshold])

print(f"\nDemand Distribution:")
print(f"Low demand days (≤{low_demand_threshold:.0f} rentals): {low_demand_days} days ({low_demand_days/len(df)*100:.1f}%)")
print(f"Medium demand days: {medium_demand_days} days ({medium_demand_days/len(df)*100:.1f}%)")
print(f"High demand days (>{high_demand_threshold:.0f} rentals): {high_demand_days} days ({high_demand_days/len(df)*100:.1f}%)")

print("\n✅ STEP 3 COMPLETE: Comprehensive EDA analysis finished.")
print("🔍 Ready for Step 4: Data Preparation and Preprocessing.")
# ============================================================================

# ============================================================================
# STEP 4: DATA PREPARATION AND PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("🛠️ STEP 4: DATA PREPARATION AND PREPROCESSING")
print("="*70)

# 4.1 Feature Engineering: Extract time-based features
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.weekday
df['dayofyear'] = df['datetime'].dt.dayofyear

print("✅ Time-based features extracted")

# 4.2 Drop unnecessary columns (including string categorical columns)
drop_cols = ['instant', 'datetime', 'casual', 'registered', 'day_category', 'month_name', 'day_type', 'day_of_year']
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
        print(f"Dropped: {col}")

print(f"Remaining columns: {df.columns.tolist()}")

# 4.3 Define features and target (only numeric features)
feature_cols = [col for col in df.columns if col not in ['count']]
target_col = 'count'

print(f"\nFeatures for modeling: {feature_cols}")
print(f"Target variable: {target_col}")
print(f"Number of features: {len(feature_cols)}")

# 4.4 Verify all features are numeric
print("\nData types check:")
for col in feature_cols:
    print(f"{col:15}: {df[col].dtype}")

# 4.5 Check for missing values
print(f"\nMissing values check:")
missing_vals = df[feature_cols + [target_col]].isnull().sum()
print(missing_vals)

if missing_vals.sum() > 0:
    print("⚠️ Found missing values - handling...")
    # Fill missing values if any
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Filled missing values in {col} with median")
else:
    print("✅ No missing values found")

# 4.6 Prepare data for modeling
X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"\nData shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# 4.7 Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\nTrain-Test Split Results:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Features: {X_train.shape[1]}")

# 4.8 Feature scaling (standardization)
from sklearn.preprocessing import StandardScaler

print(f"\nApplying StandardScaler...")
scaler = StandardScaler()

# Fit scaler on training data and transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature scaling completed")

# Convert back to DataFrames for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)

# 4.9 Verify scaling results
print(f"\nScaling verification:")
print(f"Training data mean (should be ~0): {X_train_scaled.mean().mean():.6f}")
print(f"Training data std (should be ~1): {X_train_scaled.std().mean():.6f}")

# 4.10 Final data summary
print(f"\nFinal preprocessed data summary:")
print(f"Features: {list(X_train_scaled.columns)}")
print(f"Training samples: {len(X_train_scaled)}")
print(f"Testing samples: {len(X_test_scaled)}")
print(f"Target variable range: {y.min()} to {y.max()}")

print("\n" + "="*70)
print("✅ STEP 4 COMPLETE: Data preprocessing finished successfully")
print("🚀 Ready for Step 5: Model Building and Evaluation")
print("="*70)
# ============================================================================
# Note: The code above is a complete script for data loading, visualization, EDA, and preprocessing.
# Ensure you have the necessary libraries installed and the dataset path is correct.
# You can run this script in a Python environment with access to the required libraries.
# ============================================================================

# ============================================================================
# STEP 5: MODEL BUILDING AND EVALUATION (COMPREHENSIVE)
# ============================================================================

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

print("\n" + "="*70)
print("🚀 STEP 5: MODEL BUILDING AND EVALUATION")
print("="*70)

# 5.1 Define regression models
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(
        max_iter=100,
        random_state=42
    ),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(
        alpha=1.0,
        random_state=42
    ),
    'Lasso Regression': Lasso(
        alpha=1.0,
        random_state=42
    ),
    'Decision Tree': DecisionTreeRegressor(
        random_state=42,
        max_depth=20
    )
}

print(f"📊 Models to evaluate: {len(models)}")
print("🎯 All models configured for regression task")

# 5.2 Model training and evaluation function
def evaluate_regression_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a regression model with comprehensive metrics"""
    
    print(f"\n{'='*50}")
    print(f"🔥 Training: {model_name}")
    print(f"{'='*50}")
    
    # Training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Training metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    # Testing metrics
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"📈 TRAINING PERFORMANCE:")
    print(f"   MSE:    {train_mse:.2f}")
    print(f"   RMSE:   {train_rmse:.2f}")
    print(f"   MAE:    {train_mae:.2f}")
    print(f"   R²:     {train_r2:.4f}")
    
    print(f"📊 TESTING PERFORMANCE:")
    print(f"   MSE:    {test_mse:.2f}")
    print(f"   RMSE:   {test_rmse:.2f}")
    print(f"   MAE:    {test_mae:.2f}")
    print(f"   R²:     {test_r2:.4f}")
    
    # Model performance visualization
    plt.figure(figsize=(15, 5))
    
    # Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Count')
    plt.ylabel('Predicted Count')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(1, 3, 2)
    residuals = y_test - y_pred_test
    plt.scatter(y_pred_test, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Count')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot - {model_name}')
    plt.grid(True, alpha=0.3)
    
    # Distribution of residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, color='orange')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution - {model_name}')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'training_time': training_time,
        'model_object': model,
        'predictions': y_pred_test
    }

# 5.3 Train and evaluate all models
print("\n" + "-"*50)
print("🏃‍♂️ TRAINING AND EVALUATING ALL MODELS")
print("-"*50)

results = []

for model_name, model in models.items():
    try:
        result = evaluate_regression_model(
            model, model_name, 
            X_train_scaled, y_train, 
            X_test_scaled, y_test
        )
        results.append(result)
    except Exception as e:
        print(f"❌ Error training {model_name}: {str(e)}")

# 5.4 Model comparison and ranking
print("\n" + "="*70)
print("🏆 MODEL COMPARISON AND RANKING")
print("="*70)

if results:
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('test_r2', ascending=False)
    
    print("📊 MODEL PERFORMANCE COMPARISON (Ranked by Test R²):")
    print("="*90)
    print(f"{'Model':<25} {'Train RMSE':<12} {'Test RMSE':<12} {'Train R²':<10} {'Test R²':<10} {'Time(s)':<8}")
    print("-"*90)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['model_name']:<25} {row['train_rmse']:<12.2f} {row['test_rmse']:<12.2f} "
              f"{row['train_r2']:<10.4f} {row['test_r2']:<10.4f} {row['training_time']:<8.2f}")
    
    # Best model analysis
    best_model_row = comparison_df.iloc[0]
    best_model_name = best_model_row['model_name']
    
    print(f"\n🥇 BEST MODEL: {best_model_name}")
    print(f"   Test R²: {best_model_row['test_r2']:.4f}")
    print(f"   Test RMSE: {best_model_row['test_rmse']:.2f}")
    print(f"   Test MAE: {best_model_row['test_mae']:.2f}")
    
    # Model comparison visualization
    plt.figure(figsize=(12, 8))
    
    # R² comparison
    plt.subplot(2, 2, 1)
    x_pos = range(len(comparison_df))
    plt.bar(x_pos, comparison_df['test_r2'], color='skyblue', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Test R²')
    plt.title('Model Comparison - R² Score')
    plt.xticks(x_pos, comparison_df['model_name'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # RMSE comparison
    plt.subplot(2, 2, 2)
    plt.bar(x_pos, comparison_df['test_rmse'], color='lightcoral', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Test RMSE')
    plt.title('Model Comparison - RMSE')
    plt.xticks(x_pos, comparison_df['model_name'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    plt.bar(x_pos, comparison_df['training_time'], color='lightgreen', alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Model Comparison - Training Time')
    plt.xticks(x_pos, comparison_df['model_name'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Performance radar chart (for best model)
    plt.subplot(2, 2, 4)
    metrics = ['R²', 'Low RMSE', 'Low MAE', 'Speed']
    # Normalize metrics for radar chart (higher is better)
    normalized_scores = [
        best_model_row['test_r2'],  # R² (0-1, higher better)
        1 - (best_model_row['test_rmse'] / 2000),  # Normalized RMSE (lower better)
        1 - (best_model_row['test_mae'] / 1000),   # Normalized MAE (lower better)
        1 - (best_model_row['training_time'] / 10)  # Normalized time (lower better)
    ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    normalized_scores += normalized_scores[:1]  # Complete the circle
    angles += angles[:1]
    
    plt.polar(angles, normalized_scores, 'o-', linewidth=2, color='blue')
    plt.fill(angles, normalized_scores, alpha=0.25, color='blue')
    plt.xticks(angles[:-1], metrics)
    plt.ylim(0, 1)
    plt.title(f'Performance Profile - {best_model_name}')
    
    plt.tight_layout()
    plt.savefig('model_comparison_dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()

# 5.5 Final project summary
print("\n" + "="*70)
print("🎯 FINAL PROJECT SUMMARY")
print("="*70)

if results:
    final_summary = {
        'Project': 'Bike Sharing Demand Prediction',
        'Dataset Size': f"{len(df)} samples",
        'Features': len(feature_cols),
        'Models Evaluated': len(results),
        'Best Model': best_model_name,
        'Best R² Score': f"{best_model_row['test_r2']:.4f}",
        'Best RMSE': f"{best_model_row['test_rmse']:.2f}",
        'Problem Type': 'Regression',
        'Evaluation Method': 'Train-Test Split (80-20)',
        'Feature Scaling': 'StandardScaler Applied',
        'Status': 'Ready for Submission ✅'
    }
    
    print("🏆 PROJECT ACHIEVEMENTS:")
    print("="*40)
    for key, value in final_summary.items():
        print(f"  {key:<20}: {value}")
    
    # Performance interpretation
    r2_score = best_model_row['test_r2']
    if r2_score >= 0.8:
        performance_level = "Excellent"
    elif r2_score >= 0.7:
        performance_level = "Good"
    elif r2_score >= 0.6:
        performance_level = "Fair"
    else:
        performance_level = "Needs Improvement"
    
    print(f"\n💡 MODEL PERFORMANCE: {performance_level}")
    print(f"📊 The model explains {r2_score*100:.1f}% of the variance in bike demand")
    print(f"🎯 Average prediction error: ±{best_model_row['test_mae']:.0f} bikes")

print(f"\n🚀 CONGRATULATIONS! BIKE SHARING DEMAND PREDICTION PROJECT COMPLETE!")
print(f"📅 Ready for internship submission: July 1st, 2025")
print(f"📊 Both ML projects (Human Activity + Bike Sharing) completed successfully")
print("="*70)
