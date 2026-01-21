#!/usr/bin/env python
# coding: utf-8

# # Aircraft Model Training and Building
# This notebook replaces the functionality of `train_and_build.py`.
# It handles data loading, cleaning, feature engineering, model training, and recommender system building.

# In[1]:


import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Constants
DATA_PATH = os.path.join("data", "aircraft-data_nov_dec.csv")
MODELS_DIR = "models"

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

ensure_dirs()


# In[3]:


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

print("Loading dataset...")
raw = load_data(DATA_PATH)
print("Raw shape:", raw.shape)
raw.head()


# ## Exploratory Data Analysis (Before Cleaning)

# In[4]:


# Exploratory Data Analysis (Before Cleaning)
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Plots directory exists
PLOTS_DIR = os.path.join("static", "Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_missing_values(df, stage="raw"):
    """Visualizes missing values in the dataframe."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"No missing values found in {stage} data.")
        return

    plt.figure(figsize=(10, 6))
    missing.plot(kind='bar', color='salmon')
    plt.title(f"Missing Value Counts ({stage.capitalize()})")
    plt.xlabel("Columns")
    plt.ylabel("Count")
    plt.tight_layout()
    filename = f"missing_values_{stage}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()
    print(f"Saved missing values plot: {filename}")

def plot_distributions(df, stage="raw"):
    """Plots distributions of Altitude and Speed."""
    # Speed Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df['mph'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Speed Distribution ({stage.capitalize()})")
    plt.xlabel("Speed (mph)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    filename = f"speed_distribution_{stage}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show() # Show inline
    print(f"Saved speed distribution plot: {filename}")

    # Altitude Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df['alt'].dropna(), bins=30, color='lightgreen', edgecolor='black')
    plt.title(f"Altitude Distribution ({stage.capitalize()})")
    plt.xlabel("Altitude (ft)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    filename = f"altitude_distribution_{stage}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()
    print(f"Saved altitude distribution plot: {filename}")

def plot_relationships(df, stage="raw"):
    """Plots relationships between variables."""
    # Altitude vs Speed
    plt.figure(figsize=(10, 6))
    plt.scatter(df['alt'], df['mph'], alpha=0.3, color='purple')
    plt.title(f"Altitude vs Speed ({stage.capitalize()})")
    plt.xlabel("Altitude (ft)")
    plt.ylabel("Speed (mph)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    filename = f"altitude_vs_speed_{stage}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.show()
    print(f"Saved altitude vs speed plot: {filename}")

# Run Raw EDA
print("Generating Raw Data Visualizations...")
print("Missing Value Counts (Raw):\n", raw.isnull().sum())
plot_missing_values(raw, "raw")
plot_distributions(raw, "raw")
plot_relationships(raw, "raw")


# In[5]:


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize expected column names
    expected = ["tail_number", "alt", "mph", "lat", "long"]
    for col in expected:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {df.columns.tolist()}")

    # Convert numeric columns
    for col in ["alt", "mph", "lat", "long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert timestamp if exists
    if "spotted" in df.columns:
        df["spotted"] = pd.to_datetime(df["spotted"], errors="coerce")

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows missing core fields
    df = df.dropna(subset=["tail_number", "alt", "mph", "lat", "long"])

    # Domain rules to remove unrealistic values
    df = df[(df["alt"] >= 0) & (df["alt"] <= 60000)]
    df = df[(df["mph"] >= 0) & (df["mph"] <= 700)]

    # Fill missing categorical columns
    for col in ["manufacturer", "model", "reg_state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN").str.strip()
        else:
            df[col] = "UNKNOWN"

    return df

print("Cleaning dataset...")
df = clean_data(raw)
print("Cleaned shape:", df.shape)


# ## Exploratory Data Analysis (After Cleaning)

# In[6]:


def plot_eda_after(cleaned_df, raw_df):
    print("Generating Cleaned Data Visualizations...")

    # 1. Cleaned altitude histogram
    plt.figure(figsize=(10, 6))
    plt.hist(cleaned_df["alt"], bins=30, color="skyblue", edgecolor="black")
    plt.title("Altitude Histogram (Cleaned)")
    plt.xlabel("Altitude")
    plt.ylabel("Frequency")
    plt.savefig("altitude_distribution_cleaned.png")
    print("Saved plot: altitude_distribution_cleaned.png")
    plt.show()
    plt.close()

    # 2. Cleaned speed histogram
    plt.figure(figsize=(10, 6))
    plt.hist(cleaned_df["mph"], bins=30, color="lightgreen", edgecolor="black")
    plt.title("Speed Histogram (Cleaned)")
    plt.xlabel("Speed (mph)")
    plt.ylabel("Frequency")
    plt.savefig("speed_distribution_cleaned.png")
    print("Saved plot: speed_distribution_cleaned.png")
    plt.show()
    plt.close()

    # 3. Altitude vs Speed Scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(cleaned_df["alt"], cleaned_df["mph"], alpha=0.3, s=10)
    plt.title("Altitude vs Speed (Cleaned)")
    plt.xlabel("Altitude")
    plt.ylabel("Speed")
    plt.savefig("altitude_vs_speed_cleaned.png")
    print("Saved plot: altitude_vs_speed_cleaned.png")
    plt.show()
    plt.close()

    # 4. Before vs After Speed Comparison
    plt.figure(figsize=(10, 6))
    plt.hist(raw_df["mph"].dropna(), bins=30, alpha=0.4, label="Raw", color="red", density=True)
    plt.hist(cleaned_df["mph"], bins=30, alpha=0.4, label="Cleaned", color="blue", density=True)
    plt.title("Speed Distribution: Raw vs Cleaned")
    plt.xlabel("Speed (mph)")
    plt.legend()
    plt.savefig("speed_comparison_raw_vs_cleaned.png")
    print("Saved plot: speed_comparison_raw_vs_cleaned.png")
    plt.show()
    plt.close()

plot_eda_after(df, raw)
# COMMENTS:
# Data quality has improved after cleaning. Unrealistic values and missing data have been handled.
# The distributions now reflect more realistic aircraft performance metrics.


# In[7]:


def mode_or_unknown(s: pd.Series) -> str:
    s = s.dropna().astype(str).str.strip()
    if len(s) == 0:
        return "UNKNOWN"
    return s.value_counts().index[0]

def build_aircraft_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flight count proxy
    if "flight" in df.columns:
        df["_flight_count_proxy"] = df["flight"].notna().astype(int)
        count_col = "_flight_count_proxy"
    else:
        df["_flight_count_proxy"] = 1
        count_col = "_flight_count_proxy"

    profiles = (
        df.groupby("tail_number", as_index=False)
          .agg(
              avg_alt=("alt", "mean"),
              avg_mph=("mph", "mean"),
              mean_lat=("lat", "mean"),
              mean_long=("long", "mean"),
              flight_count=(count_col, "sum"),
              manufacturer=("manufacturer", mode_or_unknown),
              model=("model", mode_or_unknown),
              reg_state=("reg_state", mode_or_unknown),
          )
    )

    profiles["log_flight_count"] = np.log1p(profiles["flight_count"])
    return profiles

print("Building aircraft profiles...")
profiles = build_aircraft_profiles(df)
print("Profiles shape:", profiles.shape)
profiles.head()


# In[8]:


# Save profiles for the web app
profiles_path = os.path.join(MODELS_DIR, "aircraft_profiles.csv")
profiles.to_csv(profiles_path, index=False)
print("Saved profiles to:", profiles_path)


# In[9]:


# PREDICTION MODEL (Regression) SETUP
target = "avg_mph"
feature_cols_num = ["avg_alt", "mean_lat", "mean_long", "log_flight_count"]
feature_cols_cat = ["manufacturer", "model", "reg_state"]
feature_cols = feature_cols_num + feature_cols_cat

X = profiles[feature_cols].copy()
y = profiles[target].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[10]:


def evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
    }

    rows = []
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        rows.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted_pipelines[name] = pipe

    results = pd.DataFrame(rows).sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    best_model_name = results.loc[0, "Model"]
    best_pipe = fitted_pipelines[best_model_name]
    return results, best_model_name, best_pipe

print("Training & evaluating models...")
results, best_name, best_pipe = evaluate_models(
    X_train, X_test, y_train, y_test, preprocessor
)
print("\nEvaluation Results:")
results


# In[11]:


# Visualization
def save_eval_plot(results: pd.DataFrame):
    plt.figure()
    plt.bar(results["Model"], results["RMSE"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("RMSE")
    plt.title("Model Comparison (RMSE)")
    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, "model_rmse_comparison.png")
    plt.savefig(out_path, dpi=180)
    plt.show() # Show inline as well
    return out_path

plot_path = save_eval_plot(results)
print("Saved evaluation plot:", plot_path)


# In[12]:


def plot_actual_vs_predicted(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    plt.figure(figsize=(8, 6))

    plt.scatter(y_test, preds, alpha=0.5, color='purple')

    # Diagonal reference line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")

    plt.xlabel("Actual Speed (mph)")
    plt.ylabel("Predicted Speed (mph)")
    plt.title(f"Actual vs Predicted Speed ({model_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    filename = "actual_vs_predicted_best_model.png"
    out_path = os.path.join(MODELS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved actual vs predicted plot: {out_path}")

print("\nVisualizing Best Model Performance...")
plot_actual_vs_predicted(best_pipe, X_test, y_test, best_name)


# In[13]:


# Save Predictor
predictor_path = os.path.join(MODELS_DIR, "predictor.pkl")
joblib.dump(best_pipe, predictor_path)
print("Saved best predictor:", predictor_path, "| Best =", best_name)

# Save evaluation table
eval_path = os.path.join(MODELS_DIR, "model_evaluation.csv")
results.to_csv(eval_path, index=False)
print("Saved evaluation table:", eval_path)


# In[14]:


# RECOMMENDER SYSTEM
def build_recommender(profiles: pd.DataFrame):
    # Feature space for recommendation (content-based)
    numeric_features = ["avg_alt", "avg_mph", "mean_lat", "mean_long", "log_flight_count"]
    categorical_features = ["manufacturer", "model", "reg_state"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X = profiles[numeric_features + categorical_features]
    X_vec = preprocessor.fit_transform(X)

    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
    nn.fit(X_vec)

    return nn, preprocessor

print("Building recommender...")
nn, rec_preprocess = build_recommender(profiles)

nn_path = os.path.join(MODELS_DIR, "recommender_nn.pkl")
rec_prep_path = os.path.join(MODELS_DIR, "recommender_preprocess.pkl")
joblib.dump(nn, nn_path)
joblib.dump(rec_preprocess, rec_prep_path)
print("Saved recommender:", nn_path)
print("Saved recommender preprocessor:", rec_prep_path)


# In[15]:


print("DONE. All models built and saved.")

