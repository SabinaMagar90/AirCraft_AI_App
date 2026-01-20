import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt


DATA_PATH = os.path.join("data", "aircraft-data_nov_dec.csv")
MODELS_DIR = "models"


def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize expected column names (safe)
    # Your dataset uses 'long' not 'lon', and 'alt' and 'mph'
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

    # Fill missing categorical columns if present
    for col in ["manufacturer", "model", "reg_state"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN").str.strip()
        else:
            # If not present, create placeholders to keep pipelines stable
            df[col] = "UNKNOWN"

    return df


def mode_or_unknown(s: pd.Series) -> str:
    s = s.dropna().astype(str).str.strip()
    if len(s) == 0:
        return "UNKNOWN"
    return s.value_counts().index[0]


def build_aircraft_profiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Flight count proxy: if flight column exists use count of non-null flights, else count rows
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


def save_eval_plot(results: pd.DataFrame):
    # Simple bar plot of RMSE for screenshots in report
    plt.figure()
    plt.bar(results["Model"], results["RMSE"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("RMSE")
    plt.title("Model Comparison (RMSE)")
    plt.tight_layout()
    out_path = os.path.join(MODELS_DIR, "model_rmse_comparison.png")
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


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


def main():
    ensure_dirs()

    print("Loading dataset...")
    raw = load_data(DATA_PATH)
    print("Raw shape:", raw.shape)

    print("Cleaning dataset...")
    df = clean_data(raw)
    print("Cleaned shape:", df.shape)

    print("Building aircraft profiles (aggregate by tail_number)...")
    profiles = build_aircraft_profiles(df)
    print("Profiles shape:", profiles.shape)

    # Save profiles for the web app (needed for displaying recommendations)
    profiles_path = os.path.join(MODELS_DIR, "aircraft_profiles.csv")
    profiles.to_csv(profiles_path, index=False)
    print("Saved:", profiles_path)

    # -------------------------
    # PREDICTION MODEL (Regression)
    # Predict avg_mph using other profile features
    # -------------------------
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

    print("Training & evaluating models...")
    results, best_name, best_pipe = evaluate_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    print("\nEvaluation Results:\n", results)

    plot_path = save_eval_plot(results)
    print("Saved evaluation plot:", plot_path)

    predictor_path = os.path.join(MODELS_DIR, "predictor.pkl")
    joblib.dump(best_pipe, predictor_path)
    print("Saved best predictor:", predictor_path, "| Best =", best_name)

    # -------------------------
    # RECOMMENDER (NearestNeighbors cosine similarity)
    # -------------------------
    print("Building recommender...")
    nn, rec_preprocess = build_recommender(profiles)

    nn_path = os.path.join(MODELS_DIR, "recommender_nn.pkl")
    rec_prep_path = os.path.join(MODELS_DIR, "recommender_preprocess.pkl")
    joblib.dump(nn, nn_path)
    joblib.dump(rec_preprocess, rec_prep_path)
    print("Saved recommender:", nn_path)
    print("Saved recommender preprocessor:", rec_prep_path)

    # Save evaluation table for report screenshots
    eval_path = os.path.join(MODELS_DIR, "model_evaluation.csv")
    results.to_csv(eval_path, index=False)
    print("Saved evaluation table:", eval_path)

    print("\nDONE. Now run: python app.py")


if __name__ == "__main__":
    main()
