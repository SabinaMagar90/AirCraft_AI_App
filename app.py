import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODELS_DIR = "models"
PREDICTOR_PATH = os.path.join(MODELS_DIR, "predictor.pkl")
NN_PATH = os.path.join(MODELS_DIR, "recommender_nn.pkl")
REC_PREP_PATH = os.path.join(MODELS_DIR, "recommender_preprocess.pkl")
PROFILES_PATH = os.path.join(MODELS_DIR, "aircraft_profiles.csv")


def load_artifacts():
    if not (os.path.exists(PREDICTOR_PATH) and os.path.exists(NN_PATH) and os.path.exists(REC_PREP_PATH) and os.path.exists(PROFILES_PATH)):
        raise FileNotFoundError(
            "Model files not found. Run: python train_and_build.py first."
        )
    predictor = joblib.load(PREDICTOR_PATH)
    nn = joblib.load(NN_PATH)
    rec_preprocess = joblib.load(REC_PREP_PATH)
    profiles = pd.read_csv(PROFILES_PATH)
    return predictor, nn, rec_preprocess, profiles


predictor, nn, rec_preprocess, profiles = load_artifacts()


@app.route("/")
def index():
    # for dropdowns
    manufacturers = sorted(profiles["manufacturer"].dropna().unique().tolist())
    states = sorted(profiles["reg_state"].dropna().unique().tolist())
    return render_template("index.html", manufacturers=manufacturers, states=states)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        # Get inputs
        avg_alt = float(request.form.get("avg_alt", 0))
        mean_lat = float(request.form.get("mean_lat", profiles["mean_lat"].median()))
        mean_long = float(request.form.get("mean_long", profiles["mean_long"].median()))
        log_flight_count = float(request.form.get("log_flight_count", profiles["log_flight_count"].median()))
        manufacturer = request.form.get("manufacturer", "UNKNOWN")
        model = request.form.get("model", "UNKNOWN")
        reg_state = request.form.get("reg_state", "UNKNOWN")

        X = pd.DataFrame([{
            "avg_alt": avg_alt,
            "mean_lat": mean_lat,
            "mean_long": mean_long,
            "log_flight_count": log_flight_count,
            "manufacturer": manufacturer,
            "model": model,
            "reg_state": reg_state
        }])

        pred_mph = float(predictor.predict(X)[0])
        result = {
            "pred_mph": round(pred_mph, 2),
            "inputs": X.iloc[0].to_dict()
        }

    manufacturers = sorted(profiles["manufacturer"].dropna().unique().tolist())
    states = sorted(profiles["reg_state"].dropna().unique().tolist())
    models = sorted(profiles["model"].dropna().unique().tolist())[:500]  # avoid huge dropdown

    return render_template("predict.html", result=result, manufacturers=manufacturers, states=states, models=models)


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = None

    if request.method == "POST":
        desired_alt = float(request.form.get("desired_alt", 30000))
        desired_mph = float(request.form.get("desired_mph", 450))
        reg_state = request.form.get("reg_state", "").strip()
        manufacturer = request.form.get("manufacturer", "").strip()

        # Build query row similar to profiles features
        query = {
            "avg_alt": desired_alt,
            "avg_mph": desired_mph,
            "mean_lat": float(profiles["mean_lat"].median()),
            "mean_long": float(profiles["mean_long"].median()),
            "log_flight_count": float(profiles["log_flight_count"].median()),
            "manufacturer": manufacturer if manufacturer else "UNKNOWN",
            "model": "UNKNOWN",
            "reg_state": reg_state if reg_state else "UNKNOWN",
        }

        query_df = pd.DataFrame([query])

        # Transform query & get nearest neighbors
        query_vec = rec_preprocess.transform(query_df[["avg_alt","avg_mph","mean_lat","mean_long","log_flight_count","manufacturer","model","reg_state"]])
        distances, indices = nn.kneighbors(query_vec, n_neighbors=10)

        # distances are cosine distances -> similarity = 1 - distance
        sims = (1 - distances[0]).tolist()
        idxs = indices[0].tolist()

        recs = profiles.iloc[idxs].copy()
        recs["similarity"] = sims

        # Optional filter: if user selected manufacturer/state, show only matching if possible
        if manufacturer:
            recs = recs[recs["manufacturer"].astype(str).str.lower() == manufacturer.lower()]
        if reg_state:
            recs = recs[recs["reg_state"].astype(str).str.lower() == reg_state.lower()]

        # If filtering removed all, fall back to original top list
        if recs.shape[0] == 0:
            recs = profiles.iloc[idxs].copy()
            recs["similarity"] = sims

        recommendations = recs.sort_values("similarity", ascending=False).head(10)
        recommendations = recommendations[[
            "tail_number", "manufacturer", "model", "reg_state",
            "avg_alt", "avg_mph", "flight_count", "similarity"
        ]].to_dict(orient="records")

    manufacturers = sorted(profiles["manufacturer"].dropna().unique().tolist())
    states = sorted(profiles["reg_state"].dropna().unique().tolist())
    return render_template("recommend.html", recommendations=recommendations, manufacturers=manufacturers, states=states)


if __name__ == "__main__":
    # Run: http://127.0.0.1:5000
    app.run(debug=True)
