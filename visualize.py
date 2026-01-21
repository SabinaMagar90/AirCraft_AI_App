import os
import pandas as pd
import matplotlib.pyplot as plt
from train_and_build import clean_data

DATA_PATH = os.path.join("data", "aircraft-data_nov_dec.csv")
PLOTS_DIR = os.path.join("static", "Plots")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def plot_altitude_vs_speed(df, filename="altitude_vs_speed.png"):
    plt.figure()
    plt.scatter(df["alt"], df["mph"], alpha=0.3)
    plt.xlabel("Altitude (ft)")
    plt.ylabel("Speed (mph)")
    plt.title("Altitude vs Speed")
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def plot_speed_distribution(df, filename="speed_distribution.png"):
    plt.figure()
    df["mph"].hist(bins=40)
    plt.xlabel("Speed (mph)")
    plt.ylabel("Frequency")
    plt.title("Speed Distribution")
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def plot_altitude_distribution(df, filename="altitude_distribution.png"):
    plt.figure()
    df["alt"].hist(bins=40)
    plt.xlabel("Altitude (ft)")
    plt.ylabel("Frequency")
    plt.title("Altitude Distribution")
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def plot_top_manufacturers(df, filename="top_manufacturers.png"):
    if "manufacturer" not in df.columns:
        return

    top = df["manufacturer"].value_counts().head(10)

    plt.figure()
    top.plot(kind="bar")
    plt.xlabel("Manufacturer")
    plt.ylabel("Count")
    plt.title("Top 10 Manufacturers")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=150)
    plt.close()


def main():
    ensure_plots_dir()
    df_raw = load_data()

    print("Saving plots to static/Plots/ ...")

    # 1. EDA Before Cleaning
    print("Generating 'before cleaning' plots...")
    plot_altitude_vs_speed(df_raw, "altitude_vs_speed_before_cleaning.png")
    plot_speed_distribution(df_raw, "speed_distribution_before_cleaning.png")
    plot_altitude_distribution(df_raw, "altitude_distribution_before_cleaning.png")
    plot_top_manufacturers(df_raw, "top_manufacturers_before_cleaning.png")

    # 2. EDA After Cleaning
    print("Cleaning data for comparison...")
    df_clean = clean_data(df_raw)

    print("Generating 'after cleaning' plots...")
    plot_altitude_vs_speed(df_clean, "altitude_vs_speed_after_cleaning.png")
    plot_speed_distribution(df_clean, "speed_distribution_after_cleaning.png")
    plot_altitude_distribution(df_clean, "altitude_distribution_after_cleaning.png")
    plot_top_manufacturers(df_clean, "top_manufacturers_after_cleaning.png")

    print("All plots saved successfully!")


if __name__ == "__main__":
    main()
