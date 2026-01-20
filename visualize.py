import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "aircraft-data_nov_dec.csv")
PLOTS_DIR = os.path.join("static", "Plots")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def plot_altitude_vs_speed(df):
    plt.figure()
    plt.scatter(df["alt"], df["mph"], alpha=0.3)
    plt.xlabel("Altitude (ft)")
    plt.ylabel("Speed (mph)")
    plt.title("Altitude vs Speed")
    plt.savefig(os.path.join(PLOTS_DIR, "altitude_vs_speed.png"), dpi=150)
    plt.close()


def plot_speed_distribution(df):
    plt.figure()
    df["mph"].hist(bins=40)
    plt.xlabel("Speed (mph)")
    plt.ylabel("Frequency")
    plt.title("Speed Distribution")
    plt.savefig(os.path.join(PLOTS_DIR, "speed_distribution.png"), dpi=150)
    plt.close()


def plot_altitude_distribution(df):
    plt.figure()
    df["alt"].hist(bins=40)
    plt.xlabel("Altitude (ft)")
    plt.ylabel("Frequency")
    plt.title("Altitude Distribution")
    plt.savefig(os.path.join(PLOTS_DIR, "altitude_distribution.png"), dpi=150)
    plt.close()


def plot_top_manufacturers(df):
    if "manufacturer" not in df.columns:
        return

    top = df["manufacturer"].value_counts().head(10)

    plt.figure()
    top.plot(kind="bar")
    plt.xlabel("Manufacturer")
    plt.ylabel("Count")
    plt.title("Top 10 Manufacturers")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_manufacturers.png"), dpi=150)
    plt.close()


def main():
    ensure_plots_dir()
    df = load_data()

    print("Saving plots to static/Plots/ ...")

    plot_altitude_vs_speed(df)
    plot_speed_distribution(df)
    plot_altitude_distribution(df)
    plot_top_manufacturers(df)

    print("All plots saved successfully!")


if __name__ == "__main__":
    main()
