import matplotlib.pyplot as plt
import os


def plot_bars_dict(scores, metric_name, save_path, title_name=""):
    """
    Plots the average metrics for each model.

    Parameters:
    - scores (dict): Dictionary of model averages.
    - metric_name (str): Name of the metric.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scores.keys(), scores.values(), color=plt.cm.Paired.colors)

    for bar in bars:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_val, f"{y_val:.2f}", ha="center", va="bottom")

    plt.xlabel("Forecast Method")
    plt.ylabel(f"{metric_name} Values")
    plt.title(title_name)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_bars_dict_in_one(
    df,
    save_path,
):
    """
    Generate a big picture of all the metrics.
    """
    num_metrics = len(df.columns)
    cols = 4
    rows = -(-num_metrics // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    axes = axes.flatten() if num_metrics > 1 else [axes]

    for i, metric in enumerate(df.columns):
        ax = axes[i]

        s = df[metric].dropna()
        s_sorted = s.sort_values(ascending=True)

        ax.barh(list(s_sorted.index), list(s_sorted.values), color=plt.cm.Paired.colors)
        ax.set_title(metric, fontsize=20)
        ax.set_xlabel("Score", fontsize=15)
        ax.set_ylabel("Model", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=16)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()
