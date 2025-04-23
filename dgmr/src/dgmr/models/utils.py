import os
import csv


def record_metric(filepath, metric_name, value, job_id=None):
    """Record a single metric value to a file with job_id in the filename."""
    filename = f"{metric_name}_{job_id}.txt" if job_id else f"{metric_name}.txt"
    with open(os.path.join(filepath, filename), "a") as file:
        file.write(f"{value.item()}\n")


def record_metrics_to_csv(filepath, epoch, metrics):
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            headers = ["Epoch"] + list(metrics.keys())
            writer.writerow(headers)
        row = [epoch] + [metrics[key].item() for key in metrics]
        writer.writerow(row)
