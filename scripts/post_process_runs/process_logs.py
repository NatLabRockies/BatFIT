import os
import re
from collections import defaultdict

from prettyPlot.plotting import *

from batfit import logger


def extract_job_id(filename):
    """
    Extracts the numeric job identifier from a Slurm log file name.
    """
    job_id = None
    # Search for 'slurm-', followed by digits, followed by '.out'
    # The parentheses (\d+) capture the digits so we can extract them
    match = re.search(r"slurm-(\d+)\.out", filename)

    if match:
        job_id = int(match.group(1))
    else:
        # Fallback: if the filename is slightly different (e.g., just '13115792.out'),
        # just find the first sequence of numbers in the string
        fallback_match = re.search(r"(\d+)", filename)
        if fallback_match:
            job_id = int(fallback_match.group(1))

    if job_id is None:
        logger.error(f"Could not find job id in {filename}")

    return job_id


def parse_mpi_log(filepath):
    """
    Reads the log file and returns a dictionary mapping each worker ID
    to a list of their completed tasks and durations.
    """
    # Dictionary to hold data: worker_id -> [(task_index, time_elapsed), ...]
    worker_data = defaultdict(list)
    job_id = extract_job_id(filepath)

    logger.info(f"Parsing log file for {job_id} ...")

    # Regex explanation:
    # \[(\d+)\] matches the worker ID inside brackets
    # \((\d+)/\d+\) matches the task index (ignores total tasks)
    # = ([\d.]+)s matches the elapsed time as a float
    pattern = re.compile(r"\[(\d+)\] Elapsed time \((\d+)/\d+\) = ([\d.]+)s")

    with open(filepath, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                worker_id = int(match.group(1))
                task_index = int(match.group(2))
                elapsed_time = float(match.group(3))

                worker_data[worker_id].append((task_index, elapsed_time))

    # Sort tasks by task_index to ensure chronological order for later cumulative sums
    for worker in worker_data:
        worker_data[worker].sort(key=lambda x: x[0])

    return worker_data, job_id


def plot_jobs_completed(worker_data, job_id):
    """
    Creates a bar plot showing the total number of jobs completed by each worker,
    ordered by the number of jobs completed.
    """
    logger.info(f"Plotting which jobs completed for {job_id} ...")
    os.makedirs("Figures_log", exist_ok=True)
    # Calculate total jobs per worker
    job_counts = {worker: len(tasks) for worker, tasks in worker_data.items()}

    # Sort workers based on their job counts
    sorted_workers = sorted(job_counts.keys(), key=lambda w: job_counts[w])

    # Extract data for plotting
    worker_labels = [str(w) for w in sorted_workers]
    counts = [job_counts[w] for w in sorted_workers]

    # Generate the plot
    plt.figure(figsize=(12, 6))
    plt.bar(worker_labels, counts, color="steelblue")
    pretty_labels("Worker ID", "Number of Jobs Completed", 16, grid=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join("Figures_log", f"job_completed_{job_id}.png"))
    plt.close()


def plot_cumulative_time(worker_data, job_id):
    """
    Creates a 2D line plot with one line per worker.
    X-axis: Number of tasks completed.
    Y-axis: Cumulative time elapsed.
    """
    logger.info(f"Plotting execution time for {job_id} ...")
    os.makedirs("Figures_log", exist_ok=True)
    plt.figure(figsize=(12, 8))
    for worker_id, tasks in worker_data.items():
        # Extract just the times, in the sorted order
        times = [t[1] for t in tasks]

        # Calculate cumulative sum manually
        cumulative_times = []
        current_sum = 0
        for time in times:
            current_sum += time
            cumulative_times.append(current_sum)

        # X-axis is just 1, 2, 3... up to the number of tasks
        task_numbers = list(range(1, len(cumulative_times) + 1))

        plt.plot(
            task_numbers,
            cumulative_times,
            label=f"Worker {worker_id}",
            alpha=0.8,
        )

    pretty_labels(
        "Number of Tasks Completed",
        "Total Elapsed Time (seconds)",
        16,
        grid=False,
    )

    # Add a legend, placing it outside the plot area if there are many workers
    if len(worker_data) <= 20:
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("Figures_log", f"exec_time_{job_id}.png"))


def get_k_slowest_workers(worker_data, k=5):
    # Calculate the total number of jobs completed by each worker
    job_counts = {
        worker_id: len(tasks) for worker_id, tasks in worker_data.items()
    }

    # Sort the dictionary items by the job count (the second item in the tuple, x[1])
    # This sorts in ascending order by default (lowest to highest)
    sorted_workers = sorted(job_counts.items(), key=lambda x: x[1])

    # Return the first k elements
    return sorted_workers[:k]


if __name__ == "__main__":
    # Replace 'simulation.log' with your actual file path
    jobids = [13228994, 13229030, 13229042, 13229043]
    all_log_files = [f"slurm-{ids}.out" for ids in jobids]
    for log_file_path in all_log_files:
        # 1. Parse the data
        data, job_id = parse_mpi_log(log_file_path)

        # 2. Show the load balancing (Jobs per worker)
        plot_jobs_completed(data, job_id)

        # 3. Show performance over time
        plot_cumulative_time(data, job_id)

        slow_workers = get_k_slowest_workers(data, k=5)
        logger.info(f"Job {job_id}")
        for worker_id, job_count in slow_workers:
            logger.info(f"\tTask {worker_id} ({job_count})")
