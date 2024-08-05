"""Methods for analyzing simulation logs."""

import re

import pandas as pd
from io_collection.load.load_text import load_text


def get_wall_clock_time_from_logs(
    bucket: str,
    series_name: str,
    condition_keys: list[str],
    random_seeds: list[int],
    job_arns: dict,
    pattern: str,
) -> pd.DataFrame:
    """
    Extract wall clock times from log files.

    Parameters
    ----------
    bucket
        Name of S3 bucket for input and output files.
    series_name
        Name of simulation series.
    condition_keys
        List of condition keys.
    random_seeds
        Random seeds for simulations.
    job_arns
        Map of conditions to job ARNs.
    pattern
        Regex pattern to find wall clock time in log text.

    Returns
    -------
    :
        Dataframe of wall clock times.
    """

    wall_clock_times: list[dict] = []

    for condition_key in condition_keys:
        series_key = f"{series_name}_{condition_key}" if condition_key else series_name
        job_arn = job_arns[condition_key]

        for index, seed in enumerate(random_seeds):
            log_key = f"{series_name}/logs/{series_key}_{job_arn}:{index}.log"
            print(f"Extracting wall clock time from [ {log_key} ]")

            log = load_text(bucket, log_key)
            time = re.findall(pattern, log)[0]

            wall_clock_times.append(
                {
                    "condition": condition_key,
                    "seed": seed,
                    "time": int(time),
                }
            )

    return pd.DataFrame(wall_clock_times)
