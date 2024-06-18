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
