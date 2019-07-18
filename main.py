import argparse
import pickle
from datetime import timezone
from multiprocessing.dummy import Pool as Threadpool
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray
from climata.usgs import InstantValueIO
from tqdm import tqdm


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        '--basin_file',
                        type=str,
                        help="Path to text file containing basin ids")
    parser.add_argument('-s', '--start_date', type=str, help="Start date in format yyyy-mm-dd")
    parser.add_argument('-e', '--end_date', type=str, help="End date in format yyyy-mm-dd")
    parser.add_argument('-p', '--parameter', type=str, help="USGS parameter code", required=False)
    parser.add_argument('-t', '--threads', type=int, help="Number of parallel threads", default=1)
    cfg = vars(parser.parse_args())

    cfg["basin_file"] = Path(cfg["basin_file"])
    if not cfg["basin_file"].is_file():
        raise FileNotFoundError(cfg["basin_file"])

    return cfg


def create_jobs(cfg: Dict) -> List:
    # get list of basins
    with cfg["basin_file"].open('r') as fp:
        basins = fp.readlines()
    basins = [b.strip() for b in basins]

    # check for already downloaded data
    download_dir = Path(__file__).absolute().parent / 'downloads'
    downloaded_files = list(download_dir.glob('*.p'))
    downloaded_basins = [f.name[:8] for f in downloaded_files]

    # remove these basins from the list
    basins = [b for b in basins if b not in downloaded_basins]

    jobs = []
    for basin in basins:
        job = {'station': basin, 'start_date': cfg["start_date"], 'end_date': cfg["end_date"]}
        if cfg["parameter"] is not None:
            job["parameter"] = cfg["parameter"]
        jobs.append(job)

    return jobs


def get_usgs_data(job: Dict) -> pd.DataFrame:
    data = InstantValueIO(**job)

    if (not data) or (not data[0].data):
        print(f"No data found for station {job['station']}")
        return None
    else:
        # extract data from climata time series
        data = data[0].data

        dates = []
        discharge = []
        for timestep in data:
            dates.append(timestep.date.astimezone(timezone.utc))
            discharge.append(timestep.value * 0.028316847)  # cfs/s -> m3/s

        # create data frame
        df = pd.DataFrame({'discharge': discharge}, index=pd.DatetimeIndex(dates))
        df = df.set_index('datetime')

        return df


def store_data(df: pd.DataFrame, station: str):
    out_file = Path().absolute() / "downloads" / f"{station}_hourly_discharge.nc"
    if not out_file.parent.is_dir():
        out_file.parent.mkdir(parents=True)
    arr = xarray.Dataset.from_dataframe(df)
    arr.to_netcdf(out_file)


def process_job(job: Dict):
    df = get_usgs_data(job)

    if df is not None:
        # resample 15 min values to hourly means
        df = df.resample('H').mean()

        # fill empty cells with NaNs
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # store results
        store_data(df, job["station"])


def main(cfg: Dict):
    # get list of job configurations
    jobs = create_jobs(config)

    # spawn threads
    pool = Threadpool(cfg["threads"])

    # process jobs in parallel
    for _ in tqdm(pool.imap_unordered(process_job, jobs), total=len(jobs)):
        pass


if __name__ == "__main__":
    config = get_args()
    main(config)
