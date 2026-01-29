import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import numpy as np

COLS = ["trade_id","price","qty","quote_qty","timestamp","is_buyer_maker","is_best_match"]

def infer_time_unit_from_series(ts) -> str:
    x = np.nanmedian(ts.astype("int64"))  # robust + fast
    if x < 1e11:
        return "s"
    elif x < 1e14:
        return "ms"
    elif x < 1e17:
        return "us"
    else:
        return "ns"

def trades_to_1min_bar(zip_path: Path):

    df = pd.read_csv(
        zip_path,
        compression= "zip",
        header = None,
        names = COLS
    )
    unit = infer_time_unit_from_series(df["timestamp"])
    dt = pd.to_datetime(df['timestamp'], unit = unit, utc = True)
    df["minute"] = dt.dt.floor("min")
    bars = df.groupby("minute").agg(
        open = ("price", "first"),
        high = ("price", "max"),
        low = ("price", "min"),
        close = ("price", "last"),
        volume = ("qty", "sum"),
        quote_volume = ("quote_qty", "sum"),
        n_trades = ("trade_id", "count"),
        taker_sell_qty=("qty", lambda s: s[df.loc[s.index, "is_buyer_maker"] == True].sum()),
        taker_buy_qty=("qty", lambda s: s[df.loc[s.index, "is_buyer_maker"] == False].sum())
    ).reset_index() 

    return bars


def extract_date_from_filename(filename: str) -> datetime:
    # BTCUSDT-trades-2017-08-23.zip
    date_str = filename.replace(".zip", "").split("-")[-3:]
    return datetime.strptime("-".join(date_str), "%Y-%m-%d")

def get_user_date(prompt: str) -> datetime | None:
    user_input = input(prompt).strip()
    if user_input == "":
        return None
    try:
        return datetime.strptime(user_input, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")


def read_data(folder: str) -> pd.DataFrame:
    filelist = Path(folder)
    
    start = time.perf_counter()

    start_date = get_user_date(
        "Enter start date (YYYY-MM-DD) or press Enter for earliest: "
    )
    end_date = get_user_date(
        "Enter end date   (YYYY-MM-DD) or press Enter for latest: "
    )
    
    dfs = []
    failed_files = []

    for f in sorted(filelist.glob("*.zip")):

        try:
            file_date = extract_date_from_filename(f.name)

            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            df = trades_to_1min_bar(f)
            dfs.append(df)

        except Exception as e:
            failed_files.append(f.name)
            print(f"[ERROR] Failed to process {f.name}: {e}")

    if not dfs:
        raise RuntimeError("No files were successfully processed.")

    all_trades = pd.concat(dfs, ignore_index=True)

    elapsed = time.perf_counter() - start
    print(f"Elapsed time: {elapsed:.3f} seconds")

    if failed_files:
        print(f"[WARNING] {len(failed_files)} files failed:")
        for fname in failed_files:
            print(f"  - {fname}")

    return all_trades