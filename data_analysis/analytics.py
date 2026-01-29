import pandas as pd
import numpy as np
import statsmodels.api as sm

def vol_cal(data: pd.DataFrame, window: int = 5, inplace: bool = False) -> pd.DataFrame:
    """
    Compute realized volatility over a rolling window from 1-minute close prices.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'close' column.
    window : int
        Rolling window in minutes (default: 5).
    inplace : bool
        If False, operate on a copy of the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with RV and annualized RV columns added.
    """

    if not inplace:
        data = data.copy()

    MIN_PER_YEAR = 365 * 24 * 60

    rv_name = f"rv_{window}m"
    rv_ann_name = f"rv_{window}m_ann"
    log_rv_ann = f"log_rv_{window}m_ann"

    # Vectorised log returns (faster & cleaner)
    data["log_return"] = np.log(data["close"]).diff()

    # Realized volatility
    data[rv_name] = np.sqrt(
        (data["log_return"] ** 2).rolling(window).sum()
    )

    # Annualised RV (crypto 24/7)
    data[rv_ann_name] = data[rv_name] * np.sqrt(MIN_PER_YEAR / window)

    # Log Volatility
    data[log_rv_ann] = np.log(np.maximum(data[rv_ann_name], 1e-8))  # avoid log(0)

    return data

def fit_to_har(df: pd.DataFrame, short: str = "5m", med: str = "60m", long: str = "1440m", split_ratio: float = 0.8,) -> pd.DataFrame:
    """
    Fit daily, weekly, and monthly realized volatilities to HAR model structure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rv_1m_ann' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with HAR columns added.
    """

    df = df.copy()

    # columns
    short_col = f"log_rv_{short}_ann"
    med_col = f"log_rv_{med}_ann"
    long_col = f"log_rv_{long}_ann"

    # target: future volatility
    df["log_target_rv"] = df[short_col].shift(-5)

    # drop NaNs

    cols = ["log_target_rv", short_col, med_col, long_col]
    df = df[cols].dropna()

    # time split AFTER dropna
    split = int(len(df) * split_ratio)
    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()

    # fit on train
    X_train = sm.add_constant(train[[short_col, med_col, long_col]])
    y_train = train["log_target_rv"]
    model = sm.OLS(y_train, X_train).fit()

    # predict on train + test (so you can compare)
    train["log_rv_hat"] = model.predict(X_train)

    X_test = sm.add_constant(test[[short_col, med_col, long_col]])
    test["log_rv_hat"] = model.predict(X_test)

    return train, test, model

def test_sig_up(data: pd.DataFrame, thr: float = 1.2) -> pd.DataFrame:
    """
    Test if volatility actually rose given an expansion ratio exceeding a threshold of 1.2.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'rv_hat', 'rv_5m', and 'target_rv' columns.
    thr : float
        Threshold for expansion ratio - if exceeded, indicates possible upward movement of volatility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'sig_up' column added.
    """
    test = data.copy()

    test["log_S"] = test["log_rv_hat"] - test["log_rv_5m_ann"]          # forecasted expansion ratio
    test["S"] = np.exp(test["log_S"])
    test["log_A"] = test["log_target_rv"] - test["log_rv_5m_ann"]       # actual expansion ratio
    test["A"] = np.exp(test["log_A"])
    test["vol_up"] = (test["A"] > 1.0).astype(int)      # did vol actually rise?

    mask = test["S"] > thr

    return test, mask