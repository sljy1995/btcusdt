import pandas as pd
from data_prep import read_data
from data_analysis.analytics import vol_cal, fit_to_har, test_sig_up, fit_garch_on_resid
from data_analysis.plots import plot_log_returns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

if __name__ == "__main__":
    data = read_data("binance_data/")
    # Vectorised log returns (faster & cleaner)
    data["log_return"] = np.log(data["close"]).diff()
    data = vol_cal(data, window=5)
    data = vol_cal(data, window=60)
    data = vol_cal(data, window=1440)
    print(data['log_return'].describe())
    print(data['log_rv_5m_ann'].describe())
    train_df, test_df, model = fit_to_har(data, long="1440m")
    train_resid = train_df["log_target_rv"] - train_df["log_rv_hat"]

    # Compute and print ACF of residuals and squared residuals
    acf_resid = acf(train_resid.dropna(), nlags=20)
    acf_resid_sq = acf(train_resid**2, nlags=50, fft=True)
    # 95% confidence band
    T = len(train_resid)
    conf = 1.96 / np.sqrt(T)
    print("ACF (residuals) first 5 lags:", acf_resid[1:6])
    print("ACF (squared residuals) first 5 lags:", acf_resid_sq[1:6])
    print("95% conf band:", conf)

    plot_acf(train_resid, lags=50)
    plt.title("ACF of HAR Residuals")
    plt.show()

    plot_acf(train_resid**2, lags=50)
    plt.title("ACF of Squared HAR Residuals")
    plt.show()

    # Mean dependence
    lb_resid = acorr_ljungbox(train_resid, lags=[5, 10, 20], return_df=True)

    # Variance dependence
    lb_resid_sq = acorr_ljungbox(train_resid**2, lags=[5, 10, 20], return_df=True)

    print("Ljung-Box on residuals:\n", lb_resid)
    print("Ljung-Box on squared residuals:\n", lb_resid_sq)

    rmse = np.sqrt(mean_squared_error(test_df["log_target_rv"], test_df["log_rv_hat"]))
    mae = mean_absolute_error(test_df["log_target_rv"], test_df["log_rv_hat"])
    mse = mean_squared_error(test_df["log_target_rv"], test_df["log_rv_hat"])

    rel_rmse = rmse / np.abs(test_df["log_target_rv"].mean())
    rel_mae = mae / np.abs(test_df["log_target_rv"].mean())
    rel_rmse, rel_mae


    print(model.summary())
    print(f"Out-of-sample RMSE: {rmse:.6f}", f"MAE: {mae:.6f}", f"MSE: {mse:.6f}")
    print(f"Out-of-sample relative RMSE: {rel_rmse:.6f}", f"relative MAE: {rel_mae:.6f}")
    test, mask = test_sig_up(test_df)

    p = test.loc[mask, "vol_up"].mean()
    n = mask.sum()

    print("N signals:", n)
    print("P(vol rose | S > 1.2) =", p)
    print("Base rate P(vol rose) =", test["vol_up"].mean())

    """fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        test_df["log_target_rv"],
        test_df["log_rv_hat"],
        alpha=0.4
    )

    lims = [
        min(test_df["log_target_rv"].min(), test_df["log_rv_hat"].min()),
        max(test_df["log_target_rv"].max(), test_df["log_rv_hat"].max())
    ]
    ax.plot(lims, lims, linestyle="--")

    ax.set_xlabel("Actual log RV")
    ax.set_ylabel("Predicted log RV")
    ax.set_title("HAR Calibration (Test Set)")
    plt.show()"""


    """plot_df = test_df.iloc[::50]  # downsample for readability
    plt.figure(figsize=(12,4))
    plt.plot(plot_df["target_rv"], label="Actual RV(t+5)")
    plt.plot(plot_df["rv_hat"], label="Predicted RV (HAR)", alpha=0.8)
    plt.title("HAR: Predicted vs Actual RV (Out-of-sample)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show() """
