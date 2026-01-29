import pandas as pd
from data_prep import read_data
from data_analysis.analytics import vol_cal, fit_to_har, test_sig_up
from data_analysis.plots import plot_log_returns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    data = read_data("binance_data/")
    data = vol_cal(data, window=5)
    data = vol_cal(data, window=60)
    data = vol_cal(data, window=1440)
    print(data['log_return'].describe())
    print(data['log_rv_5m_ann'].describe())
    train_df, test_df, model = fit_to_har(data, long="1440m")

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


    """plot_df = test_df.iloc[::50]  # downsample for readability
    plt.figure(figsize=(12,4))
    plt.plot(plot_df["target_rv"], label="Actual RV(t+5)")
    plt.plot(plot_df["rv_hat"], label="Predicted RV (HAR)", alpha=0.8)
    plt.title("HAR: Predicted vs Actual RV (Out-of-sample)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show() """
