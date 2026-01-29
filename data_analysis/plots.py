import matplotlib.pyplot as plt
import pandas as pd

def plot_log_returns(data: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(data['minute'], data['log_return'], label='Log Returns')
    plt.xlabel('Time')
    plt.ylabel('Log Return')
    plt.title('Log Returns Over Time')
    plt.legend()
    plt.show()