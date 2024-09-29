import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Function to display summary statistics
def display_summary_stats(df):
    print("Summary Statistics:")
    print(df.describe())


# Function to plot histograms
def plot_histograms(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].hist(df['S&P500'], bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('S&P500')
    axes[0].set_xlabel('Monthly Returns')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(df['AAPL'], bins=30, color='orange', edgecolor='black')
    axes[1].set_title('AAPL')
    axes[1].set_xlabel('Monthly Returns')

    plt.tight_layout()
    plt.show()


# Function to plot correlation matrix
def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix)

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
    plt.title('Correlation Matrix between S&P500 and Stock Returns', fontsize=14)
    plt.show()


# Download S&P500 and a specific stock data
sp500 = yf.download('^GSPC', start='2004-01-01', end='2024-08-31', interval='1mo')
stock = yf.download('AAPL', start='2004-01-01', end='2024-08-31', interval='1mo')

# Calculate monthly returns
sp500['Returns'] = sp500['Adj Close'].pct_change()
stock['Returns'] = stock['Adj Close'].pct_change()

# Make dataframe to store return series
return_df = pd.DataFrame({'S&P500': sp500['Returns'], 'Stock': stock['Returns']})
return_df.dropna(inplace=True)

# Display summary statistics
display_summary_stats(return_df)

# Plot histograms
plot_histograms(return_df)

# Plot and display correlation matrix
plot_correlation_matrix(return_df)