import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt


# Function to display summary statistics
def display_summary_stats(df):
    print("Summary Statistics:")
    print(df.describe())


# Function to plot histograms
def plot_histograms(df):
    num_assets = df.shape[1]
    fig, axes = plt.subplots(1, num_assets, figsize=(12, 5), sharey=True)

    for i, asset in enumerate(df.columns):
        axes[i].hist(df[asset], bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(asset)
        axes[i].set_xlabel('Monthly Returns')

    axes[0].set_ylabel('Frequency')
    fig.suptitle('Monthly Returns Distribution for Portfolio Assets', fontsize=16)
    plt.tight_layout()
    plt.show()


# Function to plot correlation matrix
def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    print("Correlation Matrix:")
    print(corr_matrix)

    # Plot a heatmap of the correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
    plt.title('Correlation Matrix between Portfolio Assets', fontsize=14)
    plt.show()


# Main function
def main():
    # Download financial data as portfolio
    # '^GSPC' = S&P 500 Index, 'AAPL' = Apple, 'MSFT' = Microsoft, 'GC=F' = Gold futures, 'TLT' = Long-term bonds
    assets = ['^GSPC', 'AAPL', 'MSFT', 'GC=F', 'TLT']
    data = yf.download(assets, start='2004-01-01', end='2024-08-31', interval='1mo')

    # Calculate monthly returns for all assets
    returns = data['Adj Close'].pct_change().dropna()

    # Display summary statistics
    display_summary_stats(returns)

    # Display histograms of returns
    plot_histograms(returns)

    # Display correlation matrix between assets
    plot_correlation_matrix(returns)

if __name__ == '__main__':
    main()