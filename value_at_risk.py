import numpy as np
import yfinance as yf
from scipy.stats import norm


# Function to calculate Parametric VaR
def calculate_parametric_var(returns, weights, confidence_level=0.95, initial_investment=30000):
    # Calculate the variance-covariance matrix of the asset returns
    cov_matrix = returns.cov()

    # Calculate the portfolio standard deviation
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Define the Z-score for the confidence level
    z_score = norm.ppf(confidence_level)

    # Calculate the Parametric VaR
    parametric_var = z_score * portfolio_std
    parametric_var_usd = parametric_var * initial_investment

    return parametric_var_usd, portfolio_std


# Function for Monte Carlo VaR
def monte_carlo_var(returns, weights, iterations=50000, confidence_level=0.95, initial_investment=30000):
    # Calculate portfolio expected return and standard deviation
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    portfolio_mean = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Simulate portfolio returns using Monte Carlo
    np.random.seed(42)  # Generate random number
    simulated_returns = np.random.normal(portfolio_mean, portfolio_std, iterations)

    # Calculate VaR based on simulated returns
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    monte_carlo_var_usd = var * initial_investment

    return monte_carlo_var_usd


# Main function
def main():
    # Step 1: Download data for your portfolio assets
    assets = ['^GSPC', 'AAPL', 'MSFT', 'GC=F', 'TLT']  # S&P500, Apple, Microsoft, Gold, Treasury Bonds
    data = yf.download(assets, start='2004-01-01', end='2024-08-31', interval='1mo')

    # Step 2: Calculate the monthly returns for the assets
    returns = data['Adj Close'].pct_change().dropna()

    # Step 3: New portfolio weights for higher-risk allocation
    weights = np.array([0.30, 0.15, 0.15, 0.15, 0.25])  # S&P500, AAPL, MSFT, Gold, TLT

    initial_investment = 30000  # Seed money
    confidence_level = 0.95

    # Calculate Parametric VaR
    parametric_var_usd, portfolio_std = calculate_parametric_var(returns, weights, confidence_level, initial_investment)
    print(f"Parametric VaR (95% confidence, Monthly): ${parametric_var_usd:.2f}")

    # Calculate annualized VaR correctly
    parametric_var_annual_usd = portfolio_std * np.sqrt(12) * norm.ppf(confidence_level) * initial_investment
    print(f"Parametric VaR (95% confidence, Annualized): ${parametric_var_annual_usd:.2f}")

    # Calculate Monte Carlo VaR
    monte_carlo_var_usd = monte_carlo_var(returns, weights, iterations=50000, confidence_level=confidence_level,
                                          initial_investment=initial_investment)
    print(f"Monte Carlo VaR (95% confidence, Monthly): ${monte_carlo_var_usd:.2f}")


if __name__ == "__main__":
    main()


