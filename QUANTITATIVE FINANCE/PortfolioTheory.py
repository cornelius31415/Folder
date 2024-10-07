# Reinitialisiere die notwendigen Schritte aus dem Code, um den Plot und die Ergebnisse zu erstellen.

# Importiere benötigte Bibliotheken
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Ticker List for the stocks in my portfolio
tickers = ['CL', 
           'PG', 
           # 'KMB', 
           # 'KHC', 
            'NVDA',
           # 'CBK.DE',
           # 'QCOM'
           ]

# Download historic data
data = yf.download(tickers, start='2020-10-01', end='2024-10-01')['Adj Close']

# Calculate daily return
returns = data.pct_change().dropna()

# Amount of stocks in portfolio
num_assets = len(tickers)

# Simulating random portfolios
num_portfolios = 10000
results = np.zeros((3, num_portfolios))  # Füge eine zusätzliche Dimension für die Gewichtungen hinzu
weight_matrix = np.zeros((num_portfolios, num_assets))  # Speichert die Gewichtungen der Portfolios

# Setze den Zufallsgenerator
np.random.seed(42)

# Simulation von Portfolios
for i in range(num_portfolios):
    # Zufällige Gewichtungen für die Aktien
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    # Gewichtungen speichern
    weight_matrix[i, :] = weights
    
    # Erwartete Portfolio-Rendite und -Volatilität
    portfolio_return = np.sum(weights * returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Sharpe-Ratio (Annahme risikofreier Zinssatz = 0)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    # Ergebnisse speichern
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

# Erstelle einen DataFrame der Ergebnisse
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

# Finde das Portfolio mit der höchsten Sharpe-Ratio
max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
max_sharpe_portfolio = results_df.iloc[max_sharpe_idx]
max_sharpe_weights = weight_matrix[max_sharpe_idx, :]

# Finde das Portfolio mit der niedrigsten Volatilität
min_volatility_idx = results_df['Volatility'].idxmin()
min_volatility_portfolio = results_df.iloc[min_volatility_idx]
min_volatility_weights = weight_matrix[min_volatility_idx, :]

# Finde das Portfolio mit der höchsten Rendite
max_return_idx = results_df['Return'].idxmax()
max_return_portfolio = results_df.iloc[max_return_idx]
max_return_weights = weight_matrix[max_return_idx, :]

# Plotten der effizienten Grenze, inklusive des Punktes für das Portfolio mit der höchsten Rendite
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(max_sharpe_portfolio[1], max_sharpe_portfolio[0], color='r', marker='*', s=200, label='Max Sharpe Ratio')
plt.scatter(min_volatility_portfolio[1], min_volatility_portfolio[0], color='b', marker='*', s=200, label='Min Volatility')
plt.scatter(max_return_portfolio[1], max_return_portfolio[0], color='g', marker='*', s=200, label='Max Return')
plt.title('Efficient Frontier with Max Return Portfolio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.legend()
plt.show()


# Anzeigen der optimalen Portfolios inklusive des höchsten Return-Portfolios
print("Max Return Portfolio:")
print(f"Return: {max_return_portfolio['Return']:.2f}")
print(f"Volatility: {max_return_portfolio['Volatility']:.2f}")
print("Weights:")
for ticker, weight in zip(tickers, max_return_weights):
    print(f"{ticker}: {weight:.2%}")

print("\nMax Sharpe Ratio Portfolio:")
print(f"Return: {max_sharpe_portfolio['Return']:.2f}")
print(f"Volatility: {max_sharpe_portfolio['Volatility']:.2f}")
print("Weights:")
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"{ticker}: {weight:.2%}")

print("\nMin Volatility Portfolio:")
print(f"Return: {min_volatility_portfolio['Return']:.2f}")
print(f"Volatility: {min_volatility_portfolio['Volatility']:.2f}")
print("Weights:")
for ticker, weight in zip(tickers, min_volatility_weights):
    print(f"{ticker}: {weight:.2%}")


