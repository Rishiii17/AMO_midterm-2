import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
# FEtching tesla stock data
tesla = yf.Ticker("TSLA")
history = tesla.history(period="30d")

# plot
plt.figure()
plt.plot(history['Close'], 'b-', label='TSLA stock price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TESLA stock price for last 30 days')
plt.legend()
plt.show()

#(b)
#prepare data
X= np.arange(len(history)).reshape(-1, 1)
Y= history['Close'].values.reshape(-1, 1)
X_scaled = (X - np.mean(X)) / np.std(X)
Y_scaled = (Y - np.mean(Y)) / np.std(Y)
#stochastic gradient descent
def sgd_linear_regression(X, Y, lr=0.0001, epochs=1000):
    weights = 0.0
    Bias= 0.0

    for _ in range(epochs):
        i=  np.random.randint(0, len(X))
        Y_pred = weights * X[i] + Bias
        error = Y_pred - Y[i]
        gradient_weights = 2 * error * X[i]
        gradient_bias = 2 * error
        weights -= lr * gradient_weights
        Bias -= lr * gradient_bias
    return weights, Bias

X_mean, X_std = np.mean(X), np.std(X)
Y_mean, Y_std = np.mean(Y), np.std(Y)
weight_scaled, Bias_scaled = sgd_linear_regression(X_scaled, Y_scaled)
weights =  (Y_std / X_std) * weight_scaled
Bias = Bias_scaled * Y_std + Y_mean - weights * X_mean
print(f"Regression Model: Price = {weights} * Day + {Bias}")

# (C)Plotting data v/s regression line
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, label='Stock price')
plt.plot(X, weights * X + Bias,'r-', label='Linear regression')
plt.xlabel('Days')
plt.ylabel('Stock price')
plt.title('Stock price with linear regression')
plt.legend()
plt.grid(True)
plt.show()

# (D)Future price prediction
# noinspection PyUnresolvedReferences
F_days= np.arange(len[X], len(X)+90).reshape(-1, 1)
future_prices = weights * F_days + Bias
plt.figure(figsize=(12, 6))
plt.plot(X, Y, 'bo-', label='Historical Prices')
plt.plot(F_days, future_prices, 'r--', label='Projected Prices')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.title('Tesla Stock Price Projection')
plt.legend()
plt.grid(True)
plt.show()
print(f"Predicted price (90 days): ${float(future_prices[-1]):.2f}")