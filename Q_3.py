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
X= np.array(len(history)).reshape(-1, 1)
Y= history['Close'].values
#stochastic gradient descent
def sgd_linear_regression(X, Y, lr=0.01, epochs=1000):
    weights, Bias = 0.0, 0.0
    num_examples = len(X)
    for epoch in range(epochs):
        i=  np.random.random_integers(0, num_examples)
        Y_pred = weights * X[i] + Bias
        gradient_weights = 2* (Y_pred - Y[i])* X[i]
        gradient_bias = 2* (Y_pred - Y[i])
        weights -= lr * gradient_weights
        Bias -= lr * gradient_bias
    return weights, Bias
weights, Bias = sgd_linear_regression(X, Y)
print(f"Model : y = {weights[0]:.2f}, Bias = {Bias:.2f}")

# Plotting data v/s regression line

plt.figure(figsize=(12, 8))
plt.scatter(X, Y, label='Stock price')
plt.plot(X, weights * X + Bias, label='Linear regression')
plt.xlabel('Days')
plt.ylabel('Stock price v/s SGD Linear fit')
plt.legend()
plt.show()

# Future price prediction
F_days= np.arrange(len[X], len(X)+90).reshape(-1, 1)
future_prices = weights * F_days + Bias
print(f"Predicted price (90 days):$", future_prices[-1][0])