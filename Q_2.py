import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
# objective and constraint
def f(x1, x2):
    return (x1 - 1)**2 + 2 * (x2 - 2)**2

def h1(x1, x2):
    return 1 - x1**2 - x2**2

def h2(x1, x2):
    return x1 + x2
# generate grid
x1 = np.linspace(-2, 2, 600)
x2 = np.linspace(-2, 2, 600)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)
H1 = h1(X1, X2)
H2 = h2(X1, X2)
feasible = (H1 >= 0) & (H2 >= 0)
# Contours of f(x)
plt.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(label='f(x)')
#feasible region (h(x)>=0)
safe_sqrt = np.sqrt(np.maximum(1 - x1**2, 0))
plt.fill_between(x1, -x1, safe_sqrt, where=(1 - x1**2)>=0, alpha=0.3, label='Feasible Set')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Contour Plot of $f(x)$ and Feasible Set')
plt.legend()
plt.show()


# question 2 part-b


def objective(x):
    return (x[0] - 1)**2 + 2 * (x[1] - 2)**2
def barrier(x):
    h1= 1-x[0]**2 - x[1]**2
    h2= x[0] + x[1]**2
    if h1 <= 0 or h2 <= 0:
        epsilon = 1e-10
        if h1<= epsilon or h2<= epsilon:
         return 1e10
    return -np.log(h1) - np.log(h2)
def penalty(x, mu):
    return objective(x) + mu * barrier(x)
x0 = np.array([0.5,0.5])
mu=1.0
epsilon=0.002
x_his=[x0]
while mu >= epsilon:
    res = minimize(lambda x: penalty(x, mu), x0, method='BFGS')
    x0 = res.x                    # Update to the new minimizer
    x_his.append(x0)         # Save history
    mu /= 2         # Reduce barrier influence
print("optimal solution: ", x0)
# question 2 part c
x_his= np.array(x_his)
plt.figure(figsize=(10,6))
plt.contour(X1, X2, Z, levels=20, cmap='viridis')
plt.colorbar(label='f(x)')
safe_sqrt = np.sqrt(np.maximum(1 - x1**2, 0))
plt.fill_between(x1, -x1, safe_sqrt, where=(1 - x1**2)>0, alpha=0.3, label='Feasible Set')
plt.plot(x_his[:,0], x_his[:,1], marker='o', color='red', label='Optimization Path')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Trajectory of Solution')
plt.legend()
plt.grid()
plt.show()