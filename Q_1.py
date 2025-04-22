import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return (x[0]+5)**2 + (x[1]+8)**2 + (x[2]+7)**2 + 2*x[0]**2*x[1]**2 + 4*x[0]**2*x[2]**2

def g(x):
    return np.array([
        2*(x[0]+5) + 4*x[0]*x[1]**2 + 8*x[0]*x[2]**2,
        2*(x[1]+8) + 4*x[0]**2*x[1],
        2*(x[2]+7) + 8*x[0]**2*x[2]
    ])

X = np.array([1, 1, 1])
epsilon = 10^-6
maximum_iteration: int=1000
f_values=[]
for i in range(maximum_iteration):
    G=g(X)
    if np.linalg.norm(G)<epsilon:
        break
    alpha = 1.0
    while f(X - alpha * G) > f(X) - 0.5*alpha*np.dot(G, G):
        alpha *= 0.5
    X = X-alpha*G
    f_values.append(f(X))

plt.plot(f_values, 'b-o', )
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence of Steepest Descent')
plt.grid(True)
plt.show()