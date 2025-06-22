import numpy as np
import matplotlib.pyplot as plt



def f(x):
    return 240 * x ** 2 + 25.8 * x + 1



def exp_func(x):
    return np.exp(21.25 * x)


x = np.linspace(-0.1, 0.1, 400)

y_quadratic = f(x)
y_exp = exp_func(x)


plt.plot(x, y_quadratic, label='f(x) = 240x^2 + 25.8x + 1')
plt.plot(x, y_exp, label='e^{21.25x}')
plt.scatter([0, 0.05, -0.02], [1, 2.89, 0.58], color='red', label='给定的点')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()