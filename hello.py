import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma, truncnorm
import math

a=5
b=2

def f(x):
    return gamma.pdf(x,a,0,1/b)


def g(x):
    return (b**a)*(x**(a-1))*(math.exp(-b*x))/math.gamma(a)

# x = np.arange(0,10,0.05)
# y1 = [f(y) for y in x]
# y2 = [g(y) for y in x]

# plt.plot(x,y1,x,y2)
# plt.show()
x = []
y = []
# for i in range(1000):
#     # print(truncnorm.rvs(0, np.inf, loc=5))
#     x.append(truncnorm.rvs(-2, np.inf, loc=2))
#     y.append(np.random.normal(2,1))

# plt.hist(x,100)
# plt.show()
# plt.hist(y,100)
# plt.show()
trunc_obs = 1975 + np.floor(np.random.uniform(0,5,20))
nontrunc_obs = 1980 + np.floor(np.random.uniform(0,10,80))
print(trunc_obs)
t = np.concatenate((trunc_obs, nontrunc_obs))
print(t)