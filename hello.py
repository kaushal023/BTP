import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma
import math

a=5
b=2

def f(x):
    return gamma.pdf(x,a,0,1/b)


def g(x):
    return (b**a)*(x**(a-1))*(math.exp(-b*x))/math.gamma(a)

x = np.arange(0,10,0.05)
y1 = [f(y) for y in x]
y2 = [g(y) for y in x]

plt.plot(x,y1,x,y2)
plt.show()