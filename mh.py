from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma
import math
#Generating samples
# (lambda)^(-1/alpha)*np.random.weibull(alpha,size)
# 1.4338927605047063
# 1.0514938867377324
# 1.8502519628486847
# 7.084554638262535
# 6.310770218280761
# 9.009222551985482

x = np.arange(1,200.)/50.
def weib(x,l,a):
    
    return a*l*(x)**(a-1)*np.exp(-(l*(x)**a))
    # return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


alpha = 2
lamb_0 = 8
lamb_1 = 6
lamb_2 = 10
size = 1000
n = 100
count = 0

x_1 = []
x_2 = []
y = []
T_L = []
nu = []
m_1 = 0
m_2 = 0


def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(100):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum


#Sampling installation years
trunc_obs = 1975 + np.floor(np.random.uniform(0,5,20))
nontrunc_obs = 1980 + np.floor(np.random.uniform(0,10,80))

# print(trunc_obs)
i = 0
for k in range(100):
    if(k < 20):
        nu.append(0)
        T_L.append((1980 - trunc_obs[k])/100)
    else:
        nu.append(1)
        T_L.append(0)

while count < 100:

    u_0 = (lamb_0)**(-1/alpha)*np.random.weibull(alpha)
    u_1 = (lamb_1)**(-1/alpha)*np.random.weibull(alpha)
    u_2 = (lamb_2)**(-1/alpha)*np.random.weibull(alpha)

    if( u_0 < min(u_1, u_2)):
        continue
    else:
        x_1.append(min(u_0,u_1))
        x_2.append(min(u_0,u_2))
        y.append(min(u_0,u_1,u_2))
        if(x_1[count] < x_2[count]):
            m_1 +=1
        else: m_2 +=1
        count +=1

# plt.hist(x_1, 10)
# plt.show()
# plt.hist(x_2, 10)
# plt.show()
# print(y)
print(m_1)
print(m_2)

a = 0.01
b = 0.01

def p(l00,l10,l20, l01,l11,l21):
    g00 = gamma.pdf(l00,a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
    g10 = gamma.pdf(l10, a+ m_1, 0, 1/(b + Delta(alpha)))
    g20 = gamma.pdf(l20, a+ m_2, 0, 1/(b + Delta(alpha)))
    g01 = gamma.pdf(l01,a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
    g11 = gamma.pdf(l11, a+ m_1, 0, 1/(b + Delta(alpha)))
    g21 = gamma.pdf(l21, a+ m_2, 0, 1/(b + Delta(alpha)))
    h = ((1 + l11/l01 + l21/l01)/(l11+l21)) / ((1 + l10/l00 + l20/l00)/(l10+l20))
    # print(h)
    if h> 100:
        return 0
    h = h**(m_1+m_2)
    return h*g01*g11*g21 / (g00*g10*g20)


def q(l00,l10,l20, l01,l11,l21):
    g0 = gamma.pdf(l00, l01)
    g1 = gamma.pdf(l10,l11)
    g2 = gamma.pdf(l20,l21)
    # g0 = gamma.pdf(l00, a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
    # g1 = gamma.pdf(l10,a+ m_2, 0, 1/(b + Delta(alpha)))
    # g2 = gamma.pdf(l20,a+ m_2, 0, 1/(b + Delta(alpha)))
    return g0*g1*g2


l_0 = [4]
l_1 = [3]
l_2 = [5]

N = 5000
i = 0
while(i<N):
    # print(i)
    lo_0 = np.random.gamma(l_0[i], 1)
    lo_1 = np.random.gamma(l_1[i], 1)
    lo_2 = np.random.gamma(l_2[i], 1)
    # lo_0 = np.random.gamma(a + m_1 + m_2,1/(b + Delta(alpha)))
    # lo_1 = np.random.gamma(a + m_1, 1/(b + Delta(alpha)))
    # lo_2 = np.random.gamma(a + m_2, 1/(b + Delta(alpha)))
    if(lo_0 < 0.05 or lo_1 <0.05 or lo_2 < 0.05): continue
    

    R = min(1 , (p(l_0[i], l_1[i], l_2[i], lo_0, lo_1, lo_2) * q(l_0[i], l_1[i], l_2[i], lo_0, lo_1, lo_2)) / (q(lo_0, lo_1, lo_2, l_0[i], l_1[i], l_2[i])) )
    U = np.random.uniform(0,1)
    if U <= R:
        # print(lo_0,lo_1,lo_2)
        l_0.append(lo_0)
        l_1.append(lo_1)
        l_2.append(lo_2)
        i+=1
        print(i)
    # print(i)

tit = "l0=" + lamb_0 + ", l1=" + lamb_1 + ", l2=" + lamb_2

x = np.arange(0,N+1,1)
plt.subplot(2,2,1)
plt.plot(x, l_0)

plt.title(tit)
plt.subplot(2,2,2)
plt.plot(x, l_1)
plt.subplot(2,2,3)
plt.plot(x, l_2)
plt.show()

print(mean(l_0))
print(mean(l_1))
print(mean(l_2))