from ast import Del
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma, norm, truncnorm
import math
from math import log

df = pd.read_csv("etmoriginal.csv")
# y = df['exit']
# nu = [0 for i in range(1013)]
# T_L = df['entry']
# T_L = [x/100 for x in T_L]
# n = len(y)
# y = [x/100 for x in y]
# print(y)
m_1 = 0
m_2 = 0
y = []
T_L = []


def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(len(y)):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum


for i in range(len(df['cause'])):
    if(df['cause'][i] == 1):
        m_1 +=1
        y.append(df['exit'][i]/100)
        T_L.append(df['entry'][i]/100)
    elif(df['cause'][i] == 3): 
        m_2 +=1
        y.append(df['exit'][i]/100)
        T_L.append(df['entry'][i]/100)


nu = [0 for i in range(len(y))]
print(m_1)
print(m_2)

logprod_y = 1

for i in range(0,len(y)):
    logprod_y += log(y[i])


def r(l00,l10,l20,a0,l01,l11,l21,a1):
    # print('h')
    g00 = gamma.pdf(l00,a + m_1 + m_2, 0, 1/(b + Delta(a0)))
    g10 = gamma.pdf(l10, a+ m_1, 0, 1/(b + Delta(a0)))
    g20 = gamma.pdf(l20, a+ m_2, 0, 1/(b + Delta(a0)))
    g01 = gamma.pdf(l01,a + m_1 + m_2 , 0, 1/(b + Delta(a1)))
    g11 = gamma.pdf(l11, a+ m_1, 0, 1/(b + Delta(a1)))
    g21 = gamma.pdf(l21, a+ m_2, 0, 1/(b + Delta(a1)))
    ap0 = (m_1 + m_2 - 1)*log(a0) + (a0-1)*logprod_y - (a+m_1 + m_2 + a + m_1 + a + m_2)*log(b + Delta(a0))
    ap1 = (m_1 + m_2 - 1)*log(a1) + (a1-1)*logprod_y - (a+m_1 + m_2 + a + m_1 + a + m_2)*log(b + Delta(a1))
    # print(gamma.pdf(a0, a + m_1 + m_2, 0, 1/b))
    # print(gamma.pdf(a1, a + m_1 + m_2, 0, 1/b))
    # ap0 = log(gamma.pdf(a0, a + m_1 + m_2, 0, 1/b)) + (a0-1)*log(prod_y)
    # ap1 = log(gamma.pdf(a1, a + m_1 + m_2, 0, 1/b)) + (a1-1)*log(prod_y)
    if(g01 == 0 or g11== 0 or g21==0):
        return 0
    h = ((1 + l11/l01 + l21/l01)/(l11+l21)) / ((1 + l10/l00 + l20/l00)/(l10+l20))
    h = (m_1+m_2)* log(h)
    p = h + log(g01) + log(g11) + log(g21) + ap1 -log(g00) - log(g10) - log(g20) - ap0
    # p = h + log(g01) + log(g11) + log(g21) -log(g00) - log(g10) - log(g20)

    #logscale for q
    # q00 = gamma.pdf(l00,l01)
    # q10 = gamma.pdf(l10,l11)
    # q20 = gamma.pdf(l20,l21)
    # q01 = gamma.pdf(l01,l00)
    # q11 = gamma.pdf(l11,l10)
    # q21 = gamma.pdf(l21,l20)
    # q = log(q00) + log(q10) + log(q20) - log(q01) - log(q11) - log(q21)
    q00 = norm.cdf(l00)
    q10 = norm.cdf(l10)
    q20 = norm.cdf(l20)
    q01 = norm.cdf(l01)
    q11 = norm.cdf(l11)
    q21 = norm.cdf(l21)
    qa0 = norm.cdf(a0)
    qa1 = norm.cdf(a1)
    q = log(q00) + log(q10) + log(q20) + log(qa0) - log(q01) - log(q11) - log(q21) - log(qa1)
    return math.exp(p+q)


a = 0.01
b = 0.01

l_0 = [2]
l_1 = [1]
l_2 = [4]
alp = [2]

N = 10000
M = 10000
c = 0.5
sigma = 1
p = 0.05
index = int((1-p)*M)
min_l0 = 999999
min_l1 = 999999
min_l2 = 999999
min_a = 999999
ind_l0 = 0
ind_l1 = 0
ind_l2 = 0
ind_a = 0
i = 0
k = 0

while(i<N):
    
    # lo_0 = np.random.gamma(l_0[i]*l_0[i]/c, c/l_0[i])
    # lo_1 = np.random.gamma(l_1[i]*l_1[i]/c, c/l_1[i])
    # lo_2 = np.random.gamma(l_2[i]*l_2[i]/c, c/l_2[i])
    # lo_0 = np.random.normal(l_0[i], sigma)
    # lo_1 = np.random.normal(l_1[i], sigma)
    # lo_2 = np.random.normal(l_2[i], sigma)
    lo_0 = truncnorm.rvs(-l_0[i], np.inf, loc=l_0[i])
    lo_1 = truncnorm.rvs(-l_1[i], np.inf, loc=l_1[i])
    lo_2 = truncnorm.rvs(-l_2[i], np.inf, loc=l_2[i])
    alpha_n = truncnorm.rvs(-alp[i], np.inf, loc=alp[i])
    
    # if(i - index >= 0):
    #     if((l_0[i] - l_0[i-index]) < min_l0):
    #         min_l0 = l_0[i] - l_0[i-index]
    #         ind_l0 = i
    #     if((l_1[i] - l_1[i-index]) < min_l1):
    #         min_l1 = l_1[i] - l_1[i-index]
    #         ind_l1 = i
    #     if((l_2[i] - l_2[i-index]) < min_l2):
    #         min_l2 = l_2[i] - l_2[i-index]
    #         ind_l2 = i
    #     if((alp[i] - alp[i-index]) < min_a):
    #         min_a = alp[i] - alp[i-index]
    #         ind_a = i

    

    if(lo_0 <= 0 or lo_1<= 0 or lo_2<=0):
        k+=1
        continue

    # R = min(1 , (p(l_0[i], l_1[i], l_2[i], lo_0, lo_1, lo_2) * q(l_0[i], l_1[i], l_2[i], lo_0, lo_1, lo_2)) / (q(lo_0, lo_1, lo_2, l_0[i], l_1[i], l_2[i])) )
    # print(R)
    R = min(1, r(l_0[i], l_1[i], l_2[i], alp[i],  lo_0, lo_1, lo_2, alpha_n))
    U = np.random.uniform(0,1)
    if U <= R:
        # print(lo_0,lo_1,lo_2,alpha_n)
        # print("acc", lo_0,lo_1,lo_2)
        l_0.append(lo_0)
        l_1.append(lo_1)
        l_2.append(lo_2)
        alp.append(alpha_n)
        i+=1
        print(i)
    # print(i)
# print(k)


del l_0[:N-M]
del l_1[:N-M]
del l_2[:N-M]
del alp[:N-M]

tit = "Est:" + str(mean(l_0))[:7] + "," + str(mean(l_1))[:7]  + "," + str(mean(l_2))[:7] 
x = np.arange(0,M+1,1)
plt.subplot(2,2,1)
plt.plot(x, l_0)

plt.title(tit)
plt.subplot(2,2,2)
plt.plot(x, l_1)
plt.subplot(2,2,3)
plt.plot(x, l_2)
plt.subplot(2,2,4)
plt.plot(x, alp)
plt.show()


l_0.sort()
l_1.sort()   
l_2.sort()
alp.sort()


for j in range(M+1):
    if(j - index >= 0):
        if((l_0[j] - l_0[j-index]) < min_l0):
            min_l0 = l_0[j] - l_0[j-index]
            ind_l0 = j
        if((l_1[j] - l_1[j-index]) < min_l1):
            min_l1 = l_1[j] - l_1[j-index]
            ind_l1 = j
        if((l_2[j] - l_2[j-index]) < min_l2):
            min_l2 = l_2[j] - l_2[j-index]
            ind_l2 = j
        if((alp[j] - alp[j-index]) < min_a):
            min_a = alp[j] - alp[j-index]
            ind_a = j


print(mean(l_0))
print(mean(l_1))
print(mean(l_2))
print(mean(alp))

print("HPD CIs")
print("[", l_0[ind_l0-index],",", l_0[ind_l0],"]")
print("[", l_1[ind_l1-index],",", l_1[ind_l1],"]")
print("[", l_2[ind_l2-index],",", l_2[ind_l2],"]")
print("[", alp[ind_a-index],",", alp[ind_a],"]")