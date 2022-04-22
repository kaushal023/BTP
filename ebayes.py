from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma, norm, truncnorm
import math
from math import log


alpha = 1.5
lamb_0 = 1.2
lamb_1 = 0.5
lamb_2 = 1.5
size = 1000
n = 100
count = 0

df = pd.read_csv('SimulatedPar.csv')
T_L = df["T_L"]
# T_R = df['T_R']
y = df['y']
nu = df['nu']
delta = df['delta']
M = df['M']

df = pd.read_csv('Est.csv')
alp = df['alpha']
alpha = alp.iloc[-1]

m_1 = M[0]
m_2 = M[1]
m_3 = M[2]
right_ctime = 2021


def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(100):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum

prod_y = 1

for i in range(n):
    if(delta[i] == 0):
        prod_y *= y[i]




def r(l00,l10,l20,l01,l11,l21,a,b):
    # print('h')
    g00 = gamma.pdf(l00,a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
    g10 = gamma.pdf(l10, a+ m_1, 0, 1/(b + Delta(alpha)))
    g20 = gamma.pdf(l20, a+ m_2, 0, 1/(b + Delta(alpha)))
    g01 = gamma.pdf(l01,a + m_1 + m_2 , 0, 1/(b + Delta(alpha)))
    g11 = gamma.pdf(l11, a+ m_1, 0, 1/(b + Delta(alpha)))
    g21 = gamma.pdf(l21, a+ m_2, 0, 1/(b + Delta(alpha)))
    # ap0 = (m_1 + m_2 - 1)*log(a0) + (a0-1)*log(prod_y)
    # ap1 = (m_1 + m_2 - 1)*log(a1) + (a1-1)*log(prod_y)
    # print(gamma.pdf(a0, a + m_1 + m_2, 0, 1/b))
    # print(gamma.pdf(a1, a + m_1 + m_2, 0, 1/b))
    # ap0 = log(gamma.pdf(a0, a + m_1 + m_2, 0, 1/b)) + (a0-1)*log(prod_y)
    # ap1 = log(gamma.pdf(a1, a + m_1 + m_2, 0, 1/b)) + (a1-1)*log(prod_y)
    if(g01 == 0 or g11== 0 or g21==0):
        return 0
    h = ((1 + l11/l01 + l21/l01)/(l11+l21)) / ((1 + l10/l00 + l20/l00)/(l10+l20))
    h = (m_1+m_2)* log(h)
    p = h + log(g01) + log(g11) + log(g21) -log(g00) - log(g10) - log(g20)
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
    # qa0 = norm.cdf(a0)
    # qa1 = norm.cdf(a1)
    q = log(q00) + log(q10) + log(q20) - log(q01) - log(q11) - log(q21)
    return math.exp(p+q)

l0 = []
l1 = []
l2 = []


N0 = 10
for j in range(N0):

    l_0 = [lamb_0]
    l_1 = [lamb_1]
    l_2 = [lamb_2]
    

    N = 1000
    c = 0.5
    sigma = 1
    p = 0.05
    index = int((1-p)*N)
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
    a = np.random.gamma(1,1/0.5)
    b = np.random.gamma(1,1/0.5)

    while(i<N):
        # print(i)
        # lo_0 = np.random.gamma(l_0[i]*l_0[i]/c, c/l_0[i])
        # lo_1 = np.random.gamma(l_1[i]*l_1[i]/c, c/l_1[i])
        # lo_2 = np.random.gamma(l_2[i]*l_2[i]/c, c/l_2[i])
        # lo_0 = np.random.normal(l_0[i], sigma)
        # lo_1 = np.random.normal(l_1[i], sigma)
        # lo_2 = np.random.normal(l_2[i], sigma)
        lo_0 = truncnorm.rvs(-l_0[i], np.inf, loc=l_0[i])
        lo_1 = truncnorm.rvs(-l_1[i], np.inf, loc=l_1[i])
        lo_2 = truncnorm.rvs(-l_2[i], np.inf, loc=l_2[i])
        # alpha_n = truncnorm.rvs(-alp[i], np.inf, loc=alp[i])
        
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
        R = min(1, r(l_0[i], l_1[i], l_2[i], lo_0, lo_1, lo_2,a,b))
        U = np.random.uniform(0,1)
        if U <= R:
            # print(lo_0,lo_1,lo_2,alpha_n)
            # print("acc", lo_0,lo_1,lo_2)
            l_0.append(lo_0)
            l_1.append(lo_1)
            l_2.append(lo_2)
            # alp.append(alpha_n)
            i+=1
            print(i,j)
        # print(i)
    
    l0.append(mean(l_0))
    l1.append(mean(l_1))
    l2.append(mean(l_2))
    
print(mean(l0))
print(mean(l1))
print(mean(l2))

