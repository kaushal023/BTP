import math
from csv import reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#OLD H values
# ~0
# 6.5
# 16

m_1 = 16
m_2 = 84
n = 100

df = pd.read_csv('SimPar.csv')
T_L = df["T_L"]
# T_R = df['T_R']
y = df['y']
nu = df['nu']


def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(100):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum


# print(T_L)

alpha = 2
N= 1 #For E-bayesian, increase N=1000 & n=1000
sum_0 = 0
sum_1 = 0
sum_2 = 0
sum_0_1 = 0
sum_1_1 = 0
sum_2_1 = 0


# print(k)
# a = np.random.gamma(0.1,1/3)
# b = np.random.gamma(0.1,1/3)
a = 0.01
b = 0.01

n = 1000
lambda_0 = np.random.gamma(a,1/(b + Delta(alpha)), n)
lambda_0_1 = np.random.gamma(a + m_1 + m_2,1/(b + Delta(alpha)), n)
lambda_1 = np.random.gamma(a + m_1, 1/(b + Delta(alpha)), n)
lambda_2 = np.random.gamma(a + m_2, 1/(b + Delta(alpha)), n)


h = []
h_1 = []


for i in range(n):
    base = (lambda_0[i]+lambda_1[i]+lambda_2[i])/(lambda_1[i]+lambda_2[i])
    base_1 = (1+ lambda_1[i]/lambda_0_1[i] + lambda_2[i]/lambda_0_1[i] )/(lambda_1[i]+lambda_2[i])
    h.append(math.pow(base,m_1+m_2))
    h_1.append(math.pow(base_1, m_1+m_2))

num_0 = 0
num_0_1 = 0
num_1 = 0
num_1_1 = 0
num_2 = 0
num_2_1 = 0
denom = 0
denom_1 = sum(h_1)


arr = h_1

newList = [x / denom_1 for x in arr]

newList.sort(reverse=True)
# print(newList)
print(max(newList))

index = []

# for i in range(n):
#     if(newList[i] >= 0.2):
#         index.append(i)
#         # print(newList[i])
#         h_1[i] = 0


for i in range(n):
    num_0 += lambda_0[i]*h[i]
    num_1 += lambda_1[i]*h[i]
    num_2 += lambda_2[i]*h[i]
    denom += h[i]
    num_0_1 += lambda_0_1[i]*h_1[i]
    num_1_1 += lambda_1[i]*h_1[i]
    num_2_1 += lambda_2[i]*h_1[i]
    # denom_1 += h_1[i]

denom_1 = sum(h_1)
sum_0+=(num_0/denom)
sum_1+=(num_1/denom)
sum_2+=(num_2/denom)

sum_0_1+=(num_0_1/denom_1)
sum_1_1+=(num_1_1/denom_1)
sum_2_1+=(num_2_1/denom_1)

# print(h)
# print(h_1)


# print(sum_0/N)
# print(sum_1/N)
# print(sum_2/N)

print(sum_0_1)
print(sum_1_1)
print(sum_2_1)


# arr = h_1

# newList = [x / denom_1 for x in arr]

# # newList.sort(reverse=True)
# # print(newList)
# print(max(newList))

# plt.hist(newList)
# plt.show()

