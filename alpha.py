import math
from csv import reader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


m_1 = 37
m_2 = 63
n = 100
ans=0
df = pd.read_csv('SimPar.csv')
T_L = df["T_L"]
# T_R = df['T_R']
y = df['y']
nu = df['nu']

prod_y = 1

for i in range(n):
    prod_y *= y[i]



def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(100):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum




def f(x):
    #pdf of alpha is taken as gamma(0.9,1/3)
    # a = 1
    # beta = 1/3
    lamb = 1/3
    b = 0.5
    a_0 = 1
    a_1 = 5
    a_2 = 7.5
    # pi_x = ((beta)**(a))*(x**(a-1))*(math.exp(-beta*x)) / (math.gamma(a))
    pi_x = lamb * (math.exp(-lamb*x))
    p_y = prod_y**(x-1)

   
    num = pi_x*p_y*(x**(m_1+m_2))
    d = Delta(x)
    # print(d)
    den = ((b + d)**(a_0 + m_1 + m_2)) * ((b + d)**(a_1 + m_1)) * ((b + d)**(a_2 + m_2))
    # if x==0:
    #     return (num/den)
    return (num/den)


#print(f(0))
i=0


x = np.arange(2,6,0.001)
y = [f(t) for t in x]
plt.plot(x,y)
plt.show()







# while(1):
#     U = np.random.uniform(0,1)
#     E1 = np.random.exponential(1)
#     E2 = np.random.exponential(1)
#     E = E1 + E2
#     #for D
#     U2 = np.random.uniform(0,1)
#     D = math.ceil(math.sqrt(6/(math.pi*math.pi*U2)))
#     # print(U2)
#     # print(D)
#     Z = E/D
#     Y = math.exp(-Z)
#     X = (U*Z)/(1-Y)

#     print(i)
#     i +=1
#     if(Y <= f(X)):
#         ans=X
#         break

# print(ans)