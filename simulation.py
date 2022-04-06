from re import M
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Generating samples
# (lambda)^(-1/alpha)*np.random.weibull(alpha,size)


x = np.arange(1,200.)/50.
def weib(x,l,a):
    
    return a*l*(x)**(a-1)*np.exp(-(l*(x)**a))
    # return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


alpha = 3
lamb_0 = 2
lamb_1 = 10
lamb_2 = 15
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


#Sampling installation years
trunc_obs = 1975 + np.floor(np.random.uniform(0,5,20))
nontrunc_obs =1980 + np.floor(np.random.uniform(0,10,80))

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

    if( u_0 <= min(u_1, u_2) and u_1==u_2):
        continue
    else:
        x_1.append(min(u_0,u_1))
        x_2.append(min(u_0,u_2))
        y.append(min(u_0,u_1,u_2))
        if(x_1[count] < x_2[count]):
            m_1 +=1
        else: m_2 +=1
        count +=1



# print(y)
print(m_1)
print(m_2)

dict = {'T_L': T_L, 'y': y, 'nu': nu}
df = pd.DataFrame(dict)

df.to_csv('SimPar.csv')