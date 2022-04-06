import numpy as np


#First sample the lambdas
#then calculate h
#calculate avg

alpha_1 = 3
beta_1 = 5
alpha_0 = 4
beta_0 = 6

N = 10000
sum = 0

for i in range(N):
    lambda_0 = np.random.gamma(alpha_0,1/beta_0)
    lambda_1 = np.random.gamma(alpha_1,1/beta_1)
    lambda_2 = np.random.gamma(alpha_1,1/beta_1)

    h = ()


