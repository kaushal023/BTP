import numpy as np

alpha_1 = 3
beta_1 = 5
alpha_0 = 4
beta_0 = 6

N = 10000
sum = 0

for i in range(N):
    a = np.random.gamma(alpha_0,1/beta_0)
    b = np.random.gamma(alpha_1,1/beta_1)

    sum += np.random.gamma(a,1/b)

print(sum/N)

