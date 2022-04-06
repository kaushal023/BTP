import math
from csv import reader
import numpy as np
import pandas as pd


n = 100
T_L = []
T_R = []
y = []
nu = []
delta = []
m_0 = 0
m_1 = 0
m_2 = 0
m_3 = 0

def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(100):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum




with open('table.jpg.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        T_L.append(1980 - int(row[1]))
        T_L.append(1980 - int(row[6]))
        T_L.append(1980 - int(row[11]))
        T_R.append(2008 - int(row[1]))
        T_R.append(2008 - int(row[6]))
        T_R.append(2008 - int(row[11]))
        y.append(int(row[2]) - int(row[1]))
        y.append(int(row[7]) - int(row[6]))
        y.append(int(row[12]) - int(row[11]))
        nu.append(int(row[3]))
        nu.append(int(row[8]))
        nu.append(int(row[13]))
        delta.append(int(row[4]))
        delta.append(int(row[9]))
        delta.append(int(row[14]))
        if(int(row[4]) == 1):
            m_1 +=1
        elif(int(row[4]) == 2):
            m_2 +=1
        else:
            m_3 +=1
        if(int(row[9]) == 1):
            m_1 +=1
        elif(int(row[9]) == 2):
            m_2 +=1
        else:
            m_3 +=1
        if(int(row[14]) == 1):
            m_1 +=1
        elif(int(row[14]) == 2):
            m_2 +=1
        else:
            m_3 +=1
        
        # print(int(row[1]))
with open('table.jpg (1).csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        T_L.append(1980 - int(row[1]))
        T_L.append(1980 - int(row[6]))
        # T_L.append(1980 - int(row[11]))
        T_R.append(2008 - int(row[1]))
        T_R.append(2008 - int(row[6]))
        # T_R.append(2008 - int(row[11]))
        y.append(int(row[2]) - int(row[1]))
        y.append(int(row[7]) - int(row[6]))
        # y.append(int(row[12]) - int(row[11]))
        nu.append(int(row[3]))
        nu.append(int(row[8]))
        # nu.append(int(row[13]))
        delta.append(int(row[4]))
        delta.append(int(row[9]))
        # delta.append(int(row[14]))
        if(int(row[4]) == 1):
            m_1 +=1
        elif(int(row[4]) == 2):
            m_2 +=1
        else:
            m_3 +=1
        if(int(row[9]) == 1):
            m_1 +=1
        elif(int(row[9]) == 2):
            m_2 +=1
        else:
            m_3 +=1
        # if(int(row[14]) == 1):
        #     m_1 +=1
        # elif(int(row[14]) == 2):
        #     m_2 +=1


T_L[:] = [x / 100 for x in T_L]
T_R[:] = [x / 100 for x in T_R]
y[:] = [x / 100 for x in y]
alpha = 2.8

# print(m_2) #33
# print(m_1) #14
# print(m_3) #53
# print(Delta(alpha))

dict = {'T_L': T_L, 'T_R': T_R, 'y': y, 'nu': nu}

df = pd.DataFrame(dict)

df.to_csv('ObsPar.csv')




N= 1
sum_0 = 0
sum_1 = 0
sum_2 = 0


for k in range(N):
    # print(k)
    a = np.random.gamma(0.5,1/3)
    b = np.random.gamma(0.5,1/3)


    lambda_0 = np.random.gamma(a,1/(b + Delta(alpha)), 10000)
    lambda_1 = np.random.gamma(a + m_1, 1/(b + Delta(alpha)), 10000)
    lambda_2 = np.random.gamma(a + m_2, 1/(b + Delta(alpha)), 10000)


    h = []
    n = 10000

    for i in range(n):
        base = (lambda_0[i]+lambda_1[i]+lambda_2[i])/(lambda_1[i]+lambda_2[i])
        h.append(math.pow(base,m_1+m_2))

    num_0 = 0
    num_1 = 0
    num_2 = 0
    denom = 0
    for i in range(n):
        num_0 += lambda_0[i]*h[i]
        num_1 += lambda_1[i]*h[i]
        num_2 += lambda_2[i]*h[i]
        denom += h[i]


    sum_0+=(num_0/denom)
    sum_1+=(num_1/denom)
    sum_2+=(num_2/denom)





print(sum_0/N)
print(sum_1/N)
print(sum_2/N)

# # print(m_1/Delta(alpha))
# # print(m_2/Delta(alpha))
