from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gamma, norm, truncnorm
import math
from math import log
#Generating samples
# (lambda)^(-1/alpha)*np.random.weibull(alpha,size)
# 1.4338927605047063
# 1.0514938867377324
# 1.8502519628486847
# 7.084554638262535
# 6.310770218280761
# 9.009222551985482
# 1.0226725134159995
# 0.47698717602511553
# 1.6407498458858796
# 0.9901921313410079
# 0.70256081965499
# 1.7143834808549618
# HPD CIs
# [ 3.655385505694601 , 0.1832661934746705 ]
# [ 1.2976517724366359 , 0.17565419506504243 ]
# [ 3.216273797088346 , 1.3674241742680882 ]


x = np.arange(1,200.)/50.
def weib(x,l,a):
    
    return a*l*(x)**(a-1)*np.exp(-(l*(x)**a))
    # return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def Delta(alpha):
    y_sum = 0
    t_sum = 0

    for i in range(100):
        y_sum += math.pow(y[i], alpha)
        if(nu[i]== 0):
            t_sum += (math.pow(T_L[i], alpha))

    return y_sum - t_sum


alpha = 1.2
lamb_0 = 1.2
lamb_1 = 0.5
lamb_2 = 1.5
size = 1000
n = 100
count = 0

#for AE. MSE has to be done
l0_BE = []
l1_BE = []
l2_BE = []
al_BE = []


#AL
l0_CI = []
l1_CI = []
l2_CI = []
al_CI = []


#CP
l0_count = 0
l1_count = 0
l2_count = 0
al_count = 0

NS = 100

for t in range(NS):


    x_1 = []
    x_2 = []
    y = []
    T_L = []
    nu = []
    delta = []
    M = [] #[m1,m2,m3]
    m_1 = 0
    m_2 = 0
    m_3 = 0
    right_ctime = 2021


    #Sampling installation years
    trunc_obs = 1960 + np.floor(np.random.uniform(0,10,20))
    nontrunc_obs = 1975 + np.floor(np.random.uniform(0,25,80))
    install_year = np.concatenate((trunc_obs, nontrunc_obs))
    # print(trunc_obs)
    i = 0
    for k in range(100):
        if(k < 20):
            nu.append(0)
            T_L.append((1975 - trunc_obs[k])/100)
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
            yi = min(u_0,u_1,u_2)
            if(install_year[count] + yi*100 <= right_ctime):
                y.append(yi)
                delta.append(0)
                if(x_1[count] < x_2[count]):
                    m_1 +=1
                else: m_2 +=1
            else:
                y.append((right_ctime-install_year[count])/100)
                delta.append(1)
                m_3 +=1
            
            count +=1

    M = [m_1, m_2, m_3]
    for i in range(97):
        M.append(0)

    dict = {'T_L': T_L, 'y': y, 'nu': nu, 'delta': delta, 'M': M}
    df = pd.DataFrame(dict)

    df.to_csv('SimulatedPar.csv')

    prod_y = 1

    for i in range(n):
        if(delta[i] == 0):
            prod_y *= y[i]


    # plt.hist(x_1, 10)
    # plt.show()
    # plt.hist(x_2, 10)
    # plt.show()
    # print(y)
    # print(m_1)
    # print(m_2)
    # print(m_3)

    a = 0.01
    b = 0.01


    def r(l00,l10,l20,a0,l01,l11,l21,a1):
        # print('h')
        g00 = gamma.pdf(l00,a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
        g10 = gamma.pdf(l10, a+ m_1, 0, 1/(b + Delta(alpha)))
        g20 = gamma.pdf(l20, a+ m_2, 0, 1/(b + Delta(alpha)))
        g01 = gamma.pdf(l01,a + m_1 + m_2 , 0, 1/(b + Delta(alpha)))
        g11 = gamma.pdf(l11, a+ m_1, 0, 1/(b + Delta(alpha)))
        g21 = gamma.pdf(l21, a+ m_2, 0, 1/(b + Delta(alpha)))
        ap0 = (m_1 + m_2 - 1)*log(a0) + (a0-1)*log(prod_y)
        ap1 = (m_1 + m_2 - 1)*log(a1) + (a1-1)*log(prod_y)
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




    def p(l00,l10,l20, l01,l11,l21):
        g00 = gamma.pdf(l00,a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
        g10 = gamma.pdf(l10, a+ m_1, 0, 1/(b + Delta(alpha)))
        g20 = gamma.pdf(l20, a+ m_2, 0, 1/(b + Delta(alpha)))
        g01 = gamma.pdf(l01,a + m_1 + m_2 , 0, 1/(b + Delta(alpha)))
        g11 = gamma.pdf(l11, a+ m_1, 0, 1/(b + Delta(alpha)))
        g21 = gamma.pdf(l21, a+ m_2, 0, 1/(b + Delta(alpha)))
        # print("lam",l00,l10,l20, l01,l11,l21)
        # print(g00,g10,g20, g01,g11,g21)
        if(g01 == 0 or g11== 0 or g21==0):
            return 0
        h = ((1 + l11/l01 + l21/l01)/(l11+l21)) / ((1 + l10/l00 + l20/l00)/(l10+l20))
        h = (m_1+m_2)* math.log(h)
        # print(h)
        # if h> 100:
        #     return 0
        # h = math.exp(h)
        t = h + math.log(g01) + math.log(g11) + math.log(g21) -math.log(g00) - math.log(g10) - math.log(g20)
        return math.exp(t)


    def q(l00,l10,l20, l01,l11,l21):
        g0 = gamma.pdf(l00, l01)
        g1 = gamma.pdf(l10,l11)
        g2 = gamma.pdf(l20,l21)
        # g0 = gamma.pdf(l00, a + m_1 + m_2, 0, 1/(b + Delta(alpha)))
        # g1 = gamma.pdf(l10,a+ m_2, 0, 1/(b + Delta(alpha)))
        # g2 = gamma.pdf(l20,a+ m_2, 0, 1/(b + Delta(alpha)))
        return g0*g1*g2


    l_0 = [lamb_0]
    l_1 = [lamb_1]
    l_2 = [lamb_2]
    alp = [alpha]

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
            print(i,t)
        # print(i)
    # print(k)


    # tit = "l0=" + str(lamb_0) + ",l1=" + str(lamb_1) + ",l2=" + str(lamb_2)
    # tit = tit + "Est:" + str(mean(l_0))[:7] + "," + str(mean(l_1))[:7]  + "," + str(mean(l_2))[:7] 
    # x = np.arange(0,N+2,1)
    # plt.subplot(2,2,1)
    # plt.plot(x, l_0)

    # plt.title(tit)
    # plt.subplot(2,2,2)
    # plt.plot(x, l_1)
    # plt.subplot(2,2,3)
    # plt.plot(x, l_2)
    # plt.subplot(2,2,4)
    # plt.plot(x, alp)
    # plt.show()




    l_0.sort()
    l_1.sort()   
    l_2.sort()
    alp.sort()

    for j in range(N+1):
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



    l_0.append(mean(l_0))
    l_1.append(mean(l_1))
    l_2.append(mean(l_2))
    alp.append(mean(alp))

    # dict = {'l0': l_0, 'l1': l_1, 'l2': l_2, 'alpha': alp}
    # df = pd.DataFrame(dict)

    # df.to_csv('Est.csv')




    l0_BE.append(mean(l_0))
    l1_BE.append(mean(l_1))
    l2_BE.append(mean(l_2))
    al_BE.append(mean(alp))
    print("HPD CIs")
    print("[", l_0[ind_l0-index],",", l_0[ind_l0],"]")
    print("[", l_1[ind_l1-index],",", l_1[ind_l1],"]")
    print("[", l_2[ind_l2-index],",", l_2[ind_l2],"]")
    print("[", alp[ind_a-index],",", alp[ind_a],"]")
    l0_CI.append(l_0[ind_l0] - l_0[ind_l0-index])
    l1_CI.append(l_1[ind_l1] - l_1[ind_l1-index])
    l2_CI.append(l_2[ind_l2] - l_2[ind_l2-index])
    al_CI.append(alp[ind_a] - alp[ind_a-index])

    if(lamb_0 >= l_0[ind_l0-index] and lamb_0 <= l_0[ind_l0]): l0_count +=1
    if(lamb_1 >= l_1[ind_l1-index] and lamb_1 <= l_1[ind_l1]): l1_count +=1
    if(lamb_2 >= l_2[ind_l2-index] and lamb_2 <= l_2[ind_l2]): l2_count +=1
    if(alpha >= alp[ind_a-index] and alpha <= alp[ind_a]): al_count +=1


print("Lambda0:")
print("AE:", mean(l0_BE))
print("AL:", mean(l0_CI))
print("CP:", l0_count/NS)

print("Lambda1:")
print("AE:", mean(l1_BE))
print("AL:", mean(l1_CI))
print("CP:", l1_count/NS)

print("Lambda2:")
print("AE:", mean(l2_BE))
print("AL:", mean(l2_CI))
print("CP:", l2_count/NS)

print("Alpha:")
print("AE:", mean(al_BE))
print("AL:", mean(al_CI))
print("CP:", al_count/NS)