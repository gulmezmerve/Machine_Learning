import numpy as np
import math
import matplotlib.pyplot as plt


x = np.linspace(-20, 20, num = 100)

def discrimant_func(mean,variance,p):
  standart_sapma = math.sqrt(variance)
  g = -1/2*math.log(2*math.pi)-math.log(standart_sapma) - ((x -mean)**2)/(2*variance)
  x2 = -1/(2*variance)
  x1 = 2*mean/(2*variance)
  x0 = -(mean**2)/(2*variance)-1/2*math.log(2*math.pi)-math.log(standart_sapma)+math.log(p)
  final = np.array([x2,x1,x0])
  return final,g;

def pdf(mean,variance_2):
  variance = math.sqrt(variance_2)
  f_pdf = (1/(math.sqrt(2*math.pi)*variance))*np.exp(-((x-mean)**2/(2*variance_2)))
  return f_pdf

#Q1a solutions.
g1_coef,g1 = discrimant_func(-5,16,0.5)
g2_coef,g2 = discrimant_func(10,32,0.5)

g_coef= g2_coef-g1_coef

g_coef.tolist()
x_coef = np.roots(g_coef)
x_bound = x_coef[1]

g1_pdf = pdf(-5,16)*0.5
g2_pdf = pdf(10,32)*0.5

f,(ax1,ax2) = plt.subplots(2,sharex=True,figsize=(7.5,10))
ax1.plot(x,g1,label='g1,p=0.5')
ax1.plot(x,g2,label='g2,p=0.5')
ax1.legend()
ax1.set_xlabel('numbers')
ax1.set_ylabel('dicrimant func. value')
ax1.legend()
ax1.set_title('Discriminant Functions-a')
ax2.plot(x,g1_pdf,label='Class-1')
ax2.plot(x,g2_pdf,label='Class-1')
ax2.legend()
ax2.set_xlabel('numbers')
ax2.set_ylabel('posterior values')
ax2.plot([x_bound, x_bound],[0,0.10],label='Decision Boundry')
print(x_bound)
ax2.set_title('Posterior of classes')


#Q1b solutions.
g1_coef,g1= discrimant_func(-5,16,0.1)
g2_coef,g2 = discrimant_func(10,32,0.9)

g_coef= g2_coef-g1_coef

g_coef.tolist()
x_coef = np.roots(g_coef)
x_bound = x_coef[1]

g1_pdf = pdf(-5,16)*0.1
g2_pdf = pdf(10,32)*0.9

f,(ax1,ax2) = plt.subplots(2,sharex=True,figsize=(7.5,10))

ax1.plot(x,g1,label='g1,p=0.1')
ax1.plot(x,g2,label='g2,p=0.9')
ax1.set_xlabel('numbers')
ax1.set_ylabel('dicrimant func. value')
ax1.legend()
ax1.set_title('Discriminant Functions-b')
ax2.plot(x,g1_pdf,label='class 1')
ax2.plot(x,g2_pdf,label='class 2')
ax2.set_xlabel('numbers')
ax2.set_ylabel('posterior values')
ax2.plot([x_bound, x_bound],[0,0.10],label='Decision Boundry')
print(x_bound)
ax2.set_title('Posterior of classes-b')
ax2.legend()


#Q1c solutions.

f,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,sharex=True,figsize=(10,15))
mu1, sigma1 = -5, 4
mu2,sigma2 = 10,np.sqrt(32)
s1 = np.random.normal(mu1, sigma1, 10)
s2 = np.random.normal(mu2, sigma2, 10)
ax1.hist(s1)
ax1.hist(s2)
ax2.set_title('N=100-a')
ax1.set_title('N=10-a')
ax1.set_xlabel('numbers')
ax1.legend()
s1_100 = np.random.normal(mu1, sigma1, 100)
s2_100 = np.random.normal(mu2, sigma2, 100)
ax2.hist(s1_100)
ax2.hist(s2_100)
ax3.hist(s1)
ax3.hist(s2)
ax4.hist(s1_100)
ax4.hist(s2_100)
ax3.set_title('N=10-b')
ax4.set_title('N=100-b')
# for a section N=10
s1_mean_10 = s1.mean()
s1_std_10 = s1.std()
s2_mean_10 = s2.mean()
s2_std_10 = s2.std()
s1a_x,s1_a_func = discrimant_func(s1_mean_10,s1_std_10**2,0.5)
s2a_x,s2_a_func = discrimant_func(s2_mean_10,s2_std_10**2,0.5)
sa_coef = s2a_x-s1a_x;
sa_coef.tolist()
x_coef = np.roots(sa_coef)
ax1.plot([x_coef[1], x_coef[1]],[0,4])

# for a section N=100
s1_mean = s1_100.mean()
s1_std = s1_100.std()
s2_mean = s2_100.mean()
s2_std = s2_100.std()
s1a_x,s1_a_func = discrimant_func(s1_mean,s1_std**2,0.5)
s2a_x,s2_a_func = discrimant_func(s2_mean,s2_std**2,0.5)
sa_coef = s2a_x-s1a_x;
sa_coef.tolist()
x_coef = np.roots(sa_coef)
ax2.plot([x_coef[1], x_coef[1]],[0,30])

# for b section N=10
s1a_x,s1_a_func = discrimant_func(s1_mean_10,s1_std_10**2,0.1)
s2a_x,s2_a_func = discrimant_func(s2_mean_10,s2_std_10**2,0.9)
sa_coef = s2a_x-s1a_x;
sa_coef.tolist()
x_coef = np.roots(sa_coef)
ax3.plot([x_coef[1], x_coef[1]],[0,4])

# for b section N=100
s1a_x,s1_a_func = discrimant_func(s1_mean,s1_std**2,0.1)
s2a_x,s2_a_func = discrimant_func(s2_mean,s2_std**2,0.9)
sa_coef = s2a_x-s1a_x;
sa_coef.tolist()
x_coef = np.roots(sa_coef)
ax4.plot([x_coef[1], x_coef[1]],[0,30])
print(x_coef[1])

#Q1d solutions.
g1_x,g1 = discrimant_func(-5,16,0.4)
g2_x,g2 = discrimant_func(10,32,0.4)
g3_x,g3 = discrimant_func(+5,48,0.2)

x = np.linspace(-20, 20, num = 100)

q4d_g1_pdf = pdf(-5,16)*0.4
q4d_g2_pdf = pdf(10,32)*0.4
q4d_g3_pdf = pdf(5,48)*0.2

q13xcoef =g3_x-g1_x
q13xcoef.tolist()
x_coef = np.roots(q13xcoef)

q23xcoef = g3_x-g2_x
q23xcoef.tolist()
x_bound1 = np.roots(q23xcoef)
print(x_bound1[1])
ax5.plot(x,q4d_g1_pdf,label='class1')
ax5.plot(x,q4d_g2_pdf,label='class2')
ax5.plot(x,q4d_g3_pdf,label='class3')
ax5.plot([x_coef[1], x_coef[1]],[0,0.05],label='Decision_Boundry-1')
print(x_coef[1])
ax5.plot([x_bound1[1], x_bound1[1]],[0,0.05],label='Decision_Boundry-2')
ax5.legend()
ax5.set_title('Three classes')

plt.show()