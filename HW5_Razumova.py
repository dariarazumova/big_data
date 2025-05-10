import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import inv

M = 100 # размерность x
std = 0.5
# d - ширина PSF
# PSF - Point spread function (функция рассеяния точки)

def PSF(i):
  if i == 0:
    d = 10; al = -d; ar = d; a = np.ones(ar-al+1); # Rectangular Profile
  elif i == 1:
    d = 10; al = -d; ar = d; a = (d+1)*np.ones(ar-al+1) - abs(np.linspace(al,ar,2*d+1)); # Triangular Profile
  elif i == 2:
    d = 10; al = -d; ar = d; a = np.linspace(al,ar,2*d+1)**2; # Parabolic Profile
  elif i == 3:
    d = 10; al = -d; ar = d; a = np.exp(-np.linspace(al,ar,2*d+1)**2/(2*(d/3)**2)); # Gaussian Profile
  elif i == 4:
    d = 20; al = 0; ar = d; a = np.exp(-(np.linspace(al,ar,2*d+1))/(d/4)); # Exponential (asymmetric) Profile
  else:
   d = 10; al = -d; ar = d; a = np.random.randn(ar-al+1); # Random Profile
  return a/np.sum(a), ar, al

def matr_A(M, ar, al, a):
    N = M + ar-al # размерность y
    A = np.zeros([N,M]) # Matrix A determined by PSF a
    for j in range(M):
        for i in range(ar-al):
            A[j+i,j] = a[i]
    return A,N

#PSF generation
# 0 - rectangular, 1 - triangular, 2 - parabolic, 3 - gaussian, 4 - exponential, else - random
a, ar, al = PSF(4)
#fig, ax3 = plt.subplots()
#ax3.plot(a)
#ax3.set_title('Нормированный PSF')

S = std*np.eye(M+ar-al) # Стандартное отклонение "белого шума" 
mu = np.random.randn(M,1) # "белый шум"
#fig, ax = plt.subplots()
#ax.plot(mu)
#ax.set_title('Белый шум')

b0 = np.arange(9, 0, -1)
b = np.hstack((b0, np.zeros(M-len(b0))))
B = toeplitz(b)  # Симметричная матрица Тёплица с "треугольным" профилем
F = B@np.transpose(B) # Ковариационная матрица x Dx=F
x = B@mu

A, N = matr_A(M, ar, al, a)
y = A@x + S@np.random.randn(N,1) # измерение
#fig, ax4 = plt.subplots()
#grid = abs((len(x)-len(y))//2)
#ax4.plot(x, label = 'Исходный сигнал')
#ax4.plot(y[grid:-grid], label = 'Измерение')
#ax4.legend()

# Каноническая информация (T,v)
T = np.transpose(A)@inv(S)@A
v = np.transpose(A)@inv(S)@y

# Оценка без априорной информации
x_est = inv(T)@v
sigma_est = np.sqrt(np.diag(inv(T)))

# Оценка с априорной инфомацией
x_est_apr_inf = inv(T+inv(F))@v 
sigma_est_apr_inf = np.sqrt(np.diag(inv(T+inv(F)))) 

fig, ax5 = plt.subplots()
fig, ax6 = plt.subplots()
ax5.plot(x, label = 'Исходный сигнал x')
ax5.plot(x_est, label = 'Оценка x', color = 'red')
ax5.plot(x_est.squeeze() + sigma_est, '-.', label = 'Стандартное отклонение', color = 'green')
ax5.plot(x_est.squeeze() - sigma_est, '-.', color = 'green')
ax5.set_title('Оценка x без априорной информации (одно измерение)')
ax5.legend()
ax6.plot(x, label = 'Исходный сигнал x')
ax6.plot(x_est_apr_inf, label = 'Оценка x', color = 'red')
ax6.plot(x_est_apr_inf.squeeze() + sigma_est_apr_inf, '-.', label = 'Стандартное отклонение', color = 'green')
ax6.plot(x_est_apr_inf.squeeze() - sigma_est_apr_inf, '-.', color = 'green')
ax6.set_title('Оценка x c априорной информации (одно измерение)')
ax6.legend()

# Много измерений 
for i in range(5):
    T_mult = np.zeros([M,M])
    v_mult = np.zeros([M,1])
    a_mult, ar_mult, al_mult = PSF(i)
    A_mult, N_mult = matr_A(M, ar_mult, al_mult, a_mult)
    y_mult = A_mult@x + S@np.random.randn(N,1)
    T_i = np.transpose(A_mult)@inv(S)@A_mult
    v_i = np.transpose(A_mult)@inv(S)@y_mult

    T_mult += T_i
    v_mult += v_i

# Оценка без априорной информации
x_est_mult = inv(T_mult)@v_mult
sigma_est_mult = np.sqrt(np.diag(inv(T_mult)))

# Оценка с априорной инфомацией
x_est_apr_inf_mult = inv(T_mult + inv(F))@v_mult  
sigma_est_apr_inf_mult = np.sqrt(np.diag(inv(T_mult + inv(F))))

fig, ax7 = plt.subplots()
fig, ax8 = plt.subplots()
ax7.plot(x, label = 'Исходный сигнал x')
ax7.plot(x_est_mult, label = 'Оценка x', color = 'red')
ax7.plot(x_est_mult.squeeze() + sigma_est_mult, '-.', label = 'Стандартное отклонение', color = 'green')
ax7.plot(x_est_mult.squeeze() - sigma_est_mult, '-.', color = 'green')
ax7.set_title('Оценка x без априорной информации (пять измерений)')
ax7.legend()
ax8.plot(x, label = 'Исходный сигнал x')
ax8.plot(x_est_apr_inf_mult, label = 'Оценка x', color = 'red')
ax8.plot(x_est_apr_inf_mult.squeeze() + sigma_est_apr_inf_mult, '-.', label = 'Стандартное отклонение', color = 'green')
ax8.plot(x_est_apr_inf_mult.squeeze() - sigma_est_apr_inf_mult, '-.', color = 'green')
ax8.set_title('Оценка x c априорной информации (пять измерений)')
ax8.legend()
