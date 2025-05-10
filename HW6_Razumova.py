import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import toeplitz
from numpy.linalg import inv

M = 100 # размерность x
K = 10
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

# 1)
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

# 2)
mu_1 = np.random.randn(M,1) # "белый шум"

phi = B@mu_1
psi = A@phi + S@np.random.randn(N,1) # измерение
fig, ax4 = plt.subplots()
grid = abs((len(phi)-len(psi))//2)
ax4.plot(phi, label = 'Исходный сигнал')
ax4.plot(psi[grid:-grid], label = 'Измерение')
ax4.legend()

# Каноническая калибровочная информация (G, H)
G = psi@np.transpose(phi)
H = phi@np.transpose(phi)

# 3)
A0 = G@inv(H)
alpha = np.trace(inv(H)@F)
J = alpha*S
Q = (alpha+1)*inv(np.transpose(A0)@inv(S)@A0 + inv(F))
x_est = Q@np.transpose(A0)@inv((alpha+1)*S)@y
sigma_est = np.sqrt(np.diag(Q))

fig, ax5 = plt.subplots()
ax5.plot(x, label = 'Исходный сигнал x')
ax5.plot(x_est, label = 'Оценка x', color = 'red')
ax5.plot(x_est.squeeze() + sigma_est, '-.', label = 'Стандартное отклонение', color = 'green')
ax5.plot(x_est.squeeze() - sigma_est, '-.', color = 'green')
ax5.set_title('Оценка x без априорной информации (одно измерение)')
ax5.legend()
