import numpy as np
from matplotlib import pyplot as plt

n_point = 8 #число наблюдений
#n_point = 30
#n_point = 300
#n_point = 1000

m = 5 #степень полинома + 1
a = np.array([2, 0.5, -1, 6, 3.8]) #коэффициенты   
x = np.random.rand(n_point) #сортированный вектор случайных х
y = np.zeros([n_point, 1]) #вектор y
sigma2 = 1 #sigma^2
eps = sigma2*np.random.randn(n_point) #ошибки
    
n = 0 #инициализация n
V = 0 #инициализация V
v = np.zeros([m, 1]) #инициализация v
T = np.zeros([m, m]) #инициализация T
    
for i in range(n_point):
        
    y[i] = a[0] + a[1]*x[i] + a[2]*x[i]**2 + a[3]*x[i]**3 + a[4]*x[i]**4 + eps[i]
        
    F_x_i = np.array([[1, x[i], x[i]**2, x[i]**3, x[i]**4]])

#обновление канонической информации
    n_i = 1
    n = n + n_i
        
    V_i = y[i]**2
    V = V + V_i
        
    v_i = F_x_i.transpose()*y[i]
    v = v + v_i
        
    T_i = F_x_i.transpose()*F_x_i 
    T = T + T_i
    
a_est = np.linalg.inv(T)@v #оценка коэффициентов a
x_est = np.arange(0, 1, 0.01) 
    
f_true = a[0] + a[1]*x_est + a[2]*x_est**2 + a[3]*x_est**3 + a[4]*x_est**4 #истинная функция
f_est = a_est[0] + a_est[1]*x_est + a_est[2]*x_est**2 + a_est[3]*x_est**3 + a_est[4]*x_est**4 #приближение
    
F_est = (np.array([x_est**0, x_est**1, x_est**2, x_est**3, x_est**4]).transpose())
Dfx_est = sigma2*F_est@np.linalg.inv(T)@F_est.transpose() #дисперсия известна
Dfx_2_est = ((V-v.transpose()@np.linalg.inv(T)@v)/(n_point- m)*F_est@np.linalg.inv(T)@F_est.transpose()) #дисперсия неизвестна
    
#построение графиков
fig, ax = plt.subplots()
ax.plot(x, y, '.',  color = 'blue', label = 'Измерения')
ax.plot(x_est, f_true, '-', color = 'black', label = 'Исходные данные')
ax.plot(x_est, f_est, '-', color = 'red', label = 'Приближение')
ax.plot(x_est, f_est + np.sqrt(np.diag(Dfx_est)), '-.', color = 'green', label="Ошибки (дисперсия известна)")
ax.plot(x_est, f_est - np.sqrt(np.diag(Dfx_est)), '-.', color = 'green')
ax.plot(x_est, f_est + np.sqrt(np.diag(Dfx_2_est)), '-.', color = 'orange', label="Ошибки (дисперсия неизвестна)")
ax.plot(x_est, f_est - np.sqrt(np.diag(Dfx_2_est)), '-.', color = 'orange')
ax.set_title('Число наблюдений = ' + str(n_point))
ax.set_xlabel('x')
plt.legend()
plt.show()
     
        