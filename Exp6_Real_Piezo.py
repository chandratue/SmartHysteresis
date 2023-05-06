import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
fd = FiniteDifference(order=2, d=1)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
from sklearn.metrics import mean_absolute_error
import timeit
import pandas as pd

start_train = timeit.timeit()

df = pd.read_csv (r'Data/hysteresis_v_150_1hz.csv')
data = df.to_numpy()
data_col = data[:,1:4]
data_final = data_col[35000:45000,:]
t = data_final[:,0]
x = data_final[:,2]
y = data_final[:,1]

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time [s]', fontsize = 12)
ax1.set_ylabel('Voltage [V]', color=color, fontsize = 12)
ax1.plot(t, x, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Displacement [$\mu$m]', color=color, fontsize = 12)  # we already handled the x-label with ax1
ax2.plot(t, y, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp6_Real_Piezo/inp_out.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

y = y.reshape(-1,1)

x_exact = 150*np.sin(6.28318530724*t-46.2178545458)

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((x - x_exact)**2)/np.mean(x_exact**2)
print("Relative Error Test in x: ", relative_error_test*100, "%")

dx = fd._differentiate(x_exact, t)
mdx = np.abs(dx)
dy = fd._differentiate(y, t)

my = np.abs(y)
my = my.reshape(-1,)
mdy = np.abs(dy)
mdy = mdy.reshape(-1,)

y = y.reshape(-1,)
dy = dy.reshape(-1,)
t1 = dx
t2 = mdx*y
t3 = dx*my

X = np.stack((y, x, dx, mdx, my), axis=-1)

opt = ps.STLSQ(threshold=0.01)
model = ps.SINDy(optimizer=opt)
model.fit(X,t)

end_train = timeit.timeit()
print("Time training:", np.abs(end_train - start_train))

model.print()

def test_model(y, t):
    x0 = y
    x1 = 150*np.sin(6.28318530724*t-46.2178545458)
    x2 = 942.477796086*np.cos(6.28318530724*t-46.2178545458)
    x3 = np.abs(x2)
    x4 = np.abs(y)
    dydt = -0.176 - 2.388*x0 + 0.581*x1 + 0.119*x2 - 0.076*x0*x4
    return dydt

start_test = timeit.timeit()

ytest_0 = -14.030457
y_test = odeint(test_model, ytest_0, t)

end_test = timeit.timeit()
print("Time testing", np.abs(end_test - start_test))

#plt.grid(linestyle='dotted')
plt.plot(x, y, 'r')
plt.plot(x, y_test, linewidth=2, linestyle=':')
plt.xlabel('Voltage [V]', fontsize = 12)
plt.ylabel('Displacement [$\mu$m]', fontsize = 12)
plt.legend(['Ground truth' , 'Learned relation'], prop={'size': 10}, frameon=False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp6_Real_Piezo/model.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

from sklearn.metrics import r2_score
r2 = r2_score(y, y_test)
print("R2 Score: ", r2)

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((y_test - y)**2)/np.mean(y**2)
print("Relative Error Test: ", relative_error_test*100, "%")

err_t = np.abs(y-y_test)
plt.plot(t, err_t)
plt.xlabel('Time [s]', fontsize = 12)
plt.ylabel('Absolute error', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp6_Real_Piezo/error.pdf", dpi = 3000, bbox_inches='tight')
#plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y, y_test))
nrmse = rmse*100/(np.max(y)-np.min(y))
print("NRMSE: ", nrmse)