import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
fd = FiniteDifference(order=2, d=1)
from scipy.signal import savgol_filter

filename = 'Data/mag10.txt'
data = np.loadtxt(filename)

t = np.linspace(0, 10, 1000)
y = data[:,0]
x = 20*np.sin(pi*t/5 - 3*pi/2)
dx = fd._differentiate(x, t)
mdx = np.abs(dx)

yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3
dy = fd._differentiate(yhat, t)

my = np.abs(y)
my = my.reshape(-1,)
mx = np.abs(x)
mx = mx.reshape(-1,)

y = y.reshape(-1,)
dy = dy.reshape(-1,)
mdy = np.abs(dy)
mdy = mdy.reshape(-1,)

X = np.stack((y, x, dx, mdx, my), axis=-1)

from pysindy.feature_library import FourierLibrary, CustomLibrary
lib_fourier = FourierLibrary(n_frequencies=2)

opt = ps.STLSQ(threshold=0.01)
model = ps.SINDy(optimizer=opt)
model.fit(X,t)
model.print()

t=t.reshape(-1,)
y=y.reshape(-1,)
my=my.reshape(-1,)

def test_model(y, t):
    x0 = y
    x1 = 20*np.sin(pi*t/5 - 3*pi/2)
    x2 = 4*pi*np.cos(pi*t/5 - 3*pi/2)
    x3 = np.abs(x2)
    x4 = np.abs(y)
    dydt = -0.814*x0 + 0.651*x1 + 2.983*x2 + 0.063*x0*x4 - 0.026*x1*x3 - 0.058*x1*x4 - 0.132*x2*x3 - 0.061*x2*x4
    return dydt

ytest_0 = 17.46060934
y_test = odeint(test_model, ytest_0, t)

plt.plot(y, x/100, 'r')
plt.plot(y_test, x/100, linewidth=2, linestyle=':')
plt.xlabel('H[A/m]', fontsize = 12)
plt.ylabel('B[T]', fontsize = 12)
plt.legend(['Ground truth' , 'Learned relation'], prop={'size': 10})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp8_NO27/model.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((y_test - y)**2)/np.mean(y**2)
print("Relative Error Test: ", relative_error_test*100, "%")

filename2 = 'Data/data_file_Loops.txt'
data = np.loadtxt(filename2, skiprows=1)

x_ax = data[:,0]
y_hystlab = data[:,1]

plt.plot(y, x/100, 'r', linewidth=2, linestyle=':')
plt.plot(y_test, x/100, 'y', linewidth=2, linestyle=':')
plt.plot(y_hystlab, x_ax/100, 'b', linewidth=2, linestyle=':')
plt.xlabel('H [A/m]', fontsize = 12)
plt.ylabel('B [T]', fontsize = 12)
plt.xlim([17.5, 17.72])
plt.ylim([0.195, 0.203])
plt.legend(['Experimental Data' ,'SINDy', 'Hystlab'], loc='upper left', prop={'size': 10})
plt.xticks(fontsize=12, rotation = '30')
plt.yticks(fontsize=12)
plt.savefig("Results/Exp8_NO27/compare2.pdf", dpi=3000,bbox_inches='tight')
plt.show()

