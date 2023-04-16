import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
fd = FiniteDifference(order=2, d=1)

t = np.linspace(0, 10*pi, 1000)
x = np.sin(t)

dx = fd._differentiate(x, t)
mdx = np.abs(dx)

def model(y, t):
    dydt = 5*np.cos(t) - 0.25*np.abs(np.cos(t))*y - 0.5*np.cos(t)*np.abs(y)
    return dydt

y0 = 0
y = odeint(model, y0, t)

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
plt.savefig("Results/Exp2_BW_Simulated/inp_out.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

dy = fd._differentiate(y, t)
my = np.abs(y)

my = my.reshape(-1,)
y = y.reshape(-1,)
dy = dy.reshape(-1,)

X = np.stack((y, x, dx, mdx, my), axis=-1)

model = ps.SINDy()
model.fit(X,t)
model.print()

c1 = 4.999
c2 = -0.25
c3 = -0.499
def test_model(y, t):
    dydt = c1*np.cos(t) + c2*np.abs(np.cos(t))*y + c3*np.cos(t)*np.abs(y)
    return dydt

ytest_0 = 0
y_test = odeint(test_model, ytest_0, t)

plt.plot(x, y, 'r')
plt.plot(x, y_test, linewidth=2, linestyle=':')
plt.xlabel('Voltage [V]', fontsize = 12)
plt.ylabel('Displacement [$\mu$m]', fontsize = 12)
plt.legend(['Ground truth' , 'Learned relation'], prop={'size': 10})
plt.xticks(fontsize=12, rotation='45')
plt.yticks(fontsize=12)
plt.savefig("Results/Exp2_BW_Simulated/model.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((y_test - y)**2)/np.mean(y**2)
print("Relative Error Test: ", relative_error_test*100, "%")