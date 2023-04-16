import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
fd = FiniteDifference(order=2, d=1)

t = np.linspace(0,10*pi,1000)
x = 4*np.sin(t)
dx = fd._differentiate(x, t)
mdx = np.abs(dx)

def model(y, t):
    dydt = 5*4*np.cos(t) - 0.25*4*np.abs(np.cos(t))*y - 0.5*4*np.cos(t)*np.abs(y)
    return dydt

y0 = 0
y = odeint(model, y0, t)
z = y**2

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Time [s]', fontsize = 12)
ax1.set_ylabel('Voltage [kV]', color=color, fontsize = 12)
ax1.plot(t, x, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Displacement [$\mu$m]', color=color, fontsize = 12)  # we already handled the x-label with ax1
ax2.plot(t, z, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp7_Butterfly/inp_out.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

dy = fd._differentiate(y, t)
dz = fd._differentiate(z, t)
my = np.abs(y)
my = my.reshape(-1,)
y = y.reshape(-1,)
dy = dy.reshape(-1,)
t1 = 2*y*dx
t2 = 2*y*mdx*y
t3 = 2*y*dx*my
terms = 5*t1-0.2*t2-0.5*t3

z = z.reshape(1000,)

X = np.stack((z, x, 2*y*dx, 2*y*mdx*y, 2*y*dx*my), axis=-1)

model = ps.SINDy()
model.fit(X,t)
model.print()

c1 = 4.993
c2 = -0.250
c3 = -0.498

def model(w,t):
    dydt = c1*(4*np.cos(t)) + c2*(4*np.abs(np.cos(t))*w[0]) +c3*(4*np.cos(t)*np.abs(w[0]))
    dzdt = 2*w[0]*(c1*(4*np.cos(t)) + c2*(4*np.abs(np.cos(t))*w[0]) +c3*(4*np.cos(t)*np.abs(w[0])))
    dwdt = [dydt,dzdt]
    return dwdt

# initial condition
w0 = [0,0]

# solve ODE
w = odeint(model, w0, t)

plt.plot(x, z, 'r')
plt.plot(x, w[:, 1], linewidth=2, linestyle=':')
plt.xlabel('Voltage [kV]', fontsize = 12)
plt.ylabel('Displacement [$\mu$m]', fontsize = 12)
plt.legend(['Ground truth' , 'Learned relation'], prop={'size': 10})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp7_Butterfly/model.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((w[:, 1] - z)**2)/np.mean(z**2)
print("Relative Error Test: ", relative_error_test*100, "%")