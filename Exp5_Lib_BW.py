import numpy as np
import matplotlib.pyplot as plt
import pysindy as ps
from scipy.integrate import odeint
from math import pi
from pysindy.differentiation import FiniteDifference
fd = FiniteDifference(order=2, d=1)

t = np.linspace(0,20*pi,1000)
x = t*np.sin(t)
dx = fd._differentiate(x, t)
mdx = np.abs(dx)

def model(y, t):
    dydt = 1*np.abs(t*np.cos(t) + np.sin(t))*t*np.sin(t) - 0.25 * np.abs(t*np.cos(t) + np.sin(t))*y + 20*(t*np.cos(t) + np.sin(t))
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
plt.savefig("Results/Exp5_Lib_BW/inp_out.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

dy = fd._differentiate(y, t)
my = np.abs(y)
my = my.reshape(-1,)
y = y.reshape(-1,)
dy = dy.reshape(-1,)
t1 = dx
t2 = mdx*y
t3 = dx*my

X = np.stack((y, x, dx, mdx, my), axis=-1)

model = ps.SINDy()
model.fit(X,t)
model.print()

c1 = 19.846
c2 = -0.247
c3 = 0.990
def test_model(y, t):
    dydt = c3*np.abs(t*np.cos(t) + np.sin(t))*t*np.sin(t) + c2* np.abs(t*np.cos(t) + np.sin(t))*y + c1*(t*np.cos(t) + np.sin(t))
    return dydt

ytest_0 = 0
y_test1 = odeint(test_model, ytest_0, t)

c1 = 20
c2 = -0.25
c3 = 1
y_truth = odeint(test_model, ytest_0, t)

plt.plot(x, y, 'r')
plt.plot(x, y_test1, linewidth=2, linestyle=':')
plt.xlabel('Voltage [V]', fontsize = 12)
plt.ylabel('Displacement [$\mu$m]', fontsize = 12)
plt.legend(['Ground truth' , 'Learned relation'], prop={'size': 10})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig("Results/Exp5_Lib_BW/model.pdf", dpi = 3000, bbox_inches='tight')
plt.show()

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((y_test1 - y_truth)**2)/np.mean(y_truth**2)
print("Relative Error Test SINDy: ", relative_error_test*100, "%")

t1=t1.reshape(-1,1)
t2=t2.reshape(-1,1)
t3=t3.reshape(-1,1)
Y = np.concatenate((t1, t2, t3), axis=1)
dy = dy.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Y, dy, test_size=0.25, random_state=42)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
print("Coefficients from Ridge regression: ",clf.coef_)
print("Intercept from Ridge regression: ",clf.intercept_)

c1 = 3.7543
c2 = -0.024
c3 = 0.025
def test_model(y, t):
    dydt = c1*(t*np.cos(t) + np.sin(t)) + c2*np.abs(t*np.cos(t) + np.sin(t))*y + c3*(t*np.cos(t) + np.sin(t))*np.abs(y)
    return dydt

ytest_0 = 0
y_test2 = odeint(test_model, ytest_0, t)

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((y_test2 - y)**2)/np.mean(y**2)
print("Relative Error Test Ridge: ", relative_error_test*100, "%")

from sklearn.linear_model import LinearRegression
linreg = LinearRegression().fit(X_train, y_train)

print("Coefficients from Linear regression: ",linreg.coef_)

c1 = 3.754
c2 = -0.024
c3 = 0.025
def test_model(y, t):
    dydt = c1*(t*np.cos(t) + np.sin(t)) + c2*np.abs(t*np.cos(t) + np.sin(t))*y + c3*(t*np.cos(t) + np.sin(t))*np.abs(y)
    return dydt

ytest_0 = 0
y_test3 = odeint(test_model, ytest_0, t)

y = y.reshape(-1,1)
# Compute the relative L2 error norm (generalization error)
relative_error_test = np.mean((y_test3 - y)**2)/np.mean(y**2)
print("Relative Error Test Lin reg: ", relative_error_test*100, "%")

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(Y, dy, test_size=0.25, random_state=42)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils
# import torch.utils.data
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Type of optimizer (ADAM or LBFGS)
# opt_type = "LBFGS"
# # Random Seed for dataset generation
# sampling_seed = 78
# torch.manual_seed(sampling_seed)
#
# # Number of training samples
# n_samples = 1000
#
# x = X_train
# y = y_train
# y = y.reshape(-1,1)
#
# x = x.astype(np.float32)
# y = y.astype(np.float32)
#
# x = torch.from_numpy(x)
# y = torch.from_numpy(y)
#
# batch_size = n_samples
# training_set = DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=batch_size, shuffle=True)
#
#
# class NeuralNet(nn.Module):
#
#     def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons):
#         super(NeuralNet, self).__init__()
#         # Number of input dimensions n
#         self.input_dimension = input_dimension
#         # Number of output dimensions m
#         self.output_dimension = output_dimension
#         # Number of neurons per layer
#         self.neurons = neurons
#         # Number of hidden layers
#         self.n_hidden_layers = n_hidden_layers
#         # Activation function
#         self.activation = nn.Tanh()
#
#         self.input_layer = nn.Linear(self.input_dimension, self.neurons)
#         self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers)])
#         self.output_layer = nn.Linear(self.neurons, self.output_dimension)
#
#     def forward(self, x):
#         # The forward function performs the set of affine and non-linear transformations defining the network
#         # (see equation above)
#         x = self.activation(self.input_layer(x))
#         for k, l in enumerate(self.hidden_layers):
#             x = self.activation(l(x))
#         return self.output_layer(x)
#
#
# def NeuralNet_Seq(input_dimension, output_dimension, n_hidden_layers, neurons):
#     modules = list()
#     modules.append(nn.Linear(input_dimension, neurons))
#     modules.append(nn.Tanh())
#     for _ in range(n_hidden_layers):
#         modules.append(nn.Linear(neurons, neurons))
#         modules.append(nn.Tanh())
#     modules.append(nn.Linear(neurons, output_dimension))
#     model = nn.Sequential(*modules)
#     return model
#
# # Model definition
# my_network = NeuralNet(input_dimension=x.shape[1], output_dimension=y.shape[1], n_hidden_layers=4, neurons=20)
# # my_network = NeuralNet_Seq(input_dimension=x.shape[1], output_dimension=y.shape[1], n_hidden_layers=4, neurons=20)
#
# def init_xavier(model, retrain_seed):
#     torch.manual_seed(retrain_seed)
#     def init_weights(m):
#         if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
#             g = nn.init.calculate_gain('tanh')
#             torch.nn.init.xavier_uniform_(m.weight, gain=g)
#             #torch.nn.init.xavier_normal_(m.weight, gain=g)
#             m.bias.data.fill_(0)
#     model.apply(init_weights)
#
# # Random Seed for weight initialization
# retrain = 128
# # Xavier weight initialization
# init_xavier(my_network, retrain)
# # Model definition
#
# # Predict network value of x
# print(my_network(x))
#
# if opt_type == "ADAM":
#     optimizer_ = optim.Adam(my_network.parameters(), lr=0.001)
# elif opt_type == "LBFGS":
#     optimizer_ = optim.LBFGS(my_network.parameters(), lr=0.1, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)
# else:
#     raise ValueError("Optimizer not recognized")
#
#
# def fit(model, training_set, num_epochs, optimizer, p, verbose=True):
#     history = list()
#
#     # Loop over epochs
#     for epoch in range(num_epochs):
#         if verbose: print("################################ ", epoch, " ################################")
#
#         running_loss = list([0])
#
#         # Loop over batches
#         for j, (x_train_, u_train_) in enumerate(training_set):
#             def closure():
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 # forward + backward + optimize
#                 u_pred_ = model(x_train_)
#                 # Item 1. below
#                 loss = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
#                 # Item 2. below
#                 loss.backward()
#                 # Compute average training loss over batches for the current epoch
#                 running_loss[0] += loss.item()
#                 return loss
#
#             # Item 3. below
#             optimizer.step(closure=closure)
#
#         print('Loss: ', (running_loss[0] / len(training_set)))
#         history.append(running_loss[0])
#
#     return history
#
# n_epochs = 1000
# history = fit(my_network, training_set, n_epochs, optimizer_, p=2, verbose=True )
#
# plt.grid(True, which="both", ls=":")
# plt.plot(np.arange(1,n_epochs+1), np.log10(history), label="Train Loss")
# plt.legend()
#
# X_test = X_test.astype(np.float32)
# X_test = torch.from_numpy(X_test)
#
# y_test = y_test.astype(np.float32)
# y_test = torch.from_numpy(y_test)
#
# y_test_pred = my_network(X_test).reshape(-1,1)
#
# y_test = y_test.reshape(-1,1)
#
# # Compute the relative L2 error norm (generalization error)
# relative_error_test = torch.mean((y_test_pred - y_test)**2)/torch.mean(y_test**2)
# print("Relative Error Test DNN: ", relative_error_test.detach().numpy()*100, "%")
