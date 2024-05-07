import torch
import torch.autograd as autograd
import numpy as np
import json
import time
import os
import re
import argparse
from scipy.integrate import dblquad, tplquad, quad
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../models")

from models.burgers2d import *
from models.train2d import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Calculating RHS and LHS
'''

' Auxiliary functions '
def f_exact(X):
    x1 = X[:,0]
    x2 = X[:,1]
    t = X[:,2]
    # u(x, y, t)=\frac{x+y-2\cdot x \cdot t}{1-2\cdot t^2},
    temp1 = (x1+x2-(2*x1*t))/(1-(2*(t**2)))
    # v(x, y, t)=\frac{x-y-2\cdot y \cdot t}{1-2\cdot t^2},
    temp2 = (x1-x2-(2*x2*t))/(1-(2*(t**2)))
    u_temp = torch.zeros(x1.size()[0],2)
    u_temp[:,0] = temp1
    u_temp[:,1] = temp2
    return u_temp.to(device)

def func_gen(PINN, x1, x2, t):
    x1 = torch.tensor(x1).to(device)
    x2 = torch.tensor(x2).to(device)
    t = torch.tensor(t).to(device)
    X_test = torch.hstack((x1.flatten()[:,None], x2.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    PINN.to(device)
    u_hat = torch.mean((PINN(X_test) - f_exact(X_test))**2)
    ret = u_hat.detach().numpy().squeeze()
    return ret

def integral_pde(PINN, x1, x2, t):
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    t = torch.tensor(t)
    X_test = torch.hstack((x1.flatten()[:,None],x2.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    # Pass through the NN
    g=X_test.clone()
    g.requires_grad=True # Enable differentiation
    PINN.to(device)
    u=PINN(g) # NN output

    # Pass through the PDE
    u_hat1_x = u[:, 0].reshape([g.size()[0],1])
    u_hat1_y = u[:, 1].reshape([g.size()[0],1])
    grad_u_hat1_x = torch.autograd.grad(outputs = u_hat1_x, inputs = g, grad_outputs = torch.ones(u_hat1_x.shape).to(device), create_graph = True)  # dx dy and dt
    grad_u_hat1_y = torch.autograd.grad(outputs = u_hat1_y, inputs = g, grad_outputs = torch.ones(u_hat1_y.shape).to(device), create_graph = True)  # dx dy and dt
    dx1 = grad_u_hat1_x[0][:, 0].reshape([g.size()[0],1])  # dx1
    dy1 = grad_u_hat1_x[0][:, 1].reshape([g.size()[0],1])  # dx2
    dt1 = grad_u_hat1_x[0][:, 2].reshape([g.size()[0],1])  # dt
    dx2 = grad_u_hat1_y[0][:, 0].reshape([g.size()[0],1])  # dx1
    dy2 = grad_u_hat1_y[0][:, 1].reshape([g.size()[0],1])  # dx2
    dt2 = grad_u_hat1_y[0][:, 2].reshape([g.size()[0],1])  # dt
    L1 = torch.add(dt1,torch.mul(dx1, u_hat1_x))
    L1 = torch.add(L1,torch.mul(dy1, u_hat1_y))
    L1 = L1.pow(2).sum()
    L1 = torch.div(L1, len(g))
    L2 = torch.add(dt2, torch.mul(dx2, u_hat1_x))
    L2 = torch.add(L2, torch.mul(dy2, u_hat1_y))
    L2 = L2.pow(2).sum()
    L2 = torch.div(L2, len(g))
    ret = (L1 + L2).detach().numpy().squeeze()

    return ret

def integral_t(PINN, x1, x2, t, delta):
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    t = torch.tensor(t)
    X_test = torch.hstack((x1.flatten()[:,None],x2.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    x3 = X_test.clone()
    x3.requires_grad = True

    u_hat3 = PINN(x3)
    u_hat3_x = u_hat3[:, 0].reshape([x3.size()[0],1])
    u_hat3_y = u_hat3[:, 1].reshape([x3.size()[0],1])
    u_0_x = u1_0(x3, delta).reshape([x3.size()[0],1])
    u_0_y = u2_0(x3, delta).reshape([x3.size()[0],1])
    L5 = torch.add(u_hat3_x, torch.neg(u_0_x)).pow(2).sum()
    L6 = torch.add(u_hat3_y, torch.neg(u_0_y)).pow(2).sum()
    L5 = torch.div(L5, len(x3))
    L6 = torch.div(L6, len(x3))
    R_t = L5 + L6
    ret = R_t.detach().numpy().squeeze()

    return ret

def jacobian_u_max(PINN, t_min, t_max):
    t = t_min
    J1 = np.abs((1 - 2*t)/(1-2*(t**2)))
    J23 = np.abs(1/(1-2*(t**2)))
    J4 = np.abs((-1 - 2*t)/(1-2*(t**2)))
    
    t = t_max
    J1_ = np.abs((1 - 2*t)/(1-2*(t**2)))
    J23_ = np.abs(1/(1-2*(t**2)))
    J4_ = np.abs((-1 - 2*t)/(1-2*(t**2)))
    
    ret = max(J1+J23, J23+J4, J1_+J23_, J23_+J4_)
    
    return ret
    
def integral_u(PINN, t_min, t_max):
    t = t_min
    T1 = np.divide((16*t - 6),12*(1-2*(t**2)))
    t = t_max
    T2 = np.divide((16*t - 6),12*(1-2*(t**2)))
    ret = T2 - T1
    
    return ret


def jacobian_u_theta_max(PINN, x1, x2, t):
    
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    t = torch.tensor(t)
    X_test = torch.hstack((x1.flatten()[:,None],x2.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    # Pass through the NN
    g=X_test.clone()
    g.requires_grad=True # Enable differentiation
    PINN.to(device)
    u=PINN(g) # NN output

    # Pass through the PDE
    u_hat1_x = u[:, 0].reshape([g.size()[0],1])
    u_hat1_y = u[:, 1].reshape([g.size()[0],1])
    grad_u_hat1_x = torch.autograd.grad(outputs = u_hat1_x, inputs = g, grad_outputs = torch.ones(u_hat1_x.shape).to(device), create_graph = True)  # dx dy and dt
    grad_u_hat1_y = torch.autograd.grad(outputs = u_hat1_y, inputs = g, grad_outputs = torch.ones(u_hat1_y.shape).to(device), create_graph = True)  # dx dy and dt
    J1 = grad_u_hat1_x[0][:, 0].reshape([g.size()[0],1]).abs() # dx
    J2 = grad_u_hat1_x[0][:, 1].reshape([g.size()[0],1]).abs() # dy
    J3 = grad_u_hat1_y[0][:, 0].reshape([g.size()[0],1]).abs() # dx
    J4 = grad_u_hat1_y[0][:, 1].reshape([g.size()[0],1]).abs() # dy
    
    max1 = (J1 + J3).max()
    max2 = (J2 + J4).max()
    
    ret = max(max1, max2).detach().numpy().squeeze()
    
    return ret

def integral_u_theta(PINN, x1, x2, t):

    x1 = torch.tensor(x1).to(device)
    x2 = torch.tensor(x2).to(device)
    t = torch.tensor(t).to(device)
    X_test = torch.hstack((x1.flatten()[:,None], x2.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    PINN.to(device)
    u_theta = torch.mean(PINN(X_test)**2)
    ret = u_theta.detach().numpy().squeeze()
    
    return ret


def LHS(PINN, t_min):
#     ret = tplquad(lambda x1, x2, t: func_gen(x1, x2, t), t_min, t_max, 0, 1 , 0, 1, epsrel=1.49e-04, epsabs=1.49e-04)[0]
    data = generate_sample_one(500000, t_min)
    x1 = data[:,0]
    x2 = data[:,1]
    t = data[:,2]
    ret1 = func_gen(PINN, x1, x2, t)
    ret = np.log(ret1)
    
    return ret,ret1

def RHS(PINN, t_min, delta):
    data = generate_sample_three(500000, t_min)
    x1 = data[:,0]
    x2 = data[:,1]
    t = data[:,2]
    C2 = 0
    C2 += integral_t(PINN, x1, x2, t, delta)
    data = generate_sample_one(500000, t_min)
    x1 = data[:,0]
    x2 = data[:,1]
    t = data[:,2]
    C2 += integral_pde(PINN, x1, x2, t)
    
    t_max = t_min+(1/np.sqrt(2))
    C2 += (4 * jacobian_u_max(PINN, t_min, t_max) * integral_u(PINN, t_min, t_max))
    
    C2 += (4 * jacobian_u_theta_max(PINN, x1, x2, t) * integral_u_theta(PINN, x1, x2, t))
    
    
    C1 = 1 + (4 * jacobian_u_max(PINN, t_min, t_max)) + (4 * jacobian_u_theta_max(PINN, x1, x2, t))
    
#     ret = np.log(C2) + np.log( (-1/np.sqrt(2)) + ((C1/4)* np.exp(C1/np.sqrt(2))) )
    ret = np.log(C2) + np.log(C1/4) + (C1/np.sqrt(2))
    
    return ret


' Calculating for several different PINNs '

def calculate_RHSLHS(width, file_dir):
    
    layers = np.array([3, width, width, width, width, width, width, 2])
    
    LHS_list = []
    RHS_list = []
    delta_list = []
    fractional_error = []
    u_norm = []
    
    t_min_list = []
    for fname in os.listdir(file_dir):
        if(re.match(r".*tmintmax_(?P<tmin>[-]*0.[^.]+)",fname)):
            m = re.match(r".*tmintmax_(?P<tmin>[-]*0.[^.]+)",fname)["tmin"][:-1]
            t_min_list.append(float(m))
    t_min_list.sort()
    print(f"Calculating for width = {width} and file directory = {file_dir}")

    for t_min in t_min_list:
        start_time = time.time()
        for f in os.listdir(path=file_dir):
            if(f"tmintmax_{t_min}{t_min+0.707}" in f and f.startswith("Burger")):
                t_max = t_min + (1/np.sqrt(2))
                delta = t_max
                PINN = FCN(layers, delta)
                PINN.load_state_dict(torch.load(f"{file_dir}/{f}",map_location=device))
                LHS_out = LHS(PINN, t_min)
                LHS_list.append(LHS_out[0])
                RHS_list.append(RHS(PINN, t_min, delta))
                delta_list.append(delta)
                u_norm.append(integral_u(PINN, t_min, t_max))
                fractional_error.append(LHS_out[1]/integral_u(PINN, t_min, t_max))
                end_time = time.time()
                print(f"{(end_time-start_time)/60} minutes ------- delta : {delta}")
                
    output = {"RHS":RHS_list, "LHS":LHS_list, "delta":delta_list}
    with open(f'dict_RHS_LHS_delta_width{width}.json', 'w') as fp:
        json.dump(output, fp)
                
    return RHS_list, LHS_list, delta_list, fractional_error, u_norm

def preprocess_config(parser):
	parser.add_argument('--width', default= 30, type=float)
	parser.add_argument('--file_dir', default= "../models/seed1234/", type=str)
	return parser


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = preprocess_config(parser)
    args = vars(parser.parse_args())

    width = args["width"]
    file_dir = args["file_dir"]

    calculate_RHSLHS(width, file_dir)


