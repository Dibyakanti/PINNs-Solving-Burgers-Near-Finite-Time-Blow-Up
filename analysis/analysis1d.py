import torch
import torch.autograd as autograd
import numpy as np
import json
import time
import os
import re
import argparse
from scipy.integrate import dblquad,quad
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../models")

from models.burgers1d import *
from models.train1d import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Calculating RHS and LHS
'''

' Auxiliary functions '

def func_gen(PINN, x, t):
    x = torch.tensor(x)
    t = torch.tensor(t)
    X_test = torch.hstack((x.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    u_hat = (PINN(X_test) - x/(t-1))**2
    ret = u_hat.detach().numpy().squeeze()

    return ret

def func_int(PINN, x, t):
    x = torch.tensor(x)
    t = torch.tensor(t)
    X_test = torch.hstack((x.flatten()[:,None],t.flatten()[:,None])).float().to(device)
    # Pass through the NN
    g=X_test.clone()
    g.requires_grad=True # Enable differentiation (for back prop)
    u=PINN(g) # NN output
    
    # Pass through the PDE
    u_x_t = autograd.grad(u,g,torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0] # first derivative

    u_x=u_x_t[:,[0]]# we select the 2nd element for t (the first one is x) (Remember the input X=[x,t])
    u_t=u_x_t[:,[1]]# we select the 2nd element for t (the first one is x) (Remember the input X=[x,t])

    ret=u_t + (u * u_x)

    return ret

def func_tb(PINN, x, T):
  x = torch.tensor(x)
  X_test = torch.hstack((x.flatten()[:,None],T*torch.zeros_like(x).flatten()[:,None])).float().to(device)
  ret = PINN(X_test).detach().numpy().squeeze()

  return ret

def func_sb1(PINN, t):
  t = torch.tensor(t)
  X_test = torch.hstack((torch.ones_like(t).flatten()[:,None],t.flatten()[:,None])).float().to(device)
  ret = PINN(X_test).detach().numpy().squeeze()

  return ret

def func_sb2(PINN, t):
  t = torch.tensor(t)
  X_test = torch.hstack((-1*torch.ones_like(t).flatten()[:,None],t.flatten()[:,None])).float().to(device)
  ret = PINN(X_test).detach().numpy().squeeze()

  return ret


def LHS(PINN, delta):
    ret = dblquad(lambda x, t: func_gen(PINN, x, t), -1+delta, delta, -1, 1)[0]
    return ret

def RHS(PINN, delta):
    C = (1 + 2*(1/(1-delta)))
    C_1b = (1/(1-delta))**2

    t = torch.arange(-1+delta,delta,0.0001)
    X_test = torch.hstack((torch.ones_like(t).flatten()[:,None],t.flatten()[:,None])).float().to(device)
    C_2b = 3/(2*(1-delta)) + max(PINN(X_test).detach().numpy().squeeze())

    ret = (1+C*np.exp(C))*(quad(lambda x: func_tb(PINN,x,-1+delta), -1, 1)[0] \
                        + 2*C_2b*( quad(lambda t: (func_sb1(PINN, t))**2, -1+delta, delta)[0] +  quad(lambda t: (func_sb2(PINN, t))**2, -1+delta, delta)[0]) \
                        + 2*C_1b*( np.sqrt(quad(lambda t: (func_sb1(PINN, t))**2, -1+delta, delta)[0]) +  np.sqrt(quad(lambda t: (func_sb2(PINN, t))**2, -1+delta, delta)[0])) \
                        + dblquad(lambda x, t: (func_int(PINN, x, t))**2, -1+delta, delta, -1, 1)[0])
    return ret


' Calculating for several different PINNs '

def calculate_RHSLHS(width, file_dir):
    layers = np.array([2, width, width, width, width, width, width, 1])

    PINN = FCN(layers)

    LHS_list = []
    RHS_list = []
    fractional_error = []

    delta_list = []
    for fname in os.listdir(file_dir):
        if(re.match(r".*tmintmax_(?P<tmin>[-]*0.[^.]+)(?P<delta>.[^_]+)",fname)):
            m = re.match(r".*tmintmax_(?P<tmin>[-]*0.[^.]+)(?P<delta>.[^_]+)",fname)["delta"]
            delta_list.append(float(m))
    delta_list.sort()
    
    for delta in delta_list:
        start_time = time.time()
        for f in os.listdir(path=file_dir):
            if(f"tmintmax_{round(-1+delta,3)}{round(delta,3)}" in f and f.startswith("Burger")):
                PINN.load_state_dict(torch.load(f"{file_dir}/{f}",map_location=device))
                LHS_list.append(LHS(PINN, delta))
                RHS_list.append(RHS(PINN, delta))
                integral_u = 2/(3*(delta-1)*(delta-2))
                fractional_error.append(LHS_list[-1]/integral_u)
                end_time = time.time()
                print(f"{(end_time-start_time)/60} minute ; delta:{delta}")
                
    data = {"RHS":RHS_list, "LHS":LHS_list, "delta":delta_list}
    with open(f'dict_RHS_LHS_delta_width{width}.json', 'w') as fp:
        json.dump(data, fp)

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
   