import torch
import numpy as np
import json
import time
import os
import argparse
import copy
from burgers1d import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_sample_one(data_size,t_min):

    sample_temp = torch.rand(data_size,3)
    sample_temp[:, 2] = t_min + (sample_temp[:, 2] * 0.707)

    return sample_temp.to(device)

def generate_sample_two(data_size, t_min):

    data_size = 4*data_size
    sample_temp = torch.rand(data_size,3)
    sample_temp[:, 2] = t_min + (sample_temp[:, 2] * 0.707)
    # make the first element of sample_temp to be 0 in the first quarter
    sample_temp[0: len(sample_temp)//4, 0] = 0
    # make the first element of sample_temp to be 1 in the second quarter
    sample_temp[len(sample_temp)//4: len(sample_temp)//2, 0] = 1
    # make the second element of sample_temp to be 0 in the third quarter
    sample_temp[len(sample_temp)//2: 3*len(sample_temp)//4, 1] = 0
    # make the second element of sample_temp to be 1 in the fourth quarter
    sample_temp[3*len(sample_temp)//4: len(sample_temp), 1] = 1

    return sample_temp.to(device)

def generate_sample_three(data_size, t_min):

    sample_temp = torch.rand(data_size, 3)
    sample_temp[:, 2] = t_min

    return sample_temp.to(device)

def data_preparation(Nu, Nf, t_min):
    x1 = generate_sample_one(Nf, t_min)
    x2 = generate_sample_two(Nu, t_min)
    x3 = generate_sample_three(Nu, t_min)
    return x1, x2, x3

def preprocess_config(parser):
	parser.add_argument('--lr', default= 1e-4, type=float)
	parser.add_argument('--wt_decay', default=0, type=float)
	parser.add_argument('--steps', default=100000,type=int)
	parser.add_argument('--boundary_points',default=300,type=int)
	parser.add_argument('--collocation_points', default=20000,type=int)
	parser.add_argument('--time_range',default=[-0.6,0,0.02],  action='store', type=float, nargs='*')
	parser.add_argument('--width_range',default=[4,8,1],  action='store', type=int, nargs='*')
	parser.add_argument('--seed',default=[1234],  action='store', type=int, nargs='*')
	return parser


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = preprocess_config(parser)
    args = vars(parser.parse_args())


    'Training Parameters'
    lr = args["lr"]
    wt_decay = args["wt_decay"]
    steps = args["steps"]
    Nu = args["boundary_points"]
    Nf = args["collocation_points"]
    t_min_enum = dict(enumerate(np.arange(args["time_range"][0],args["time_range"][1],args["time_range"][2])))
    width_pow_enum = dict(enumerate(np.arange(args["width_range"][0],args["width_range"][1],args["width_range"][2])))

    for seed in args["seed"]:
        torch.manual_seed(seed)
        os.mkdir(f"./seed{seed}")
        for t_min_index in range(len(t_min_enum)):
            for width_pow_idx in range(len(width_pow_enum)):
                t_min = t_min_enum[t_min_index]
                width_pow = width_pow_enum[width_pow_idx]
                width = 2**width_pow

                # layers = np.array([3, width, width, width, width, width, width, 2])
                layers = np.array([3, width, width, 2])

                X1, X2, X3 = data_preparation(Nu, Nf, t_min)
                delta = t_min + (1/np.sqrt(2))
                t_max = delta

                'Store tensors to GPU'
                X1=X1.float().to(device) # Training Points (BC)
                X2=X2.float().to(device) # Training Points (BC)
                X3=X3.float().to(device) # Collocation Points
                
                'Create model'
                PINN = FCN(layers,delta)
                PINN.to(device)
                print(PINN)
                params = list(PINN.parameters())
                optimizer = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False, weight_decay=wt_decay)
                print(f"{width_pow_idx}. t_min = {t_min} ------------------ seed = {seed}")
                train_loss = []
                start_time = time.time()

                for i in range(steps):
                    if i==0:
                        print("Step --- Time --- Training Loss")
                    loss = PINN.loss(X1,X2,X3)# use mean squared error
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.detach().cpu().numpy())
                    if i%(10000)==0:
                        print(f"{i} --- {(time.time()-start_time)/60} --- {train_loss[-1]}")
                        start_time = time.time()
                    if((len(train_loss)>1 and train_loss[-1]<train_loss[-2]) or len(train_loss)==1):
                        model_least_train_loss = copy.deepcopy(PINN.state_dict())
                        epoch_least_train_loss = i

                torch.save(model_least_train_loss, f"./seed{seed}/Burger_seed={seed}_lr={lr}_width={width}_tmin,tmax_{t_min},{t_max}_Nu={Nu}_Nf={Nf}_steps={steps}_trainl={min(train_loss)}")

                'JSON files'
                store_loss = {"train_loss":[x.tolist() for x in train_loss]}
                with open(f"./seed{seed}/train_loss_width={width}_delta={delta}.json", 'w') as fp:
                    json.dump(store_loss, fp)