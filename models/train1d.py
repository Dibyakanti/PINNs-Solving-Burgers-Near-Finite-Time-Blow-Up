import torch
import numpy as np
import json
import time
import os
import argparse
import copy
from burgers2d import *

def test_data_preparation(x_min,x_max,delta,total_points):
    ' Test data '
    test_x=torch.linspace(x_min,x_max,int(1.25*total_points)).view(-1,1)
    test_t=torch.linspace(-1+delta,delta,int(1.25*total_points)).view(-1,1)
    'Create the mesh '
    test_X,test_T=torch.meshgrid(test_x.squeeze(1),test_t.squeeze(1))

    x_test=torch.hstack((test_X.transpose(1,0).flatten()[:,None],test_T.transpose(1,0).flatten()[:,None]))
    
    return x_test

def data_preparation(x_min,x_max,delta,total_points,Nu,Nf):
    t_min = -1+delta
    t_max = delta
    x=torch.linspace(x_min,x_max,total_points).view(-1,1)
    t=torch.linspace(t_min,t_max,total_points).view(-1,1)
    'Create the mesh '
    X,T=torch.meshgrid(x.squeeze(1),t.squeeze(1))

    'Transform the mesh into a 2-column vector'
    x_train=torch.hstack((X.transpose(1,0).flatten()[:,None],T.transpose(1,0).flatten()[:,None]))

    test_x=torch.linspace(x_min,x_max,int(1.1*total_points)).view(-1,1)
    test_t=torch.linspace(t_min,t_max,int(1.1*total_points)).view(-1,1)
    'Create the mesh '
    test_X,test_T=torch.meshgrid(test_x.squeeze(1),test_t.squeeze(1))

    x_test=torch.hstack((test_X.transpose(1,0).flatten()[:,None],test_T.transpose(1,0).flatten()[:,None]))

    'Initial Condition (# = 1)'
    'Left Edge: x(x,0)=sin(x)-> xmin=<x=<xmax; t=0'
    x_lin = torch.linspace(x_min,x_max,total_points)
    left_X = torch.hstack((x_lin.view(-1,1),torch.zeros(x_lin.shape[0]).view(-1,1)))
    left_U=(-1*left_X[:,0]).unsqueeze(1)

    'Boundary Conditions (# = 2)'
    'Bottom Edge: x=_xmin; tmin=< t =<tmax'
    bottom_X=torch.hstack((X[0,:][:,None],T[0,:][:,None])) # First row # The [:,None] is to give it the right dimension
    bottom_U=x_min/(bottom_X[:,1]-1).unsqueeze(1)
    'Top Edge: x=x_max; 0=<t=<1'
    top_X=torch.hstack((X[-1,:][:,None],T[-1,:][:,None])) # Last row # The [:,None] is to give it the right dimension
    top_U=x_max/(top_X[:,1]-1).unsqueeze(1)
    'Note : for X al [:,i] are same and for T all [i,:]'

    'Get all the training data into the same dataset'
    X_train_temp=torch.hstack([left_X,bottom_X,top_X])
    U_train_temp=torch.hstack([left_U,bottom_U,top_U])

    'Choose(Nu) points of our available training data:'
    idx=np.random.choice(X_train_temp.shape[0],Nu,replace=False) # randomly choose Nu number of points
    X_train_Nu=X_train_temp[idx,:]
    U_train_Nu=U_train_temp[idx,:]


    'Collocation Points (Evaluate our PDE) '
    # X_train_Nf=lb+(ub-lb)*lhs(2,Nf) # 2 as the inputs are x and t; Choose(Nf) points (Latin hypercube)
    idx=np.random.choice(x_train.shape[0],Nf,replace=False)
    X_train_Nf = x_train[idx,:]
    
#     print("Boundary shapes for the edges:",left_X.shape,bottom_X.shape,top_X.shape)
#     print("Available training data:",X_train_temp.shape,U_train_temp.shape)
#     print("Final training data:",X_train_Nu.shape,U_train_Nu.shape)
#     print("Total collocation points:",X_train_Nf.shape)

    return X_train_Nu, U_train_Nu, X_train_Nf, x_test, X_train_temp, U_train_temp

def preprocess_config(parser):
	parser.add_argument('--lr', default= 1e-4, type=float)
	parser.add_argument('--wt_decay', default=0, type=float)
	parser.add_argument('--steps', default=100000,type=int)
	parser.add_argument('--boundary_points',default=300,type=int)
	parser.add_argument('--collocation_points', default=20000,type=int)
	parser.add_argument('--lambd_list',default=[1,1,1],  action='store', type=float, nargs='*')
	parser.add_argument('--delta_range',default=[0.950,0.999,0.004],  action='store', type=float, nargs='*')
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
    delta_enum = dict(enumerate(np.arange(args["delta_range"][0],args["delta_range"][1],args["delta_range"][2])))
    width_pow_enum = dict(enumerate(np.arange(args["width_range"][0],args["width_range"][1],args["width_range"][2])))
    lambd_list = torch.tensor([args["lambd_list"]])
    
    x_min = -1
    x_max = 1
    total_points = int(1.2 * Nf)

    for seed in args["seed"]:
        torch.manual_seed(seed)
        os.mkdir(f"./seed{seed}")
        for delta_idx in range(len(delta_enum)):
            for width_pow_idx in range(len(width_pow_enum)):
                delta = delta_enum[delta_idx]
                width_pow = width_pow_enum[width_pow_idx]
                width = 2**width_pow
                layers = np.array([2, width, width, 1])

                X_train_Nu, U_train_Nu, X_train_Nf, x_test, X_train_temp, U_train_temp = data_preparation(x_min,x_max,delta,total_points,Nu,Nf)
                'Store tensors to GPU'
                X_train_Nu=X_train_Nu.float().to(device)#Training Points (BC)
                U_train_Nu=U_train_Nu.float().to(device)#Training Points (BC)
                X_train_Nf=X_train_Nf.float().to(device)#Collocation Points

                # X_test=test_data_preparation(delta).float().to(device) # the input dataset (complete)
                # X_BC_test = X_train_temp.float().to(device)
                # U_BC_test = U_train_temp.float().to(device)
                for lambd in lambd_list:
                    lambd = lambd.to(device)
                    'Create model'
                    PINN = FCN(layers,lambd,device)
                    PINN.to(device)
                    print(PINN)
                    params = list(PINN.parameters())
                    optimizer = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False, weight_decay=wt_decay)
                    print(f"LAMBDA = {'_'.join([str(x) for x in lambd])} DELTA = {delta} WIDTH = {width}")
                    train_loss = []
                    # test_loss = []
                    start_time = time.time()

                    for i in range(steps):
                        if i==0:
                            print("Step --- Time --- Training Loss --- Test Loss")
                        loss = PINN.loss(X_train_Nu,U_train_Nu,X_train_Nf)# use mean squared error
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.detach().cpu().numpy())
            #             if (i%10000)==0:
            #                 test_loss.append(PINN.test_loss(X_BC_test,U_BC_test,X_test).detach().cpu().numpy())
                        if (i%10000)==0:
                            print(f"{i} --- {(time.time()-start_time)/60} --- {train_loss[-1]}")
                            start_time = time.time()
                        # if((len(test_loss)>1 and test_loss[-1]<test_loss[-2]) or len(test_loss)==1):
                        #   model_least_test_loss = copy.deepcopy(PINN.state_dict())
                        #   epoch_least_test_loss = i
                        if((len(train_loss)>1 and train_loss[-1]<train_loss[-2]) or len(train_loss)==1):
                            model_least_train_loss = copy.deepcopy(PINN.state_dict())
                            epoch_least_train_loss = i
                    

                    ' Save models and metadata '

                    if(not os.path.exists(f"./seed{seed}")):
                      os.mkdir(f"./seed{seed}")

                    torch.save(model_least_train_loss, f"./seed{seed}/Burgers_lr={lr}_width={width}_tmin,tmax_{round(-1+delta,3)},{round(delta,3)}_Nu={Nu}_Nf={Nf}_steps={steps}_trainl={min(train_loss)}")

                    'JSON files'
                    store_loss = {"train_loss":[x.tolist() for x in train_loss]}
                    with open(f"./seed{seed}/train_loss_width={width}_delta={delta}.json", 'w') as fp:
                        json.dump(store_loss, fp)