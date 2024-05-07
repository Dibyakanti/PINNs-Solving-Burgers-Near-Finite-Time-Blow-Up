# PINNs_Burgers

This repository contains the official code for the paper : [Investigating the Ability of PINNs To Solve Burgers' PDE Near Finite-Time BlowUp](https://arxiv.org/abs/2310.05169)

## 1. Training the PINNs

### 1.1 (1+1)-Burgers'
To train the PINN and save the models for the 1+1 Burgers PDE
```
python3 ./models/train1d.py \
--lr 1e-4 \
--wt_decay 0 \
--steps 100000 \
--boundary_points 300 \
--collocation_points 20000 \
--lambd_list 1 1 1 \
--delta_range 0.950 0.999 0.004 \
--width_range 4 8 1  \
--seed 1234

Argument details :
lr: learnign rate
wt_decay: weight decay for the optimizer (regularization) 
steps: number of training epochs 
boundary_points: number of points for intial and boundary condition each 
collocation_points: number of collocation points 
lambd_list: weight for the different terms of the boundary and inital loss 
delta_range: range of delta min, max and the step ( np.arange(min,max,step) ) 
width_range: range of width min, max and the step ( np.arange(min,max,step) )  
seed: seed for the random intialization (torch)
```

### 1.2 (2+1)-Burgers'
To train the PINN and save the models for the 2+1 Burgers PDE
```
python3 ./models/train2d.py \
--lr 1e-4 \
--wt_decay 0 \
--steps 100000 \
--boundary_points 300 \
--collocation_points 20000 \
--time_range 0.950 0.999 0.004 \
--width_range 4 8 1  \
--seed 1234

Argument details :
lr: learnign rate
wt_decay: weight decay for the optimizer (regularization) 
steps: number of training epochs 
boundary_points: number of points for intial and boundary condition each 
collocation_points: number of collocation points 
time_range: range of delta min, max and the step ( np.arange(min,max,step) ) 
width_range: range of width min, max and the step ( np.arange(min,max,step) )  
seed: seed for the random intialization (torch)
```

## 2. Analysis

### 2.1 (1+1)-Burgers'

To calculate the LHS and RHS of `Equation 11` of `Theorem 4.2`

```
python3 ./analysis/analysis1d.py \
--width 30 \
--file_dir "../models/seed1234/"

Argument details :
width: width of the PINN
file_dir: directory from where to load the saved models
```

### 2.2 (2+1)-Burgers'

To calculate the LHS and RHS of `Equation 5` of `Theorem 4.1`

```
python3 ./analysis/analysis2d.py \
--width 30 \
--file_dir "../models/seed1234/"

Argument details :
width: width of the PINN
file_dir: directory from where to load the saved models
```