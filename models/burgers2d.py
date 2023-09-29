import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Define intial and boundary conditions functions
'''
def u1_0(X, delta):
    u_temp = torch.zeros(X.shape[0],1)
    x1 = X[:,0]
    x2 = X[:,1]
    u_temp[:,0] = ((1 + np.sqrt(2) - 2*delta)*x1 + x2)/(2*delta*(np.sqrt(2) - delta))

    return u_temp.reshape([X.size()[0],1]).to(device)

def u2_0(X, delta):
    u_temp = torch.zeros(X.shape[0],1)
    x1 = X[:,0]
    x2 = X[:,1]
    u_temp[:,0] = (x1 - (1 - np.sqrt(2) + 2*delta)*x2)/(2*delta*(np.sqrt(2) - delta))

    return u_temp.reshape([X.size()[0],1]).to(device)

def g_0x1(X):
    x1 = torch.zeros([X.shape[0],1])[:,0].to(device)
    x1.requires_grad = True
    x2 = X[:,1].to(device)
    t = X[:,2].to(device)
    # use cuda torch to write temp1 = (x+y-2*x*t)/(1-2*t**2)
    temp1 = torch.divide(torch.add(torch.add(x1,x2),torch.neg(torch.mul(torch.mul(2,x1),t))),torch.add(torch.neg(torch.mul(torch.mul(2,t),t)),1))
    u_temp = temp1
    return u_temp.reshape([x1.size()[0], 1])

def g_1x1(X):
    x1 = torch.ones([X.shape[0],1])[:,0].to(device)
    x1.requires_grad = True
    x2 = X[:,1].to(device)
    t = X[:,2].to(device)
    # use cuda torch to write temp1 = (x+y-2*x*t)/(1-2*t**2)
    temp1 = torch.divide(torch.add(torch.add(x1,x2),torch.neg(torch.mul(torch.mul(2,x1),t))),torch.add(torch.neg(torch.mul(torch.mul(2,t),t)),1))
    u_temp = temp1
    return u_temp.reshape([x1.size()[0], 1])


def g_0x2(X):
    x1 = X[:,0].to(device)
    x2 = torch.zeros([X.shape[0],1])[:,0].to(device)
    x2.requires_grad = True
    t = X[:,2].to(device)
    # use cuda torch to write temp1 = (x-y-2*y*t)/(1-2*t**2)
    temp1 = torch.divide(torch.add(torch.sub(x1,x2),torch.neg(torch.mul(torch.mul(2,x2),t))),torch.add(torch.neg(torch.mul(torch.mul(2,t),t)),1))
    u_temp = temp1
    return u_temp.reshape([x1.size()[0], 1])

def g_1x2(X):
    x1 = X[:,0].to(device)
    x2 = torch.ones([X.shape[0],1])[:,0].to(device)
    x2.requires_grad = True
    t = X[:,2].to(device)
    # use cuda torch to write temp1 = (x-y-2*y*t)/(1-2*t**2)
    temp1 = torch.divide(torch.add(torch.sub(x1,x2),torch.neg(torch.mul(torch.mul(2,x2),t))),torch.add(torch.neg(torch.mul(torch.mul(2,t),t)),1))
    u_temp = temp1
    return u_temp.reshape([x1.size()[0], 1])


'''
Define the Neural Net
'''
class FCN(nn.Module):
    def __init__(self,layers,delta,device):
        super().__init__() #calls __init__ from parent class
        'activation function'
        self.activation = nn.Tanh() # hyperbolic tan
        'loss function'
        # self.loss_function = nn.MSELoss(reduction ='mean') # MSE loss
        'Initialise neural network as a list using nn.Modulelist'
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.iter = 0 #For the Optimizer
        self.delta = delta
        self.layers = layers
        self.device = device
        'Xavier Normal Initialization (Initializing the weights and bias)'
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) # weights
            nn.init.zeros_(self.linears[i].bias.data) # bias

    'Forward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    'Loss'
    def lossPDE(self, x_PDE, x_BC, x_IN):
        'PDE'
        x1=x_PDE.clone()
        x1.requires_grad=True # Enable differentiation (for back prop)
        u_hat1=self.forward(x1) # NN output

        u_hat1_x = u_hat1[:, 0].reshape([x1.size()[0],1])
        u_hat1_y = u_hat1[:, 1].reshape([x1.size()[0],1])
        grad_u_hat1_x = torch.autograd.grad(outputs = u_hat1_x, inputs = x1, grad_outputs = torch.ones(u_hat1_x.shape).to(self.device), create_graph = True)  # dx dy and dt
        grad_u_hat1_y = torch.autograd.grad(outputs = u_hat1_y, inputs = x1, grad_outputs = torch.ones(u_hat1_y.shape).to(self.device), create_graph = True)  # dx dy and dt
        dx1 = grad_u_hat1_x[0][:, 0].reshape([x1.size()[0],1])  # dx
        dy1 = grad_u_hat1_x[0][:, 1].reshape([x1.size()[0],1])  # dy
        dt1 = grad_u_hat1_x[0][:, 2].reshape([x1.size()[0],1])  # dt
        dx2 = grad_u_hat1_y[0][:, 0].reshape([x1.size()[0],1])  # dx
        dy2 = grad_u_hat1_y[0][:, 1].reshape([x1.size()[0],1])  # dy
        dt2 = grad_u_hat1_y[0][:, 2].reshape([x1.size()[0],1])  # dt
        L1 = torch.add(dt1,torch.mul(dx1, u_hat1_x))
        L1 = torch.add(L1,torch.mul(dy1, u_hat1_y))
        L1 = L1.pow(2).sum()
        L1 = torch.div(L1, len(x1))
        L2 = torch.add(dt2, torch.mul(dx2, u_hat1_x))
        L2 = torch.add(L2, torch.mul(dy2, u_hat1_y))
        L2 = L2.pow(2).sum()
        L2 = torch.div(L2, len(x1))
        R_pde = L1 + L2

        'boundary'
        x2 = x_BC.clone()
        x2.requires_grad = True

        x2_x0 = x2[0: len(x2)//4]
        u_hat2_x_0 = self.forward(x2_x0)
        u_hat2_x_0_p = g_0x1(x2_x0)
        L3 = torch.add(u_hat2_x_0_p, torch.neg(u_hat2_x_0))
        L3 = L3.pow(2).sum()

        x2_x1 = x2[len(x2)//4: len(x2)//2]
        u_hat2_x_1 = self.forward(x2_x1)
        u_hat2_x_1_p = g_1x1(x2_x1)
        L3 = torch.add(L3, torch.add(u_hat2_x_1_p, torch.neg(u_hat2_x_1)).pow(2).sum())
        L3 = torch.div(L3, len(x2))

        x2_y0 = x2[len(x2)//2: 3*len(x2)//4]
        u_hat2_y_0 = self.forward(x2_y0)
        u_hat2_y_0_p = g_0x2(x2_y0)
        L4 = torch.add(u_hat2_y_0_p, torch.neg(u_hat2_y_0)).pow(2).sum()

        x2_y1 = x2[3*len(x2)//4: len(x2)]
        u_hat2_y_1 = self.forward(x2_y1)
        u_hat2_y_1_p = g_1x2(x2_y1)
        L4 = torch.add(L4, torch.add(u_hat2_y_1_p, torch.neg(u_hat2_y_1)).pow(2).sum())
        L4 = torch.div(L4, len(x2))
        R_s = L3 + L4

        'initial'
        x3 = x_IN.clone()
        x3.requires_grad = True

        u_hat3 = self.forward(x3)
        u_hat3_x = u_hat3[:, 0].reshape([x3.size()[0],1])
        u_hat3_y = u_hat3[:, 1].reshape([x3.size()[0],1])
        u_0_x = u1_0(x3, self.delta).reshape([x3.size()[0],1])
        u_0_y = u2_0(x3, self.delta).reshape([x3.size()[0],1])
        L5 = torch.add(u_hat3_x, torch.neg(u_0_x)).pow(2).sum()
        L6 = torch.add(u_hat3_y, torch.neg(u_0_y)).pow(2).sum()
        L5 = torch.div(L5, len(x3))
        L6 = torch.div(L6, len(x3))
        R_t = L5 + L6

        return R_pde + R_s + R_t
    
    # Total Loss
    def loss(self, x_PDE,x_BC, x_IN):
        loss_pde = self.lossPDE(x_PDE, x_BC, x_IN)
        return loss_pde