import torch
import torch.autograd as autograd         # computation graph
import torch.nn as nn                     # neural networks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
Define the Neural Net
'''
class FCN(nn.Module):
    def __init__(self,layers,lambd,device):
        super().__init__() #calls __init__ from parent class
        'activation function'
        self.activation = nn.Tanh() # hyperbolic tan
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean') # MSE loss
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.layers = layers
        self.lambd = lambd
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
    
    'Loss Functions'
    'Loss BC'
    def lossBC(self,x_BC,y_BC):
        loss_BC = torch.tensor(0,dtype=torch.float).to(self.device)
        for i in range(int(x_BC.shape[1]/self.layers[0])):
            f = self.forward(x_BC[:,i*self.layers[0]:(i+1)*self.layers[0]]) # layer[0] gives the input dimension eg: [a,b] => 2 and [a,b,c] => 3
            loss_BC += self.lambd[i] * self.loss_function(f,y_BC[:,i:i+1])
      
        return loss_BC
    
    'Loss PDE'
    def lossPDE(self,x_PDE):
        'Pass through the NN'
        g=x_PDE.clone()
        g.requires_grad=True # Enable differentiation (for back prop)
        u=self.forward(g) # NN output

        'Pass through the PDE'
        u_x_t = autograd.grad(u,g,torch.ones([g.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0] # first derivative

        u_x=u_x_t[:,[0]] # select the 2nd element for t (the first one is x) (Remember the input X=[x,t])
        u_t=u_x_t[:,[1]]

        f=u_t + (u * u_x)

        f_hat = torch.zeros(f.shape[0],1).to(self.device) # to minimize function

        return self.loss_function(f,f_hat)
    
    'Total Loss'
    def loss(self,x_BC,y_BC,x_PDE):
        loss_bc=self.lossBC(x_BC,y_BC)
        loss_pde=self.lossPDE(x_PDE)
        return loss_bc+loss_pde

    def test_loss(self,x_BC_test,y_BC_test,x_PDE_test):
        loss_bc=self.lossBC(x_BC_test,y_BC_test)
        loss_pde=self.lossPDE(x_PDE_test)
        return loss_bc+loss_pde