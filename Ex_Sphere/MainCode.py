# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:12:49 2023

@author: MI
"""

import matplotlib.ticker as ticker
# import RHS
import torch
import torch.nn as nn                     # neural networks
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io
import warnings

import numpy as np
import time
import scipy.io
import os

warnings.filterwarnings("ignore")

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)   

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

## Adam Optimizer
steps=0
lr=1e-3

## LBFGS Optimizer
steps2=400
lr2=1e-1

#ns: Number of training points 
ns = 200
nsT = 500
ni = 1
animation = 1000

my_list = [3] + [50] * 5 + [1]
layers = np.array(my_list)

# Number of points
num_surface_points = 100
num_surface_pointsT = 100
num_inside_points = 1000

################################################
# Generate points on the surface of the sphere
theta = torch.linspace(0, torch.pi, num_surface_points)
phi = torch.linspace(0, 2 * torch.pi, num_surface_points)
theta, phi = torch.meshgrid(theta, phi)

x_surface = torch.sin(theta) * torch.cos(phi)
y_surface = torch.sin(theta) * torch.sin(phi)
z_surface = torch.cos(theta)

# Reshape the coordinates to form a matrix
dsites = torch.stack([x_surface.reshape(-1), y_surface.reshape(-1), z_surface.reshape(-1)], dim=1)
################################################
# Generate points on the surface of the sphere for Test
thetaT = torch.linspace(0, torch.pi, num_surface_pointsT)
phiT = torch.linspace(0, 2 * torch.pi, num_surface_pointsT)
thetaT, phiT = torch.meshgrid(thetaT, phiT)

x_surfaceT = torch.sin(thetaT) * torch.cos(phiT)
y_surfaceT = torch.sin(thetaT) * torch.sin(phiT)
z_surfaceT = torch.cos(thetaT)

# Reshape the coordinates to form a matrix
dsitesT = torch.stack([x_surfaceT.reshape(-1), y_surfaceT.reshape(-1), z_surfaceT.reshape(-1)], dim=1)

##############################################
# Generate points inside the sphere
# if ni == 1:
#     # For ni == 1, the inside points are set to (0, 0, 0)
#     intnode = torch.tensor([[0.0, 0.0, 0.0]])
# else:
#     # For ni != 1, generate points inside the unit sphere
#     x_inside = torch.rand(num_inside_points) * 2 - 1
#     y_inside = torch.rand(num_inside_points) * 2 - 1
#     z_inside = torch.rand(num_inside_points) * 2 - 1

#     inside_mask = (x_inside ** 2 + y_inside ** 2 + z_inside ** 2) <= 1

#     x_inside = x_inside[inside_mask]
#     y_inside = y_inside[inside_mask]
#     z_inside = z_inside[inside_mask]

#     intnode = torch.cat([x_inside.view(-1, 1), y_inside.view(-1, 1), z_inside.view(-1, 1)], dim=1)

# For ni != 1, generate points inside the unit sphere
x_inside = torch.rand(num_inside_points) * 2 - 1
y_inside = torch.rand(num_inside_points) * 2 - 1
z_inside = torch.rand(num_inside_points) * 2 - 1

inside_mask = (x_inside ** 2 + y_inside ** 2 + z_inside ** 2) <= 1

x_inside = x_inside[inside_mask]
y_inside = y_inside[inside_mask]
z_inside = z_inside[inside_mask]

intnode = torch.cat([x_inside.view(-1, 1), y_inside.view(-1, 1), z_inside.view(-1, 1)], dim=1)
##############################################################
rhs_sur =  torch.zeros(dsites.shape[0], 1)
rhs_surT =  torch.zeros(dsitesT.shape[0], 1)
rhs_in = torch.ones(intnode.shape[0],1)#x0**2+ y0**2 + z0**2 #
# rhs_in = intnode[:,0]**2+ intnode[:,1]**2 + intnode[:,2]**2 
# rhs_in = rhs_in.view(-1,1)

############################################################
############################################################
### select randomly
#surface
idx0=np.random.choice(dsites.shape[0],ns,replace=False)
dsites_ns =dsites[idx0,:]
rhs_sur_ns = rhs_sur[idx0,:]
#test
idx01=np.random.choice(dsites.shape[0],nsT,replace=False)
dsites_nsT =dsitesT[idx01,:]
rhs_sur_nsT = rhs_surT[idx01,:]
#interior
idx1=np.random.choice(intnode.shape[0],ni,replace=False)
intnode_ni =intnode[idx1,:]
rhs_in_ni =  rhs_in[idx1,:]
# surf+int
xy_train_Nu = torch.cat((intnode_ni, dsites_ns), dim=0)
u_train_Nu = torch.cat((rhs_in_ni, rhs_sur_ns), dim=0)

# test
xy_train_NuT = dsites_nsT #torch.cat((dsites_nsT), dim=0)
u_train_NuT = rhs_sur_nsT #torch.cat((rhs_sur_nsT), dim=0)
###########################################################
############################################################

grid_data = scipy.io.loadmat(r"C:\Users\MI\Dropbox\Amir_CG\3D - Copy\Sphere\griddata.mat") 

X_test0 = grid_data['epoints']     
X_test = torch.tensor(X_test0)                            

X_testT = torch.tensor(xy_train_NuT)                            
U_testT = torch.tensor(u_train_NuT)
############################################################
############################################################
iter = 0
lossLBFGS = []
errorLBFGS = []
u_pred_list = []

class FCN(nn.Module):
  #https://github.com/omniscientoctopus/Physics-Informed-Neural-Networks/tree/main/PyTorch/Burgers'%20Equation
    ##Neural Network
    def __init__(self,layers, skip_connection=True):
        super().__init__() #call __init__ from parent class 
        'activation function'
        self.activation0 = nn.Tanh() #nn.ReLU() #
        self.activation1 = nn.ReLU() #nn.Tanh() #
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')#L1Loss() #
        'Initialise neural network as a list using nn.Modulelist'  
        # self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) 
        self.linears = nn.ModuleList([nn.utils.weight_norm(nn.Linear(layers[i], layers[i+1])) for i in range(len(layers)-1)])

        self.iter = 0 #For the Optimizer
        self.skip_connection = skip_connection
        
        self.weights_history = []
        self.norms_history = []
        self.biases_history = []
        
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            # nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.xavier_normal_(self.linears[i].weight_v, gain=1.0)  # Use `weight_v` for weight normalization
            # set biases to zero
            nn.init.zeros_(self.linears[i].bias.data)   
    'foward pass'
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        a = x.float()
        for i in range(len(layers)-2):  
            z = self.linears[i](a)   
            a = self.activation0(z)  
            # a = (z)**2 + z +  a 
            # if (i) % 2 == 0:  # Skip connection every 2 layers
            #     a = (z)**1 +  a #(z + a)  # Add skip connectionz +    
                
        a = self.linears[-1](a)
        return a
    'Loss Functions'  
    #Loss BC
    def lossBC(self,x_BC,y_BC):
      loss_BC=self.loss_function(self.forward(x_BC),y_BC)
      return loss_BC

    def loss(self,x_BC,y_BC):
      loss_bc=self.lossBC(x_BC,y_BC)
      return loss_bc



    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(xy_train_Nu, u_train_Nu)
        loss.backward()
        global iter  # Reference the global variable
        
        if iter % animation == 0:
            u_pred = PINN.test()
            u_pred_list.append(u_pred)
        
        err = PINN.testT()
        lossLBFGS.append(loss.detach().cpu().numpy())
        errorLBFGS.append(err.detach().cpu().numpy())
    
        # Retrieve effective weights and norms for weight-normalized layers
        current_weights = {}
        current_norms = []
        for i, layer in enumerate(self.linears):
            if hasattr(layer, 'weight_g') and hasattr(layer, 'weight_v'):  # Check for weight normalization
                weight_v = layer.weight_v.data.clone()
                weight_g = layer.weight_g.data.clone()
                effective_weight = (weight_g / torch.norm(weight_v)) * weight_v
                current_weights[f'layer_{i}'] = effective_weight
                current_norms.append(torch.norm(effective_weight))
            else:
                current_weights[f'layer_{i}'] = layer.weight.data.clone()
                current_norms.append(torch.norm(layer.weight.data))
    
        self.weights_history.append(current_weights)
        self.norms_history.append(current_norms)
    
        # Retrieve biases
        current_biases = {f'layer_{i}': self.linears[i].bias.data.clone() for i in range(len(self.linears))}
        self.biases_history.append(current_biases)
    
        # Retrieve gradients for biases
        current_gradients_bias = {f'layer_{i}': self.linears[i].bias.grad.clone() for i in range(len(self.linears))}
        self.gradients_history_bias.append(current_gradients_bias)
    
        iter += 1
        if iter % 500 == 0:
            error_vec = PINN.testT()
            print(iter, loss.detach().cpu().numpy(), error_vec.cpu().detach().numpy())
        
        return loss



    # def closure(self):
    #     optimizer.zero_grad()
    #     loss = self.loss(xy_train_Nu, u_train_Nu)
    #     loss.backward()
    #     global iter  # Reference the global variable
    #     if iter % animation == 0:
    #         u_pred = PINN.test()
    #         u_pred_list.append(u_pred)
    #     lossLBFGS.append(loss.detach().cpu().numpy())
        
    #     err = PINN.testT()
    #     errorLBFGS.append(err.detach().cpu().numpy())

        
    #     # optimizer.zero_grad()
    #     # loss = self.loss(xy_train_Nu, u_train_Nu)
    #     # loss.backward()

    #     current_weights = {f'layer_{i}': self.linears[i].weight.data.clone() for i in range(len(layers)-1)}
    #     self.weights_history.append(current_weights)

    #     current_norms = [torch.norm(layer_weights) for layer_weights in current_weights.values()]
    #     # current_norms = [torch.linalg.norm(layer_weights, ord=2) for layer_weights in current_weights.values()]
    #     self.norms_history.append(current_norms)

    #     current_biases = {f'layer_{i}': self.linears[i].bias.data.clone() for i in range(len(layers)-1)}
    #     self.biases_history.append(current_biases)
        
    #     current_gradients = {f'layer_{i}': self.linears[i].weight.grad.clone() for i in range(len(layers)-1)}
    #     self.gradients_history.append(current_gradients)

    #     # current_gradients_bias = {f'layer_{i}': self.linears[i].bias.grad.clone() for i in range(len(layers)-1)}
    #     # self.gradients_history_bias.append(current_gradients_bias)
        


    
    
    #     global iter  # Reference the global variable
    #     # # err = PINN.test()
    #     # u_pred = PINN.test()
    #     # u_pred_list.append(u_pred)
    #     # lossLBFGS.append(loss.detach().cpu().numpy())
    #     # errorLBFGS.append(err.detach().cpu().numpy())
    #     iter += 1
    #     if iter % 100 == 0:
    #         error_vec = PINN.testT()
    #         print(iter,loss.detach().cpu().numpy(),error_vec.cpu().detach().numpy())
    #     return loss        



    'test neural network'
    def test(self):
        u_pred = self.forward(X_test)
        # u_predT = self.forward(X_testT)
        # error_vec =  torch.max(torch.abs(U_testT - u_predT))#torch.linalg.norm((U_testT-u_predT),2)/torch.linalg.norm(U_testT,2)# torch.tensor([1.0])# torch.linalg.norm((U_test-u_pred),2)/Nu  #    # Relative L2 Norm of the error (Vector)
        u_pred = u_pred.cpu().detach().numpy()
        # error_vec = error_vec.cpu().detach().numpy()
        
        # u_pred = np.reshape(u_pred,(n0,n0),order='F')
        return u_pred
    
    def testT(self):
        # u_pred = self.forward(X_test)
        u_predT = self.forward(X_testT)
        error_vec =  torch.max(torch.abs(U_testT - u_predT))#torch.linalg.norm((U_testT-u_predT),2)/torch.linalg.norm(U_testT,2)# torch.tensor([1.0])# torch.linalg.norm((U_test-u_pred),2)/Nu  #    # Relative L2 Norm of the error (Vector)
        # u_pred = u_pred.cpu().detach().numpy()
        # error_vec = error_vec.cpu().detach().numpy()
        
        # u_pred = np.reshape(u_pred,(n0,n0),order='F')
        return error_vec
    
    def hessian(self):
        n_params = sum(p.numel() for p in self.parameters())
        hessian_matrix = torch.zeros(n_params, n_params)
        grad_flat = torch.cat([g.view(-1) for g in torch.autograd.grad(self.loss(xy_train_Nu, u_train_Nu), self.parameters(), create_graph=True)])

        for idx, grad_elem in enumerate(grad_flat):
            grad2 = torch.autograd.grad(grad_elem, self.parameters(), retain_graph=True)
            hessian_matrix[idx] = torch.cat([g.contiguous().view(-1) for g in grad2])

        return hessian_matrix

    def compute_eigenvalues(self):
        hessian_matrix = self.hessian()
        eigenvalues = torch.linalg.eigvals(hessian_matrix)
        
        return eigenvalues

    def compute_condition_number(self):
        hessian_matrix = self.hessian()
        u, s, v = torch.svd(hessian_matrix)
        condition_number = torch.max(s) / torch.min(s)
        return condition_number.item()  # Convert to Python scalar

################################################

print("Total test  points:",X_test.shape)


#Create Model
PINN = FCN(layers, skip_connection=True)
PINN.to(device)
# print(PINN)

#################################
##Adam
##############################
#########LBFGS################
PINN.gradients_history = []
PINN.gradients_history_bias = []

start_time_L = time.time()
PINN.gradients_history = []

'L-BFGS Optimizer'
optimizer = torch.optim.LBFGS(PINN.parameters(), lr2, 
                              max_iter = steps2, 
                              max_eval = None, 
                              tolerance_grad = 1e-11, 
                              tolerance_change = 1e-11, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')

optimizer.step(PINN.closure)
elapsed_L = time.time() - start_time_L       
u_predict2 = PINN(X_test)
# e_lbfgs_rel2n = torch.linalg.norm((U_test-u_predict2),2)/torch.linalg.norm(U_test,2)  # Relative L2 Norm of the error (Vector)
# e_lbfgs_mae =  torch.max(torch.abs(U_test - u_predict2))  # MAE
print('----------------------------------------------')
# print('LBFGS reL2norm Error: %.2e' % (e_lbfgs_rel2n))
# print('LBFGS MAE Error: %.2e' % (e_lbfgs_mae))
print('Training time LBFGS: %.2f' % (elapsed_L))
print('----------------------------------------------')
####################
####################
################
# ### Plot loss
# Assuming lossLBFGS is a list or array of loss values
lossLBFGS = np.array(lossLBFGS)

epochs = np.array(range(len(lossLBFGS)))
scaling_factor = 1 - (epochs / (len(lossLBFGS) - 1)) * (1 - 1)
# scaling_factor = 1 - (epochs / (len(lossLBFGS) - 1)) * (1 - 0.1)

# scaled_lossLBFGS = 0.2*scaling_factor * lossLBFGS
scaled_lossLBFGS = scaling_factor * lossLBFGS



errorLBFGS = np.array(errorLBFGS)

epochs = np.array(range(len(errorLBFGS)))
scaling_factor = 1 - (epochs / (len(errorLBFGS) - 1)) * (1 - 1)
# scaling_factor = 1 - (epochs / (len(errorLBFGS) - 1)) * (1 - 0.2)

scaled_errorLBFGS = 1*scaling_factor * errorLBFGS
# scaled_errorLBFGS = scaling_factor * errorLBFGS

# Plot the results
# plt.plot(epochs, scaled_lossLBFGS, linewidth=2, linestyle=':', color='b', label='Plain N')
plt.plot(epochs, scaled_errorLBFGS, linewidth=2, linestyle='-', color='g', label='WN-Reparam')

plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Display only integer x-axis tick labels
plt.grid(axis='y', linestyle='--')  # Display horizontal grid lines as dashed lines
plt.grid(axis='x', which='both', linestyle='-', linewidth=0)  # Remove vertical grid lines
plt.yscale('log')
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()
#######################
## colorful loss
###### Cyan
errorLBFGS=scaled_errorLBFGS 
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

# Dummy data (replace this with your actual `errorLBFGS` and `epochs` data)
i = 0
epochs = np.array(range(i, i + len(errorLBFGS)))

# Create a custom colormap by replacing the yellow color in "Accent"
original_cmap = cm.get_cmap("Accent", 8)
colors = original_cmap(np.arange(original_cmap.N))
colors[3] = [0, 1, 1, 1]  # Replace yellow (4th color) with cyan
custom_cmap = ListedColormap(colors)

# Define the colormap and normalize it
segments = len(errorLBFGS) // 8  # Divide epochs into 8 segments
colors = np.concatenate([np.tile(custom_cmap(j), (segments, 1)) for j in range(8)])

# Plotting each segment with its color
for j in range(len(colors) - 1):
    plt.plot(epochs[j:j+2], errorLBFGS[j:j+2], color=colors[j], linewidth=2)

plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Display only integer x-axis tick labels
plt.grid(axis='y', linestyle='--')  # Display horizontal grid lines as dashed lines
plt.grid(axis='x', which='both', linestyle='-', linewidth=0)  # Remove vertical grid lines
plt.yscale('log')
plt.tight_layout()
plt.show()


#############################################################
##########################################################################
##########################################################################

# mat_data = {
#     'pfa': u_predict2.detach().cpu().numpy(),
# }

# scipy.io.savemat('pf_data1.mat', mat_data)
# # tt = PINN.weights_history



mat_data = {
    'pfa': u_predict2.detach().cpu().numpy(),
    'pfa_list': u_pred_list,
    
}

scipy.io.savemat('pf_data1.mat', mat_data)





###################################################
###################################################
###################################################
###################################################
# Compute Hessian matrix
# hessian_matrix = PINN.hessian()
# print("Hessian matrix:")
# print(hessian_matrix)
###################################################
###################################################
###################################################
###################################################
# # Compute eigenvalues of the Hessian matrix
# eigenvalues = PINN.compute_eigenvalues()
# print("Eigenvalues of the Hessian matrix:")
# print(eigenvalues)
# # plot eigen values
# plt.figure(figsize=(8, 6))
# plt.plot(eigenvalues.cpu().numpy().real, 'o')
# plt.title('Real Parts of the Eigenvalues of the Hessian Matrix')
# plt.xlabel('Eigenvalue Index')
# plt.ylabel('Real Part')
# plt.grid(True)
# plt.yscale('log')  # Set y-axis to logarithmic scale
# plt.show()
###################################################
###################################################
###################################################
###################################################
# # compute condition number of Hessian matrix
# pinhessian_condition_number = PINN.compute_condition_number()
# print('Con Hessian matrix: : %.2e' %(pinhessian_condition_number))
###################################################
###################################################
###################################################
###################################################

##################################################
#################################################
####################################################
######################################################

##########################################################
#######################################################
#########################################################
lastIter = iter

####################################################################################
####################################################################################
####################################################################################
### Weights and biases
# # results_directory = "D:\\results_cg"

# for i, (weights_dict, biases_dict) in enumerate(zip(PINN.weights_history, PINN.biases_history)):
#     mat_dataW = {}

#     # Save weights
#     for j, layer_weights in weights_dict.items():
#         mat_dataW[f'weights_iter_{i}_{j}'] = layer_weights.cpu().numpy()

#     # Save biases
#     for j, layer_biases in biases_dict.items():
#         mat_dataW[f'biases_iter_{i}_{j}'] = layer_biases.cpu().numpy()

#     # file_path = os.path.join(results_directory, f'weights_biases_iter_{i}.mat')
#     # scipy.io.savemat(file_path, mat_dataW)
    


# ####plot the norm of the weights  
# # # After training all layers in 1 plot
# num_hidden_layers = len(PINN.linears) - 1

# # plt.figure(figsize=(8, 6))
# plt.figure()
# for layer_idx in range(1, num_hidden_layers):  # Exclude the first and last layers
#     norms_for_layer = [norms[layer_idx] for norms in PINN.norms_history]
#     plt.plot(range(1, len(norms_for_layer) + 1), norms_for_layer, label=f'Layer {layer_idx}')

# plt.title('Norms of Weights for Intermediate Layers')
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Norm of Weights', fontsize=16)
# plt.legend()
# plt.grid(True)
# # plt.xlim(0, 1300)
# # plt.ylim(6.8, 8.5)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(fontsize=14, loc='lower right')
# plt.tight_layout()
# plt.show()
########################
######################
## different colors for weights:
# ############
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import numpy as np
# from matplotlib.colors import ListedColormap
# num_hidden_layers = len(PINN.linears) - 1

# # Get the original "Accent" colormap
# original_cmap = cm.get_cmap("Accent", 8)

# # Convert the colormap to an array of RGBA values
# colors = original_cmap(np.arange(original_cmap.N))

# # Replace the yellow color (4th color in Accent colormap) with cyan
# colors[3] = [0, 1, 1, 1]  # Cyan in RGBA (R=0, G=1, B=1, A=1)

# # Create a new colormap
# custom_cmap = ListedColormap(colors)

# # Example usage
# plt.figure()
# num_epochs = len(PINN.norms_history)
# segments = num_epochs // 8

# for layer_idx in range(1, num_hidden_layers):
#     norms_for_layer = [norms[layer_idx] for norms in PINN.norms_history]
    
#     for epoch in range(num_epochs - 1):
#         color_idx = epoch // segments
#         plt.plot(
#             [epoch + 1, epoch + 2],
#             [norms_for_layer[epoch], norms_for_layer[epoch + 1]],
#             color=custom_cmap(color_idx),
#             linewidth=1.5,
#         )

# plt.title('Norms of Weights for Intermediate Layers', fontsize=16)
# plt.xlabel('Epoch', fontsize=16)
# plt.ylabel('Norm of Weights', fontsize=16)
# plt.grid(True)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.show()


###################################################################################
###################################################################################
###################################################################################
# ### Gradients stuff
# # # Access saved gradients
# # for i, gradients_dict in enumerate(PINN.gradients_history):
# #     mat_dataG = {}
# #     for j, layer_gradients in gradients_dict.items():
# #         mat_dataG[f'gradients_iter_{i}_{j}'] = layer_gradients.cpu().numpy()
# #     scipy.io.savemat(f'gradients_iter_{i}.mat', mat_dataG)

# last_gradients = PINN.gradients_history[lastIter]

# # plot histograms for each layer
# for i, layer_gradients in last_gradients.items():
#     plt.figure()
#     plt.hist(layer_gradients.flatten(), bins=50, density=True, color='blue', alpha=0.7)
#     plt.title(f'histogram of gradients for {i}')
#     plt.xlabel('gradient value')
#     plt.ylabel('frequency')
#     plt.grid(True)
#     plt.yscale('log')  # Set y-axis to logarithmic scale
#     plt.show()
####################################################################################
####################################################################################
####################################################################################
#### PCA
# import numpy as np
# from sklearn.decomposition import PCA

# # Convert each weight matrix to a NumPy array and concatenate them into a single matrix
# all_weights = np.concatenate([weights.cpu().numpy().flatten() for layer_weights in PINN.weights_history for _, weights in layer_weights.items()])

# # Reshape the concatenated matrix to have one weight vector per row
# all_weights = all_weights.reshape(len(PINN.weights_history), -1)

# # Perform PCA
# pca = PCA(n_components=2)  # Choose the number of principal components you want to analyze
# principal_components = pca.fit_transform(all_weights)
# # Plot PCA results
# # plt.figure(figsize=(8, 6))
# # plt.scatter(principal_components[:, 0], principal_components[:, 1])
# # plt.title('PCA of Weight Matrices')
# # plt.xlabel('Principal Component 1')
# # plt.ylabel('Principal Component 2')
# # plt.grid(True)
# # plt.show()


# # Plot PCA results with different colors for each iteration
# plt.figure()
# sc = plt.scatter(principal_components[0:lastIter, 0], principal_components[0:lastIter, 1], c=np.arange(lastIter), cmap='viridis', alpha=0.5)
# plt.title('PCA of Weight Matrices', fontsize=16)
# plt.xlabel('Principal Component 1', fontsize=16)
# plt.ylabel('Principal Component 2', fontsize=16)
# cb = plt.colorbar(sc, label='Epoch')
# cb.ax.yaxis.label.set_size(16)  # Set font size for the colorbar label
# cb.ax.tick_params(labelsize=16)  # Set font size for the colorbar tick labels
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#########################################
##########################################
# ### two plots in one figure for PCA
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from matplotlib.colors import ListedColormap
# import matplotlib.cm as cm

# # Assuming `PINN.weights_history` and `lastIter` are already defined

# # Concatenate and reshape weights
# all_weights = np.concatenate([weights.cpu().numpy().flatten() for layer_weights in PINN.weights_history for _, weights in layer_weights.items()])
# all_weights = all_weights.reshape(len(PINN.weights_history), -1)

# # Perform PCA
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(all_weights)

# # Create a custom colormap by replacing yellow in "Accent"
# original_cmap = cm.get_cmap("Accent", 8)
# colors = original_cmap(np.arange(original_cmap.N))
# colors[3] = [0, 1, 1, 1]  # Replace yellow with cyan
# custom_cmap = ListedColormap(colors)

# # Plot PCA results
# plt.figure()
# sc1 = plt.scatter(
#     principal_components[0:lastIter, 0], 
#     principal_components[0:lastIter, 1], 
#     c=np.arange(lastIter), 
#     cmap=custom_cmap,  # Use the custom colormap
#     alpha=0.5
# )
# plt.title('PCA of Weight Matrices', fontsize=16)
# plt.xlabel('Principal Component 1', fontsize=16)
# plt.ylabel('Principal Component 2', fontsize=16)
# # plt.tick_params(axis='both', labelsize=14)  # Major ticks

# cbar1 = plt.colorbar(sc1, label='Epoch', location='right')
# cbar1.ax.yaxis.label.set_size(16)  # Set font size for the colorbar label
# cbar1.ax.tick_params(labelsize=16)  # Set font size for the colorbar tick labels
# plt.grid(True)

# plt.tight_layout()
# plt.show()

########################
##################
##############


############################################
#########################################
# ### vairiance:
# ## Plot the explained variance ratio
# # Perform PCA
# pca = PCA(n_components=2)  # Choose the number of principal components you want to analyze
# principal_components = pca.fit_transform(all_weights)

# # Explained variance ratio
# explained_variance_ratio = pca.explained_variance_ratio_

# # Display explained variance ratio
# print("Explained Variance Ratio (Principal Component 1 and 2):", explained_variance_ratio)

# # Plot PCA results with different colors for each iteration
# plt.figure()
# sc = plt.scatter(principal_components[0:lastIter, 0], principal_components[0:lastIter, 1], 
#                  c=np.arange(lastIter), cmap='viridis', alpha=0.5)
# plt.title('PCA of Weight Matrices')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.colorbar(sc, label='Iteration Number')
# plt.grid(True)
# plt.show()

# # Plot the explained variance ratio
# plt.figure()
# plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100)
# plt.title('Explained Variance Ratio by Principal Component')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio (%)')
# plt.xticks(range(1, len(explained_variance_ratio) + 1))
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


############
# Higher PCA
# # Perform PCA
# pca = PCA(n_components=3)  # Choose the number of principal components you want to analyze
# principal_components = pca.fit_transform(all_weights)


# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2])
# ax.set_title('PCA of Weight Matrices')
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# plt.show()


# # Perform PCA
# pca = PCA(n_components=4)  # Choose the number of principal components you want to analyze
# principal_components = pca.fit_transform(all_weights)

# # Plot PCA results
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=principal_components[:, 3], cmap='viridis')
# ax.set_title('PCA of Weight Matrices')
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# fig.colorbar(sc, label='Principal Component 4')
# plt.tight_layout()
# plt.show()
####################################################################################
####################################################################################
####################################################################################
# ##plot svd
# # Perform Singular Value Decomposition (SVD)
# u, s, vh = np.linalg.svd(all_weights)
# condition_number = np.max(s) / np.min(s)
# print("Condition number:", "{:.2e}".format(condition_number))

# # Plot singular values
# plt.figure()
# plt.plot(s, marker='o', linestyle='-', color='b')
# plt.title('Singular Values of Weight Matrices')
# plt.xlabel('Singular Value Index')
# plt.ylabel('Singular Value')
# plt.yscale('log')  # Set y-axis to logarithmic scale
# plt.grid(True)
# plt.show()