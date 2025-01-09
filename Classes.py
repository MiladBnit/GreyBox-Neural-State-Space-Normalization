# %% [markdown]
# ## SLIMPEC Task 1.1 Cascaded Tanks Benchmark - Classes

# %% [markdown]
# #### Imports

# %%
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from Functions import hard_constraint_projection

# %% [markdown]
# #### Normalization Class

# %%
class DataScaler():

    def __init__(self, x_mean, x_std, u_mean, u_std, y_mean, y_std, device):
        self.x_mean = x_mean
        self.x_std = x_std
        self.u_mean = u_mean
        self.u_std = u_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device
    
    def move_to_device(self, device):
        self.x_mean = self.x_mean.to(device)
        self.x_std = self.x_std.to(device)
        self.u_mean = self.u_mean.to(device)
        self.u_std = self.u_std.to(device)
        self.y_mean = self.y_mean.to(device)
        self.y_std = self.y_std.to(device)
        self.device = device
        
    def zscore_normalize(self, s, s_mean, s_std):
        s_norm = (s.to(self.device) - s_mean.to(self.device))/s_std.to(self.device)
        return s_norm
    
    def zscore_denormalize(self, s_norm, s_mean, s_std):
        s_denorm = s_norm.to(self.device) * s_std.to(self.device) + s_mean.to(self.device)
        return s_denorm

# Black-Box Components
# %%
class FANSNeuralStateUpdate(nn.Module):
        
        def __init__(self, n_x_f, n_x_b, n_x_o, n_u, Binary_mat, device, n_feat=32, num_hidden_layers=1, init_std=1e-4, neuron_list=[], act_fn = nn.Tanh(),
                      include_bias={"input layer": False, "hidden layer": False, "output layer": False}, *args, **kwargs):
            super(FANSNeuralStateUpdate, self).__init__()
            
            self.act_fn = act_fn
            
            self.Binary_mat = Binary_mat
            self.n_x_f = n_x_f
            self.n_x_b = n_x_b
            self.n_x_o = n_x_o
            self.n_u = n_u

            if neuron_list == []:
                neuron_list = [n_feat for _ in range(num_hidden_layers)]
            self.neuron_list = neuron_list

            self.net_list = []
            self.init_std = init_std

            full_input_condition = []
            for i in range(self.n_x_o):
                condition = 1
                for j in range(self.n_x_f + self.n_x_b):
                    condition = condition * self.Binary_mat[i, j]
                full_input_condition.append(condition)

            selected_state_idx_mat = []
            n_x_o_mat = []
            
            i = 0
            while i < self.n_x_o:
                
                if full_input_condition[i] == 0:
                    selected_state_idx = []

                    for j in range(self.n_x_f + self.n_x_b):

                        if self.Binary_mat[i, j] == 1:
                            selected_state_idx.append(j)
                    
                    selected_state_idx_mat.append(selected_state_idx)
                    n_x_o_mat.append(1)
                    
                elif full_input_condition[i] == 1:
                    
                    n_x_o_counter = 1
                    while i + n_x_o_counter < self.n_x_o and full_input_condition[i + n_x_o_counter] == 1:
                        n_x_o_counter = n_x_o_counter + 1

                    selected_state_idx_mat.append([j_idx for j_idx in range(0, self.n_x_f + self.n_x_b)])
                    n_x_o_mat.append(n_x_o_counter)
                    i = i + n_x_o_counter - 1
                
                i = i + 1

            for i in range(len(n_x_o_mat)):
                
                selected_state_idx = selected_state_idx_mat[i]
                n_x_selected = len(selected_state_idx)

                subnet = nn.Sequential()
                if include_bias["input layer"]:
                    subnet.add_module('Input Layer', nn.Sequential(nn.Linear(n_x_selected+self.n_u, self.neuron_list[0]), self.act_fn))
                else:
                    subnet.add_module('Input Layer', nn.Sequential(nn.Linear(n_x_selected+self.n_u, self.neuron_list[0], bias=False), self.act_fn))

                for l in range(0, len(self.neuron_list) - 1):
                    if include_bias["hidden layer"]:
                        subnet.add_module('Deep Layer ' + str(l + 1), nn.Linear(self.neuron_list[l], self.neuron_list[l+1]))
                    else:
                        subnet.add_module('Deep Layer ' + str(l + 1), nn.Linear(self.neuron_list[l], self.neuron_list[l+1], bias=False))

                    subnet.add_module('Activation Function for Hidden Layer ' + str(l + 1), self.act_fn)

                if include_bias["output layer"]:
                    subnet.add_module('Output Layer ', nn.Linear(self.neuron_list[-1], n_x_o_mat[i]))
                else:
                    subnet.add_module('Output Layer ', nn.Linear(self.neuron_list[-1], n_x_o_mat[i], bias=False))
                
                for name, m in subnet.named_modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=self.init_std)

                        if name == 'Input Layer' and include_bias["input layer"]:
                            nn.init.constant_(m.bias, val=0)
                        elif name == 'Deep Layer' and include_bias["hidden layer"]:
                            nn.init.constant_(m.bias, val=0)
                        elif name == 'Output Layer' and include_bias["output layer"]:
                            nn.init.constant_(m.bias, val=0)

                setattr(self, 'subnet_' + str(i + 1), subnet)
                self.net_list.append(subnet)
            
            self.n_x_o_mat = n_x_o_mat
            self.selected_state_idx_mat = selected_state_idx_mat
            self.device = device

        def move_to_device(self, device):
            self.device = device
            
        def forward(self, x_f, x_b, u):

            x_tilde = torch.cat((x_f, x_b), dim=-1)
            B, _ = x_f.shape
            dx = torch.empty((B, self.n_x_o)).to(self.device)

            for i in range(len(self.n_x_o_mat)):

                selected_state_idx = self.selected_state_idx_mat[i]
                z = torch.cat((x_tilde[:, selected_state_idx], u), dim=-1)
                dx[:, i:i+self.n_x_o_mat[i]] = self.net_list[i](z)

            return dx
        

class FANSNeuralOutputUpdate(nn.Module):
        
        def __init__(self, n_x_f, n_x_b, n_x_o, Binary_mat, device, n_feat=32, num_hidden_layers=1, init_std=1e-4, neuron_list=[], act_fn = nn.Tanh(),
                     include_bias={"input layer": True, "hidden layer": True, "output layer": True}, *args, **kwargs):
            super(FANSNeuralOutputUpdate, self).__init__()
            
            self.act_fn = act_fn
            
            self.Binary_mat = Binary_mat
            self.n_x_f = n_x_f
            self.n_x_b = n_x_b
            self.n_x_o = n_x_o

            if neuron_list == []:
                neuron_list = [n_feat for _ in range(num_hidden_layers)]
            self.neuron_list = neuron_list

            self.net_list = []
            self.init_std = init_std

            full_input_condition = []
            for i in range(self.n_x_o):
                condition = 1
                for j in range(self.n_x_f + self.n_x_b):
                    condition = condition * self.Binary_mat[i, j]
                full_input_condition.append(condition)

            selected_state_idx_mat = []
            n_x_o_mat = []
            
            i = 0
            while i < self.n_x_o:
                
                if full_input_condition[i] == 0:
                    selected_state_idx = []

                    for j in range(self.n_x_f + self.n_x_b):

                        if self.Binary_mat[i, j] == 1:
                            selected_state_idx.append(j)
                    
                    selected_state_idx_mat.append(selected_state_idx)
                    n_x_o_mat.append(1)
                    
                elif full_input_condition[i] == 1:
                    
                    n_x_o_counter = 1
                    while i + n_x_o_counter < self.n_x_o and full_input_condition[i + n_x_o_counter] == 1:
                        n_x_o_counter = n_x_o_counter + 1

                    selected_state_idx_mat.append([j_idx for j_idx in range(0, self.n_x_f + self.n_x_b)])
                    n_x_o_mat.append(n_x_o_counter)
                    i = i + n_x_o_counter - 1
                
                i = i + 1
            
            for i in range(len(n_x_o_mat)):
                selected_state_idx = []

                selected_state_idx = selected_state_idx_mat[i]
                n_x_selected = len(selected_state_idx)

                subnet = nn.Sequential()
                if include_bias["input layer"]:
                    subnet.add_module('Input Layer', nn.Sequential(nn.Linear(n_x_selected, self.neuron_list[0]), self.act_fn))
                else:
                    subnet.add_module('Input Layer', nn.Sequential(nn.Linear(n_x_selected, self.neuron_list[0], bias=False), self.act_fn))

                for l in range(len(self.neuron_list) - 1):
                    if include_bias["hidden layer"]:
                        subnet.add_module('Deep Layer ' + str(l+1), nn.Linear(self.neuron_list[l], self.neuron_list[l+1]))
                    else:
                        subnet.add_module('Deep Layer ' + str(l+1), nn.Linear(self.neuron_list[l], self.neuron_list[l+1], bias=False))
                    subnet.add_module('Activation Function for Hidden Layer ' + str(l+1), self.act_fn)

                if include_bias["output layer"]:
                    subnet.add_module('Output Layer ', nn.Linear(self.neuron_list[-1], n_x_o_mat[i]))
                else:
                    subnet.add_module('Output Layer ', nn.Linear(self.neuron_list[-1], n_x_o_mat[i], bias=False))
                
                for name, m in subnet.named_modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=self.init_std)

                        if name == 'Input Layer' and include_bias["input layer"]:
                            nn.init.constant_(m.bias, val=0)
                        elif name == 'Deep Layer' and include_bias["hidden layer"]:
                            nn.init.constant_(m.bias, val=0)
                        elif name == 'Output Layer' and include_bias["output layer"]:
                            nn.init.constant_(m.bias, val=0)

                setattr(self, 'subnet_' + str(i + 1), subnet)
                self.net_list.append(subnet)

            self.n_x_o_mat = n_x_o_mat
            self.selected_state_idx_mat = selected_state_idx_mat
            self.device = device

        def move_to_device(self, device):
            self.device = device
        
        def forward(self, x_f, x_b):

            x_tilde = torch.cat((x_f, x_b), dim=-1)
            B, _ = x_f.shape
            y = torch.empty((B, self.n_x_o)).to(self.device)

            for i in range(len(self.n_x_o_mat)):
                
                selected_state_idx = self.selected_state_idx_mat[i]        
                z = x_tilde[:, selected_state_idx]
                y[:, i:i+self.n_x_o_mat[i]] = self.net_list[i](z)

            return y

        
class FANSNeuralStateUpdate_Zero(nn.Module):
        
        def __init__(self, n_x_o, device, *args, **kwargs):
            super(FANSNeuralStateUpdate_Zero, self).__init__()
            self.n_x_o = n_x_o
            self.device = device

        def forward(self, x_f, x_b, u):
            B, _ = x_f.shape
            dx = torch.zeros((B, self.n_x_o)).to(self.device)
            return dx

class FANSNeuralOutputUpdate_Zero(nn.Module):
        
        def __init__(self, n_x_o, device, *args, **kwargs):
            super(FANSNeuralOutputUpdate_Zero, self).__init__()
            self.n_x_o = n_x_o
            self.device = device

        def forward(self, x_f, x_b):
            B, _ = x_f.shape
            y = torch.zeros((B, self.n_x_o)).to(self.device)
            return y


class FANSNeuralStateUpdate_Nonphys(nn.Module):
        
        def __init__(self, n_x_b, n_x_o, n_u, Binary_mat, device, n_feat=32, num_hidden_layers=1, *args, **kwargs):
            super(FANSNeuralStateUpdate_Nonphys, self).__init__()
            
            self.Binary_mat = Binary_mat
            self.n_x_b = n_x_b
            self.n_x_o = n_x_o
            self.n_u = n_u
            self.n_feat = n_feat
            self.num_hidden_layers = num_hidden_layers
            self.net_list = []
            
            full_input_condition = []
            for i in range(self.n_x_o):
                condition = 1
                for j in range(self.n_x_b):
                    condition = condition * self.Binary_mat[i, j]
                full_input_condition.append(condition)

            selected_state_idx_mat = []
            n_x_o_mat = []
            
            i = 0
            while i < self.n_x_o:
                
                if full_input_condition[i] == 0:
                    selected_state_idx = []

                    for j in range(self.n_x_b):

                        if self.Binary_mat[i, j] == 1:
                            selected_state_idx.append(j)
                    
                    selected_state_idx_mat.append(selected_state_idx)
                    n_x_o_mat.append(1)
                    
                elif full_input_condition[i] == 1:
                    
                    n_x_o_counter = 1
                    while i + n_x_o_counter < self.n_x_o and full_input_condition[i + n_x_o_counter] == 1:
                        n_x_o_counter = n_x_o_counter + 1

                    selected_state_idx_mat.append([j_idx for j_idx in range(0, self.n_x_b)])
                    n_x_o_mat.append(n_x_o_counter)
                    i = i + n_x_o_counter - 1
                
                i = i + 1

            for i in range(len(n_x_o_mat)):
                
                selected_state_idx = selected_state_idx_mat[i]
                n_x_selected = len(selected_state_idx)

                subnet = nn.Sequential()
                subnet.add_module('Input Layer', nn.Sequential(nn.Linear(n_x_selected+self.n_u, self.n_feat), nn.Tanh()))

                for l in range(self.num_hidden_layers - 1):
                    subnet.add_module('Deep Layer ' + str(l + 1), nn.Linear(self.n_feat, self.n_feat))
                    subnet.add_module('Activation Function for Hidden Layer ' + str(l + 1), nn.Tanh())

                subnet.add_module('Output Layer ', nn.Linear(self.n_feat, n_x_o_mat[i]))
                
                for m in subnet.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-8)
                        nn.init.constant_(m.bias, val=0)

                setattr(self, 'subnet_' + str(i + 1), subnet)
                self.net_list.append(subnet)
            
            self.n_x_o_mat = n_x_o_mat
            self.selected_state_idx_mat = selected_state_idx_mat
            self.device = device

        def forward(self, x_b, u):

            B, _ = x_b.shape
            dx = torch.empty((B, self.n_x_o)).to(self.device)

            for i in range(len(self.n_x_o_mat)):

                selected_state_idx = self.selected_state_idx_mat[i]
                z = torch.cat((x_b[:, selected_state_idx], u), dim=-1)
                dx[:, i:i+self.n_x_o_mat[i]] = self.net_list[i](z)

            return dx
        

class FANSNeuralOutputUpdate_Nonphys(nn.Module):
        
        def __init__(self, n_x_b, n_x_o, Binary_mat, device, n_feat=32, num_hidden_layers=1, *args, **kwargs):
            super(FANSNeuralOutputUpdate_Nonphys, self).__init__()
            
            self.Binary_mat = Binary_mat
            self.n_x_b = n_x_b
            self.n_x_o = n_x_o
            self.n_feat = n_feat
            self.num_hidden_layers = num_hidden_layers
            self.net_list = []

            full_input_condition = []
            for i in range(self.n_x_o):
                condition = 1
                for j in range(self.n_x_b):
                    condition = condition * self.Binary_mat[i, j]
                full_input_condition.append(condition)

            selected_state_idx_mat = []
            n_x_o_mat = []
            
            i = 0
            while i < self.n_x_o:
                
                if full_input_condition[i] == 0:
                    selected_state_idx = []

                    for j in range(self.n_x_b):

                        if self.Binary_mat[i, j] == 1:
                            selected_state_idx.append(j)
                    
                    selected_state_idx_mat.append(selected_state_idx)
                    n_x_o_mat.append(1)
                    
                elif full_input_condition[i] == 1:
                    
                    n_x_o_counter = 1
                    while i + n_x_o_counter < self.n_x_o and full_input_condition[i + n_x_o_counter] == 1:
                        n_x_o_counter = n_x_o_counter + 1

                    selected_state_idx_mat.append([j_idx for j_idx in range(0, self.n_x_b)])
                    n_x_o_mat.append(n_x_o_counter)
                    i = i + n_x_o_counter - 1
                
                i = i + 1
            
            for i in range(len(n_x_o_mat)):
                selected_state_idx = []

                selected_state_idx = selected_state_idx_mat[i]
                n_x_selected = len(selected_state_idx)

                subnet = nn.Sequential()
                subnet.add_module('Input Layer', nn.Sequential(nn.Linear(n_x_selected, self.n_feat), nn.Tanh()))

                for l in range(self.num_hidden_layers - 1):
                    subnet.add_module('Deep Layer ' + str(l+1), nn.Linear(self.n_feat, self.n_feat))
                    subnet.add_module('Activation Function for Hidden Layer ' + str(l+1), nn.Tanh())

                subnet.add_module('Output Layer ', nn.Linear(self.n_feat, n_x_o_mat[i]))
                
                for m in subnet.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0, std=1e-8)
                        nn.init.constant_(m.bias, val=0)

                setattr(self, 'subnet_' + str(i + 1), subnet)
                self.net_list.append(subnet)

            self.n_x_o_mat = n_x_o_mat
            self.selected_state_idx_mat = selected_state_idx_mat
            self.device = device

        def forward(self, x_b):

            B, _ = x_b.shape
            y = torch.empty((B, self.n_x_o)).to(self.device)

            for i in range(len(self.n_x_o_mat)):
                
                selected_state_idx = self.selected_state_idx_mat[i]        
                z = x_b[:, selected_state_idx]
                y[:, i:i+self.n_x_o_mat[i]] = self.net_list[i](z)

            return y
# %%
class PhysicalStateUpdate(nn.Module):

    def __init__(self, f_physical, device):
        super(PhysicalStateUpdate, self).__init__()
        self.device = device
        self.f_physical = f_physical
    
    def move_to_device(self, device):
        self.device = device

    def forward(self, x, u, theta_f, **kwargs):
        dx = self.f_physical(x, u, theta_f=theta_f)
        return dx


class PhysicalOutput(nn.Module):

    def __init__(self, h_physical, device):
        super(PhysicalOutput, self).__init__()
        self.device = device
        self.h_physical = h_physical
    
    def move_to_device(self, device):
        self.device = device
                    
    def forward(self, x, theta_h, **kwargs):
        y = self.h_physical(x, theta_h=theta_h)
        return y


class PhysicalStateUpdateNormalized(nn.Module):

    def __init__(self, f_physical, device):
        super(PhysicalStateUpdateNormalized, self).__init__()
        self.device = device
        self.f_physical = f_physical
        
    def move_to_device(self, device):
        self.device = device

    def forward(self, x_norm, u_norm, theta_f, datascaler):
        x_denorm = datascaler.zscore_denormalize(x_norm, datascaler.x_mean, datascaler.x_std)
        u_denorm = datascaler.zscore_denormalize(u_norm, datascaler.u_mean, datascaler.u_std)
        dx_norm = self.f_physical(x_denorm, u_denorm, theta_f=theta_f).to(self.device)/datascaler.x_std
        return dx_norm


class PhysicalOutputNormalized(nn.Module):

    def __init__(self, h_physical, device):
        super(PhysicalOutputNormalized, self).__init__()
        self.device = device
        self.h_physical = h_physical
    
    def move_to_device(self, device):
        self.device = device
                    
    def forward(self, x_norm, theta_h, datascaler):
        x_denorm = datascaler.zscore_denormalize(x_norm, datascaler.x_mean, datascaler.x_std)
        y_denorm = self.h_physical(x_denorm, theta_h=theta_h).to(self.device)
        y_norm = datascaler.zscore_normalize(y_denorm, datascaler.y_mean, datascaler.y_std)
        return y_norm

# %% [markdown]
# #### Simulator

# %%
class StateSpaceSimulator(nn.Module):
    def __init__(self, f, h, device):
        super().__init__()
        self.f = f.to(device)
        self.h = h.to(device)
        self.device = device
    
    def move_to_device(self, device):
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.device = device

    def forward(self, x_0, u, theta_f, theta_h, min_cons_hard_x_step, max_cons_hard_x_step, min_cons_hard_y_step, max_cons_hard_y_step):
        x_step = x_0
        y_step = self.h(x_0, theta_h=theta_h)

        B, n_x = x_0.shape
        _, T, _ = u.shape # B, T, n_u
        _, n_y = y_step.shape

        x = torch.empty((B, T, n_x)).to(self.device)
        y = torch.empty((B, T, n_y)).to(self.device)

        # Euler Method for solving the Discrete Equations

        for t in range(T): 
            x[:, t, :] = x_step
            y[:, t, :] = y_step
            dx = self.f(x_step, u[:, t, :], theta_f=theta_f)
            x_step = x_step + dx
            y_step = self.h(x_step, theta_h=theta_h)

            # Imposing Constraints
            x_step = hard_constraint_projection(x_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            
        return x, y
    
# %%
class StateSpaceSimulatorAugmented_Non_normalized(nn.Module):

    def __init__(self, f, f_b, g_b, h, h_b, device):

        super().__init__()
        self.f = f
        self.f_b = f_b
        self.g_b = g_b
        self.h = h
        self.h_b = h_b
        self.device = device

    def move_to_device(self, device):
    
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.f_b = self.f_b.to(device)
        self.g_b = self.g_b.to(device)
        self.h_b = self.h_b.to(device)
        
        self.f.move_to_device(device)
        self.h.move_to_device(device)
        self.f_b.move_to_device(device)
        self.g_b.move_to_device(device)
        self.h_b.move_to_device(device)
        self.to(device=device)
        self.device = device
    
    def forward(self, x_f_0, x_b_0, u, theta_f, theta_h, datascaler,
                min_cons_hard_x_b_step, max_cons_hard_x_b_step,
                min_cons_hard_x_step, max_cons_hard_x_step,
                min_cons_hard_y_step, max_cons_hard_y_step):
        
        self.datascaler = datascaler
        x_f_step = x_f_0
        x_b_step = x_b_0
        
        # Denorming -------------------------------------------------------------------------------------------------------------
        u = datascaler.zscore_denormalize(u, self.datascaler.u_mean, self.datascaler.u_std)
        x_f_step = datascaler.zscore_denormalize(x_f_step, self.datascaler.x_mean, self.datascaler.x_std)
        min_cons_hard_x_step = datascaler.zscore_denormalize(min_cons_hard_x_step, self.datascaler.x_mean, self.datascaler.x_std)
        max_cons_hard_x_step = datascaler.zscore_denormalize(max_cons_hard_x_step, self.datascaler.x_mean, self.datascaler.x_std)
        min_cons_hard_y_step = datascaler.zscore_denormalize(min_cons_hard_y_step, self.datascaler.y_mean, self.datascaler.y_std)
        max_cons_hard_y_step = datascaler.zscore_denormalize(max_cons_hard_y_step, self.datascaler.y_mean, self.datascaler.y_std)
        # -----------------------------------------------------------------------------------------------------------------------
        
        y_step = self.h(x_f_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

        B, n_x_f = x_f_step.shape
        _, n_x_b = x_b_step.shape
        _, n_y = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_f = torch.empty((B, T, n_x_f)).to(self.device)
        x_b = torch.empty((B, T, n_x_b)).to(self.device)
        y = torch.empty((B, T, n_y)).to(self.device)

        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step = u[:, t, :]
            x_f[:, t, :] = x_f_step
            x_b[:, t, :] = x_b_step
            y[:, t, :] = y_step

            dx_b = self.g_b(x_f_step, x_b_step, u_step)
            dx_f = self.f(x_f_step, u_step, theta_f=theta_f, datascaler=datascaler) + self.f_b(x_f_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_b_step = x_b_step + dx_b
            y_step = self.h(x_f_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=min_cons_hard_x_b_step, max_cons_hard=max_cons_hard_x_b_step)
            
        # Norming --------------------------------------------------------------------------------
        x_f_norm = datascaler.zscore_normalize(x_f, self.datascaler.x_mean, self.datascaler.x_std)
        x_b_norm = x_b
        y_norm = datascaler.zscore_normalize(y, self.datascaler.y_mean, self.datascaler.y_std)
        # ------ ---------------------------------------------------------------------------------
          
        return x_f_norm, x_b_norm, y_norm


class RobustStateSpaceSimulatorAugmented_Non_normalized(nn.Module):

    def __init__(self, f, f_b, g_b, h, h_b, device):

        super().__init__()
        self.f = f
        self.f_b = f_b
        self.g_b = g_b
        self.h = h
        self.h_b = h_b
        self.device = device
    
    def move_to_device(self, device):
        
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.f_b = self.f_b.to(device)
        self.g_b = self.g_b.to(device)
        self.h_b = self.h_b.to(device)
        
        self.f.move_to_device(device)
        self.h.move_to_device(device)
        self.f_b.move_to_device(device)
        self.g_b.move_to_device(device)
        self.h_b.move_to_device(device)
        self.to(device=device)
        self.device = device

    def forward(self, x_f_0, x_b_0, u, theta_f, theta_h, datascaler,
                min_cons_hard_x_b_step, max_cons_hard_x_b_step,
                min_cons_hard_x_step, max_cons_hard_x_step,
                min_cons_hard_y_step, max_cons_hard_y_step):

        self.datascaler = datascaler
        x_f_step = x_f_0
        x_p_step = x_f_0
        x_b_step = x_b_0
        
        # Denorming -------------------------------------------------------------------------------------------------------------
        u = datascaler.zscore_denormalize(u, self.datascaler.u_mean, self.datascaler.u_std)
        x_f_step = datascaler.zscore_denormalize(x_f_step, self.datascaler.x_mean, self.datascaler.x_std)
        x_p_step = x_f_step
        min_cons_hard_x_step = datascaler.zscore_denormalize(min_cons_hard_x_step, self.datascaler.x_mean, self.datascaler.x_std)
        max_cons_hard_x_step = datascaler.zscore_denormalize(max_cons_hard_x_step, self.datascaler.x_mean, self.datascaler.x_std)
        min_cons_hard_y_step = datascaler.zscore_denormalize(min_cons_hard_y_step, self.datascaler.y_mean, self.datascaler.y_std)
        max_cons_hard_y_step = datascaler.zscore_denormalize(max_cons_hard_y_step, self.datascaler.y_mean, self.datascaler.y_std)
        # -----------------------------------------------------------------------------------------------------------------------
        
        y_step = self.h(x_p_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

        B, n_x_f = x_f_step.shape
        _, n_x_b = x_b_step.shape
        _, n_y   = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_f = torch.empty((B, T, n_x_f)).to(self.device)
        x_p = torch.empty((B, T, n_x_f)).to(self.device)
        x_b = torch.empty((B, T, n_x_b)).to(self.device)
        y   = torch.empty((B, T, n_y)).to(self.device)

        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step       = u[:, t, :]
            x_f[:, t, :] = x_f_step
            x_p[:, t, :] = x_p_step
            x_b[:, t, :] = x_b_step
            y[:, t, :]   = y_step

            dx_b = self.g_b(x_p_step, x_b_step, u_step)
            dx_p = self.f(x_p_step, u_step, theta_f=theta_f, datascaler=datascaler)
            dx_f = dx_p + self.f_b(x_p_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_p_step = x_p_step + dx_p
            # x_b_step = x_b_step + dx_b
            x_b_step = dx_b
            y_step   = self.h(x_p_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            x_p_step = hard_constraint_projection(x_p_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step   = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=min_cons_hard_x_b_step, max_cons_hard=max_cons_hard_x_b_step)

        # Norming --------------------------------------------------------------------------------
        x_f_norm = datascaler.zscore_normalize(x_f, self.datascaler.x_mean, self.datascaler.x_std)
        x_p_norm = datascaler.zscore_normalize(x_f, self.datascaler.x_mean, self.datascaler.x_std)
        x_b_norm = x_b
        y_norm = datascaler.zscore_normalize(y, self.datascaler.y_mean, self.datascaler.y_std)
        # ----------------------------------------------------------------------------------------
        
        return x_f_norm, x_b_norm, y_norm, x_p_norm
    
    
# %%
class StateSpaceSimulatorAugmented(nn.Module):

    def __init__(self, f, f_b, g_b, h, h_b, device):

        super().__init__()
        self.f = f
        self.f_b = f_b
        self.g_b = g_b
        self.h = h
        self.h_b = h_b
        self.device = device

    def move_to_device(self, device):
    
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.f_b = self.f_b.to(device)
        self.g_b = self.g_b.to(device)
        self.h_b = self.h_b.to(device)
        
        self.f.move_to_device(device)
        self.h.move_to_device(device)
        self.f_b.move_to_device(device)
        self.g_b.move_to_device(device)
        self.h_b.move_to_device(device)
        self.to(device=device)
        self.device = device
    
    def forward(self, x_f_0, x_b_0, u, theta_f, theta_h, datascaler,
                min_cons_hard_x_b_step, max_cons_hard_x_b_step,
                min_cons_hard_x_step, max_cons_hard_x_step,
                min_cons_hard_y_step, max_cons_hard_y_step):

        x_f_step = x_f_0
        x_b_step = x_b_0

        y_step = self.h(x_f_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

        B, n_x_f = x_f_step.shape
        _, n_x_b = x_b_step.shape
        _, n_y = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_f = torch.empty((B, T, n_x_f)).to(self.device)
        x_b = torch.empty((B, T, n_x_b)).to(self.device)
        y = torch.empty((B, T, n_y)).to(self.device)

        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step = u[:, t, :]
            x_f[:, t, :] = x_f_step
            x_b[:, t, :] = x_b_step
            y[:, t, :] = y_step

            dx_b = self.g_b(x_f_step, x_b_step, u_step)
            dx_f = self.f(x_f_step, u_step, theta_f=theta_f, datascaler=datascaler) + self.f_b(x_f_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_b_step = x_b_step + dx_b
            y_step = self.h(x_f_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=min_cons_hard_x_b_step, max_cons_hard=max_cons_hard_x_b_step)
            
        return x_f, x_b, y
    
# %%
class RobustStateSpaceSimulatorAugmented(nn.Module):

    def __init__(self, f, f_b, g_b, h, h_b, device):

        super().__init__()
        self.f = f
        self.f_b = f_b
        self.g_b = g_b
        self.h = h
        self.h_b = h_b
        self.device = device
    
    def move_to_device(self, device):
        
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.f_b = self.f_b.to(device)
        self.g_b = self.g_b.to(device)
        self.h_b = self.h_b.to(device)
        
        self.f.move_to_device(device)
        self.h.move_to_device(device)
        self.f_b.move_to_device(device)
        self.g_b.move_to_device(device)
        self.h_b.move_to_device(device)
        self.to(device=device)
        self.device = device

    def forward(self, x_f_0, x_b_0, u, theta_f, theta_h, datascaler,
                min_cons_hard_x_b_step, max_cons_hard_x_b_step,
                min_cons_hard_x_step, max_cons_hard_x_step,
                min_cons_hard_y_step, max_cons_hard_y_step):

        x_f_step = x_f_0
        x_p_step = x_f_0
        x_b_step = x_b_0
        y_step = self.h(x_p_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

        B, n_x_f = x_f_step.shape
        _, n_x_b = x_b_step.shape
        _, n_y   = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_f = torch.empty((B, T, n_x_f)).to(self.device)
        x_p = torch.empty((B, T, n_x_f)).to(self.device)
        x_b = torch.empty((B, T, n_x_b)).to(self.device)
        y   = torch.empty((B, T, n_y)).to(self.device)

        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step       = u[:, t, :]
            x_f[:, t, :] = x_f_step
            x_p[:, t, :] = x_p_step
            x_b[:, t, :] = x_b_step
            y[:, t, :]   = y_step

            dx_b = self.g_b(x_p_step, x_b_step, u_step)
            dx_p = self.f(x_p_step, u_step, theta_f=theta_f, datascaler=datascaler)
            dx_f = dx_p + self.f_b(x_p_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_p_step = x_p_step + dx_p
            # x_b_step = x_b_step + dx_b
            x_b_step = dx_b
            y_step   = self.h(x_p_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            x_p_step = hard_constraint_projection(x_p_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step   = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=min_cons_hard_x_b_step, max_cons_hard=max_cons_hard_x_b_step)

        return x_f, x_b, y, x_p


# %%
class StateSpaceSimulatorAugmented_MANAM(nn.Module):

    def __init__(self, f, f_b, g_b, h, h_b, device, gamma_power=1, display_gamma_condition_stat=False):

        super().__init__()
        self.device = device
        self.f = f.to(self.device)
        self.f_b = f_b.to(self.device)
        self.g_b = g_b.to(self.device)
        self.h = h.to(self.device)
        self.h_b = h_b.to(self.device)
        self.gamma_power = gamma_power
        self.display_gamma_condition_stat = display_gamma_condition_stat
    
    def move_to_device(self, device):
        
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.f_b = self.f_b.to(device)
        self.g_b = self.g_b.to(device)
        self.h_b = self.h_b.to(device)
        
        self.f.move_to_device(device)
        self.h.move_to_device(device)
        self.f_b.move_to_device(device)
        self.g_b.move_to_device(device)
        self.h_b.move_to_device(device)
        self.to(device=device)
        self.device = device

    def forward(self, x_f_0, x_b_0, u, theta_f, theta_h, datascaler,
                min_cons_hard_x_b_step, max_cons_hard_x_b_step,
                min_cons_hard_x_step, max_cons_hard_x_step,
                min_cons_hard_y_step, max_cons_hard_y_step,
                adaptive_gamma_eval=True):

        x_f_step = x_f_0
        x_b_step = x_b_0
        y_step = self.h(x_f_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)
        
        mu_x_b_step = x_b_0
        sigma_x_b_step = torch.ones_like(x_b_0)

        B, n_x_f = x_f_step.shape
        _, n_x_b = x_b_step.shape
        _, n_y = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_f = torch.empty((B, T, n_x_f)).to(self.device)
        x_b = torch.empty((B, T, n_x_b)).to(self.device)
        y = torch.empty((B, T, n_y)).to(self.device)

        D_satisfaction = 0
        D_1_satisfaction = 0
        D_2_satisfaction = 0
        D_3_satisfaction = 0
        D_4_satisfaction = 0
        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step = u[:, t, :]
            x_f[:, t, :] = x_f_step
            x_b[:, t, :] = x_b_step
            y[:, t, :] = y_step

            gamma_x_b_step_plus  = sigma_x_b_step**self.gamma_power
            gamma_x_b_step_minus = torch.ones_like(sigma_x_b_step)

            g_b_val  = self.g_b(x_f_step, x_b_step, u_step)
            delta_1  = g_b_val*(x_b_step - mu_x_b_step + (g_b_val/2)*(1 + (1/gamma_x_b_step_plus)))
            delta_2  = (1 - (1/gamma_x_b_step_plus))
            delta    = delta_1*delta_2
            epsilon = (t+2)*(sigma_x_b_step**2 - ((t+2)/(t+1))) + (x_b_step - mu_x_b_step)*(x_b_step - mu_x_b_step + (g_b_val)*(1 + (1/gamma_x_b_step_plus)) + (g_b_val**2/2)*(1 + (1/gamma_x_b_step_plus**2)))
            
            gamma_x_b_step = gamma_x_b_step_minus
            
            if adaptive_gamma_eval:
                for i in range(n_x_b):
                    for j in range(B):
                        
                        D_1 = (epsilon[j, i] >= delta[j, i] >= 0)
                        D_2 = (epsilon[j, i] < delta[j, i] < 0)
                        D_3 = (delta[j, i] < epsilon[j, i] < 0)
                        D_4 = (((2*(t+2)**2)/(t+1))*epsilon[j, i] >= delta[j, i]**2 - epsilon[j, i]**2 > 0)

                        if D_1 and self.display_gamma_condition_stat:
                            D_1_satisfaction = D_1_satisfaction + 1
                            
                        if D_2 and self.display_gamma_condition_stat:
                            D_2_satisfaction = D_2_satisfaction + 1
                            
                        if D_3 and self.display_gamma_condition_stat:
                            D_3_satisfaction = D_3_satisfaction + 1
                            
                        if D_4 and self.display_gamma_condition_stat:
                            D_4_satisfaction = D_4_satisfaction + 1
                        
                        if D_1 or D_2 or D_3 or D_4:
                            gamma_x_b_step[j, i] = gamma_x_b_step_plus[j, i]
                            D_satisfaction = D_satisfaction + 1
                        
            
            dx_f = self.f(x_f_step, u_step, theta_f=theta_f, datascaler=datascaler) + self.f_b(x_f_step, x_b_step, u_step)
            dx_b = g_b_val/gamma_x_b_step

            x_f_step = x_f_step + dx_f
            x_b_step = x_b_step + dx_b
            y_step = self.h(x_f_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=min_cons_hard_x_b_step, max_cons_hard=max_cons_hard_x_b_step)    

            sigma_x_b_step = torch.sqrt(((t+1)/(t+2)) * sigma_x_b_step**2 + ((t+1)/(t+2)**2) * (x_b_step - mu_x_b_step)**2)
            mu_x_b_step = ((t+1)/(t+2)) * mu_x_b_step + ((1)/(t+2)) * x_b_step
            
        if self.display_gamma_condition_stat:
            print(f"D_satisfaction Ratio = {(D_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_1_satisfaction Ratio = {(D_1_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_2_satisfaction Ratio = {(D_2_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_3_satisfaction Ratio = {(D_3_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_4_satisfaction Ratio = {(D_4_satisfaction/(T * n_x_b * B))*100:.2f}%")

        return x_f, x_b, y
    
    
# %%
class RobustStateSpaceSimulatorAugmented_MANAM(nn.Module):

    def __init__(self, f, f_b, g_b, h, h_b, device, gamma_power=1, display_gamma_condition_stat=False):

        super().__init__()
        self.device = device
        self.f = f.to(self.device)
        self.f_b = f_b.to(self.device)
        self.g_b = g_b.to(self.device)
        self.h = h.to(self.device)
        self.h_b = h_b.to(self.device)
        self.gamma_power = gamma_power
        self.display_gamma_condition_stat = display_gamma_condition_stat
    
    def move_to_device(self, device):
        
        self.f = self.f.to(device)
        self.h = self.h.to(device)
        self.f_b = self.f_b.to(device)
        self.g_b = self.g_b.to(device)
        self.h_b = self.h_b.to(device)
        
        self.f.move_to_device(device)
        self.h.move_to_device(device)
        self.f_b.move_to_device(device)
        self.g_b.move_to_device(device)
        self.h_b.move_to_device(device)
        self.to(device=device)
        self.device = device

    def forward(self, x_f_0, x_b_0, u, theta_f, theta_h, datascaler,
                min_cons_hard_x_b_step, max_cons_hard_x_b_step,
                min_cons_hard_x_step, max_cons_hard_x_step,
                min_cons_hard_y_step, max_cons_hard_y_step,
                adaptive_gamma_eval=True):

        x_f_step = x_f_0
        x_p_step = x_f_0
        x_b_step = x_b_0
        y_step = self.h(x_p_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

        mu_x_b_step = torch.zeros_like(x_b_0)
        sigma_x_b_step = torch.ones_like(x_b_0)

        B, n_x_f = x_f_step.shape
        _, n_x_b = x_b_step.shape
        _, n_y   = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_f = torch.empty((B, T, n_x_f)).to(self.device)
        x_p = torch.empty((B, T, n_x_f)).to(self.device)
        x_b = torch.empty((B, T, n_x_b)).to(self.device)
        y   = torch.empty((B, T, n_y)).to(self.device)

        D_satisfaction = 0
        D_1_satisfaction = 0
        D_2_satisfaction = 0
        D_3_satisfaction = 0
        D_4_satisfaction = 0
        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step       = u[:, t, :]
            x_f[:, t, :] = x_f_step
            x_p[:, t, :] = x_p_step
            x_b[:, t, :] = x_b_step
            y[:, t, :]   = y_step
            
            gamma_x_b_step_plus  = sigma_x_b_step**self.gamma_power
            gamma_x_b_step_minus = torch.ones_like(sigma_x_b_step)
            g_b_val = self.g_b(x_p_step, x_b_step, u_step)
            
            delta_1  = g_b_val*(x_b_step - mu_x_b_step + (g_b_val/2)*(1 + (1/gamma_x_b_step_plus)))
            delta_2  = (1 - (1/gamma_x_b_step_plus))
            delta    = delta_1*delta_2
            epsilon = (t+2)*(sigma_x_b_step**2 - ((t+2)/(t+1))) + (x_b_step - mu_x_b_step)*(x_b_step - mu_x_b_step + (g_b_val)*(1 + (1/gamma_x_b_step_plus)) + (g_b_val**2/2)*(1 + (1/gamma_x_b_step_plus**2)))
            
            gamma_x_b_step = gamma_x_b_step_minus
            if adaptive_gamma_eval:
                for i in range(n_x_b):
                    for j in range(B):
                        
                        D_1 = (epsilon[j, i] >= delta[j, i] >= 0)
                        D_2 = (epsilon[j, i] < delta[j, i] < 0)
                        D_3 = (delta[j, i] < epsilon[j, i] < 0)
                        D_4 = (((2*(t+2)**2)/(t+1))*epsilon[j, i] >= delta[j, i]**2 - epsilon[j, i]**2 > 0)

                        if D_1 and self.display_gamma_condition_stat:
                            D_1_satisfaction = D_1_satisfaction + 1
                            
                        if D_2 and self.display_gamma_condition_stat:
                            D_2_satisfaction = D_2_satisfaction + 1
                            
                        if D_3 and self.display_gamma_condition_stat:
                            D_3_satisfaction = D_3_satisfaction + 1
                            
                        if D_4 and self.display_gamma_condition_stat:
                            D_4_satisfaction = D_4_satisfaction + 1
                        
                        if D_1 or D_2 or D_3 or D_4:
                            gamma_x_b_step[j, i] = gamma_x_b_step_plus[j, i]
                            D_satisfaction = D_satisfaction + 1

            dx_b = g_b_val/gamma_x_b_step
            dx_p = self.f(x_p_step, u_step, theta_f=theta_f, datascaler=datascaler)
            dx_f = dx_p + self.f_b(x_p_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_p_step = x_p_step + dx_p
            # x_b_step = x_b_step + dx_b
            x_b_step = dx_b
            y_step   = self.h(x_p_step, theta_h=theta_h, datascaler=datascaler) + self.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            x_p_step = hard_constraint_projection(x_p_step, min_cons_hard=min_cons_hard_x_step, max_cons_hard=max_cons_hard_x_step)
            y_step   = hard_constraint_projection(y_step, min_cons_hard=min_cons_hard_y_step, max_cons_hard=max_cons_hard_y_step)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=min_cons_hard_x_b_step, max_cons_hard=max_cons_hard_x_b_step)
            
            mu_x_b_step = ((t+1)/(t+2)) * mu_x_b_step + ((1)/(t+2)) * x_b_step
            sigma_x_b_step = torch.sqrt(((t+1)/(t+2)) * sigma_x_b_step**2 + ((1)/(t+2)) * (x_b_step - mu_x_b_step)**2)
        
        if self.display_gamma_condition_stat:
            print(f"D_satisfaction Ratio = {(D_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_1_satisfaction Ratio = {(D_1_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_2_satisfaction Ratio = {(D_2_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_3_satisfaction Ratio = {(D_3_satisfaction/(T * n_x_b * B))*100:.2f}%")
            print(f"D_4_satisfaction Ratio = {(D_4_satisfaction/(T * n_x_b * B))*100:.2f}%")
        
        return x_f, x_b, y, x_p


# %%
class AugModel(nn.Module):

    def __init__(self,
                simulator,
                x_f_0,
                x_b_0,
                theta_f,
                theta_h,
                datascaler,
                min_cons_hard_x_b_step,
                max_cons_hard_x_b_step,
                min_cons_hard_x_step_norm,
                max_cons_hard_x_step_norm,
                min_cons_hard_y_step_norm,
                max_cons_hard_y_step_norm,
                augmentation):
        
        super().__init__()
        self.simulator = simulator
        self.x_f_0 = x_f_0
        self.x_b_0 = x_b_0
        self.theta_f = theta_f
        self.theta_h = theta_h
        self.datascaler = datascaler
        self.min_cons_hard_x_b_step = min_cons_hard_x_b_step
        self.max_cons_hard_x_b_step = max_cons_hard_x_b_step
        self.min_cons_hard_x_step_norm = min_cons_hard_x_step_norm
        self.max_cons_hard_x_step_norm = max_cons_hard_x_step_norm
        self.min_cons_hard_y_step_norm = min_cons_hard_y_step_norm
        self.max_cons_hard_y_step_norm = max_cons_hard_y_step_norm
        self.augmentation = augmentation

    def forward(self, u, domain="denorm"):

        if domain == "denorm":
            u = self.datascaler.zscore_normalize(u, self.datascaler.u_mean, self.datascaler.u_std)

        if self.augmentation == "Exact" or self.augmentation == "Exact - Uniform":
            x_f, x_b, y = self.simulator(x_f_0=self.x_f_0,
                                        x_b_0=self.x_b_0,
                                        u=u,
                                        theta_f=self.theta_f,
                                        theta_h=self.theta_h,
                                        datascaler=self.datascaler,
                                        min_cons_hard_x_b_step=self.min_cons_hard_x_b_step,
                                        max_cons_hard_x_b_step=self.max_cons_hard_x_b_step,
                                        min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                        max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                        min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                        max_cons_hard_y_step=self.max_cons_hard_y_step_norm)
            x_p = x_f
            
        elif self.augmentation == "Robust" or self.augmentation == "Robust - Uniform":
            x_f, x_b, y, x_p = self.simulator(x_f_0=self.x_f_0,
                                            x_b_0=self.x_b_0,
                                            u=u,
                                            theta_f=self.theta_f,
                                            theta_h=self.theta_h,
                                            datascaler=self.datascaler,
                                            min_cons_hard_x_b_step=self.min_cons_hard_x_b_step,
                                            max_cons_hard_x_b_step=self.max_cons_hard_x_b_step,
                                            min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                            max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                            min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                            max_cons_hard_y_step=self.max_cons_hard_y_step_norm)

        if domain == "denorm":
            x_f = self.datascaler.zscore_denormalize(x_f, self.datascaler.x_mean, self.datascaler.x_std)
            x_p = self.datascaler.zscore_denormalize(x_p, self.datascaler.x_mean, self.datascaler.x_std)
            y = self.datascaler.zscore_denormalize(y, self.datascaler.y_mean, self.datascaler.y_std)
        
        return x_f, x_p, x_b, y
    
    def step_update(self, x_f_step, x_p_step, x_b_step, u_step, domain="norm"):

        if domain == "denorm":
            u_step = self.datascaler.zscore_normalize(u_step, self.datascaler.u_mean, self.datascaler.u_std)
            x_f_step = self.datascaler.zscore_normalize(x_f_step, self.datascaler.x_mean, self.datascaler.x_std)
            x_p_step = self.datascaler.zscore_normalize(x_p_step, self.datascaler.x_mean, self.datascaler.x_std)

        if self.augmentation == "Exact":

            dx_b = self.simulator.g_b(x_f_step, x_b_step, u_step)
            dx_f = self.simulator.f(x_f_step, u_step, theta_f=self.theta_f, datascaler=self.datascaler) + self.simulator.f_b(x_f_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_b_step = x_b_step + dx_b
            y_step = self.simulator.h(x_f_step, theta_h=self.theta_h, datascaler=self.datascaler) + self.simulator.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=self.min_cons_hard_x_step_norm, max_cons_hard=self.max_cons_hard_x_step_norm)
            y_step = hard_constraint_projection(y_step, min_cons_hard=self.min_cons_hard_y_step_norm, max_cons_hard=self.max_cons_hard_y_step_norm)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=self.min_cons_hard_x_b_step, max_cons_hard=self.max_cons_hard_x_b_step)
            x_p_step = x_f_step
        
        elif self.augmentation == "Robust":

            dx_b = self.simulator.g_b(x_p_step, x_b_step, u_step)
            dx_p = self.simulator.f(x_p_step, u_step, theta_f=self.theta_f, datascaler=self.datascaler)
            dx_f = dx_p + self.simulator.f_b(x_p_step, x_b_step, u_step)
            
            x_f_step = x_f_step + dx_f
            x_p_step = x_p_step + dx_p
            x_b_step = x_b_step + dx_b
            y_step   = self.simulator.h(x_p_step, theta_h=self.theta_h, datascaler=self.datascaler) + self.simulator.h_b(x_f_step, x_b_step)

            # Imposing Constraints
            x_f_step = hard_constraint_projection(x_f_step, min_cons_hard=self.min_cons_hard_x_step_norm, max_cons_hard=self.max_cons_hard_x_step_norm)
            x_p_step = hard_constraint_projection(x_p_step, min_cons_hard=self.min_cons_hard_x_step_norm, max_cons_hard=self.max_cons_hard_x_step_norm)
            y_step   = hard_constraint_projection(y_step, min_cons_hard=self.min_cons_hard_y_step_norm, max_cons_hard=self.max_cons_hard_y_step_norm)
            x_b_step = hard_constraint_projection(x_b_step, min_cons_hard=self.min_cons_hard_x_b_step, max_cons_hard=self.max_cons_hard_x_b_step)

        if domain == "denorm":
            x_f_step = self.datascaler.zscore_denormalize(x_f_step, self.datascaler.x_mean, self.datascaler.x_std)
            x_p_step = self.datascaler.zscore_denormalize(x_p_step, self.datascaler.x_mean, self.datascaler.x_std)
            y_step = self.datascaler.zscore_denormalize(y_step, self.datascaler.y_mean, self.datascaler.y_std)

        return x_f_step, x_p_step, x_b_step, y_step


# %%
class StateSpaceSimulatorNeural(nn.Module):

    def __init__(self, g_b, h_b):

        super().__init__()
        self.g_b = g_b
        self.h_b = h_b

    def forward(self, x_b_0, u):

        x_b_step = x_b_0
        y_step = self.h_b(x_b_step)

        B, n_x_b = x_b_step.shape
        _, n_y = y_step.shape
        _, T, _ = u.shape # B, T, n_u

        x_b = torch.empty((B, T, n_x_b))
        y = torch.empty((B, T, n_y))

        # Euler Method for solving the Discrete Equations
        for t in range(T): 

            u_step = u[:, t, :]
            x_b[:, t, :] = x_b_step
            y[:, t, :] = y_step

            dx_b = self.g_b(x_b_step, u_step)
            x_b_step = x_b_step + dx_b
            y_step = self.h_b(x_b_step)
            
        return x_b, y
    
# %%       
# Mini-Batch Manipulation
class MiniBatchDataset(Dataset):
    def __init__(self, u, y):
        self.u = u
        self.y = y

    def __len__(self):
        return self.y.shape[1]

    def __getitem__(self, idx):
        return (self.u[:, idx, :], self.y[:, idx, :])

class SubsequenceDataset(Dataset):
    def __init__(self, features, targets, subseq_len, stride=1):
        self.features = features
        self.targets = targets
        self.subseq_len = subseq_len
        self.length = self.targets.shape[0]
        # self.width = self.targets.shape[1]
        self.stride = stride
        self.num_subseqs = (self.length - self.subseq_len) // self.stride + 1

    def __len__(self):
        return self.num_subseqs

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.subseq_len
        subsequences_feature = self.features[start_idx:end_idx]
        subsequences_target = self.targets[start_idx:end_idx]
        return subsequences_feature, subsequences_target



class PhysicalStateEstimator(nn.Module):
    
    def __init__(self, x_f_0_phys, n_x_f, n_x_b, batch_size):
        super(PhysicalStateEstimator, self).__init__()
        self.x_f_0_phys = x_f_0_phys
        self.batch_size = batch_size
        self.n_x_f = n_x_f
        self.n_x_b = n_x_b
        
    def forward(self, u, y, mean=0, std=0):
        x_0_phys_norm = (self.x_f_0_phys - mean[:, : self.n_x_f])/std[:, : self.n_x_f]
        x_est = x_0_phys_norm.repeat(self.batch_size, 1)
        
        return x_est
