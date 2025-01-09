import torch
from torch import nn as nn
import os
import pandas as pd
import pickle as pk
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
import matplotlib.lines as mlines
import numpy as np
#========================================================================================================================================================================================
class PhysicalStateUpdate(nn.Module):

    def __init__(self, f_physical):
        super(PhysicalStateUpdate, self).__init__()
        self.f_physical = f_physical

    def forward(self, x, u, theta_f, tau_u, tau_y):
        dx = self.f_physical(x, u, theta_f=theta_f, tau_u=tau_u, tau_y=tau_y)
        return dx


class PhysicalOutput(nn.Module):

    def __init__(self, h_physical):
        super(PhysicalOutput, self).__init__()
        self.h_physical = h_physical
                    
    def forward(self, x, theta_h):
        y = self.h_physical(x, theta_h=theta_h)
        return y
    

class StateSpaceSimulator(nn.Module):
    def __init__(self, f, h, min_cons_hard_x_step, max_cons_hard_x_step, min_cons_hard_y_step, max_cons_hard_y_step):
        super().__init__()
        self.f = f
        self.h = h
        self.min_cons_hard_x_step = min_cons_hard_x_step
        self.max_cons_hard_x_step = max_cons_hard_x_step
        self.min_cons_hard_y_step = min_cons_hard_y_step
        self.max_cons_hard_y_step = max_cons_hard_y_step

    def forward(self, x_0, u, theta_f, theta_h, tau_u, tau_y):
        x_step = x_0
        y_step = self.h(x_0, theta_h=theta_h)

        B, n_x = x_0.shape
        _, T, _ = u.shape # B, T, n_u
        _, n_y = y_step.shape

        x = torch.empty((B, T, n_x))
        y = torch.empty((B, T, n_y))

        # Retreiving Constraints
        min_cons_hard_x_step = self.min_cons_hard_x_step
        max_cons_hard_x_step = self.max_cons_hard_x_step
        min_cons_hard_y_step = self.min_cons_hard_y_step
        max_cons_hard_y_step = self.max_cons_hard_y_step

        # Euler Method for solving the Discrete Equations

        for t in range(T): 
            x[:, t, :] = x_step
            y[:, t, :] = y_step
            dx = self.f(x_step, u[:, t, :], theta_f=theta_f, tau_u=tau_u, tau_y=tau_y)
            x_step = x_step + dx
            y_step = self.h(x_step, theta_h=theta_h)

            # Imposing Constraints
            x_step = torch.clamp(x_step, min=min_cons_hard_x_step, max=max_cons_hard_x_step)
            y_step = torch.clamp(y_step, min=min_cons_hard_y_step, max=max_cons_hard_y_step)
            
        return x, y
#========================================================================================================================================================================================
# Model (2) From Worden et al. (2018)
def f_cascaded_tanks(x_step, u_ref, theta_f, tau_u, tau_y):
    
    x_1 = x_step[:, 0:1]
    x_2 = x_step[:, 1:2]
    x_3 = x_step[:, 2:3]
    u   = x_step[:, 3:4]

    theta_f_1 = theta_f[0]
    theta_f_3 = theta_f[1]
    theta_f_4 = theta_f[2]
    theta_f_5 = theta_f[3]
    theta_f_6 = theta_f[4]

    du = (tau_u -1) * u + (1 - tau_u) * u_ref

    dx_1 = -theta_f_1 * torch.sqrt(x_1) + theta_f_5/10 * x_1 + theta_f_4 * u

    x_1_overflow = x_1 + dx_1
    for b in range(x_step.shape[0]):
        if x_1_overflow[b] <= 10:
            dx_2 = theta_f_1 * torch.sqrt(x_1) - theta_f_5/10 * x_1  + theta_f_6/10 * x_2 - theta_f_3 * torch.sqrt(x_2)
        elif x_1_overflow[b] > 10:
            dx_2 = theta_f_1 * torch.sqrt(x_1) - theta_f_5/10 * x_1  + theta_f_6/10 * x_2 - theta_f_3 * torch.sqrt(x_2) + theta_f_5 * u
    
    dx_3 = (tau_y -1) * x_3 + (1 - tau_y) * x_2

    dx = torch.cat((dx_1, dx_2, dx_3, du), 1)

    return dx

# [USER-DEFINED] physical output function (h)
def h_cascaded_tanks(x_sim, theta_h):
    y_sim = x_sim[:,2:3]

    return y_sim
#========================================================================================================================================================================================
def generate_train_test_data(f, h,
                             theta_f, theta_h, x0, tau_u, tau_y,
                             min_cons_hard_x_step, max_cons_hard_x_step,
                             min_cons_hard_y_step, max_cons_hard_y_step,
                             u_train, u_test, y_train, y_test,
                             len_branch, seed, bias,
                             clamp_ub, clamp_lb,
                             w_mass, w_scale):

    torch.manual_seed(seed)
    len_train = u_train.shape[0]
    len_test = u_test.shape[0]
    len_leaf = min(len_test, len_train)
    u_gen_branch_train = torch.empty(0,)
    u_gen_branch_test = torch.empty(0,)
    u_train_branch = torch.empty(0,)
    u_test_branch = torch.empty(0,)
    y_train_branch = torch.empty(0,)
    y_test_branch = torch.empty(0,)
    n_leaf = int(len_branch / len_leaf)
    
    r = torch.rand(n_leaf + 1)
    w_train = w_scale * (r - w_mass)
    w_test = 1 - w_train
    
    for i in range(n_leaf):
        
        weight_train = torch.clamp(w_train[i] + bias, clamp_lb, clamp_ub)
        weight_test = torch.clamp(w_test[i] - bias, clamp_lb, clamp_ub)

        u_gen_leaf_train = weight_train * u_train + (1 - weight_train) * u_test
        u_gen_branch_train = torch.cat((u_gen_branch_train, u_gen_leaf_train[:]))
        
        u_gen_leaf_test = weight_test * u_train + (1 - weight_test) * u_test
        u_gen_branch_test = torch.cat((u_gen_branch_test, u_gen_leaf_test[:]))

        u_train_branch = torch.cat((u_train_branch, u_train[:]))
        u_test_branch = torch.cat((u_test_branch, u_test[:]))

        y_train_branch = torch.cat((y_train_branch, y_train[:]))
        y_test_branch = torch.cat((y_test_branch, y_test[:]))

    weight_train = torch.clamp(w_train[-1] + bias, clamp_lb, clamp_ub)
    weight_test = torch.clamp(w_test[-1] + bias, clamp_lb, clamp_ub)
            
    u_gen_leaf_train = weight_train * u_train[0:len_branch - n_leaf*len_leaf] + (1 - weight_train) * u_test[0:len_branch - n_leaf*len_leaf]
    u_gen_branch_train = torch.cat((u_gen_branch_train, u_gen_leaf_train[:]))
    
    u_gen_leaf_test = weight_test * u_train[0:len_branch - n_leaf*len_leaf] + (1 - weight_test) * u_test[0:len_branch - n_leaf*len_leaf]
    u_gen_branch_test = torch.cat((u_gen_branch_test, u_gen_leaf_test[:]))

    u_train_branch = torch.cat((u_train_branch, u_train[:len_branch - n_leaf*len_leaf]))
    u_test_branch = torch.cat((u_test_branch, u_test[:len_branch - n_leaf*len_leaf]))

    y_train_branch = torch.cat((y_train_branch, y_train[:len_branch - n_leaf*len_leaf]))
    y_test_branch = torch.cat((y_test_branch, y_test[:len_branch - n_leaf*len_leaf]))

    u_gen_branch_train = torch.clamp(u_gen_branch_train, 0)
    u_gen_branch_test = torch.clamp(u_gen_branch_test, 0)
    
    # Creating Physical-Component Objects
    f_phys = PhysicalStateUpdate(f)
    h_phys = PhysicalOutput(h)
    # Creating The Simulator Object (Physical Model)
    simulator_phys = StateSpaceSimulator(f_phys, h_phys,
                                         min_cons_hard_x_step=min_cons_hard_x_step,
                                         max_cons_hard_x_step=max_cons_hard_x_step,
                                         min_cons_hard_y_step=min_cons_hard_y_step,
                                         max_cons_hard_y_step=max_cons_hard_y_step)

    x_pred_gen_train, y_pred_gen_train = simulator_phys(x0, u_gen_branch_train.view(1, -1, 1), theta_f=theta_f, theta_h=theta_h, tau_u=tau_u, tau_y=tau_y)
    x_pred_gen_train = torch.reshape(x_pred_gen_train[:, :, -1], (u_gen_branch_train.size(0), -1))
    y_pred_gen_train = torch.reshape(y_pred_gen_train, (u_gen_branch_train.size(0), -1))
    
    x_pred_gen_test, y_pred_gen_test = simulator_phys(x0, u_gen_branch_test.view(1, -1, 1), theta_f=theta_f, theta_h=theta_h, tau_u=tau_u, tau_y=tau_y)
    x_pred_gen_test = torch.reshape(x_pred_gen_test[:, :, -1], (u_gen_branch_test.size(0), -1))
    y_pred_gen_test = torch.reshape(y_pred_gen_test, (u_gen_branch_test.size(0), -1))
    
    anim_dict = {"y_train_branch": y_train_branch,
                 "y_test_branch": y_test_branch,
                 "u_train_branch": u_train_branch,
                 "u_test_branch": u_test_branch}

    return u_gen_branch_train, y_pred_gen_train[:, 0], u_gen_branch_test, y_pred_gen_test[:, 0], anim_dict

def generate_instance_dataset(f, h,
                              theta_f, theta_h, x0, tau_u, tau_y,
                              min_cons_hard_x_step, max_cons_hard_x_step, min_cons_hard_y_step, max_cons_hard_y_step,
                              u_train, u_test, y_train, y_test,
                              len_train_data, len_test_data, seed, bias,
                              clamp_ub, clamp_lb,
                              w_mass, w_scale):
    
    u_train_gen, y_train_gen, u_test_gen, y_test_gen, anim_dict = generate_train_test_data(f=f, h=h,
                                                                                            theta_f=theta_f, theta_h=theta_h, x0=x0, tau_u=tau_u, tau_y=tau_y,
                                                                                            min_cons_hard_x_step=min_cons_hard_x_step,
                                                                                            max_cons_hard_x_step=max_cons_hard_x_step,
                                                                                            min_cons_hard_y_step=min_cons_hard_y_step,
                                                                                            max_cons_hard_y_step=max_cons_hard_y_step,
                                                                                            u_train=u_train, u_test=u_test, y_train=y_train, y_test=y_test,
                                                                                            len_branch=len_train_data, seed=seed, bias=bias,
                                                                                            clamp_ub=clamp_ub,
                                                                                            clamp_lb=clamp_lb,
                                                                                            w_mass=w_mass, w_scale=w_scale)
    
    
    dataset_gen_dict = {"u_train": u_train_gen, "u_test": u_test_gen, "y_train": y_train_gen, "y_test": y_test_gen}
    
    return dataset_gen_dict, anim_dict, anim_dict

def generate_multi_dataset(f, h,
                           theta_f, theta_h, x0, tau_u_list, tau_y_list,
                           min_cons_hard_x_step, max_cons_hard_x_step, min_cons_hard_y_step, max_cons_hard_y_step,
                           u_train, u_test, y_train, y_test,
                           len_train_data, len_test_data, seed_list, bias_list,
                           clamp_ub, clamp_lb,
                           w_mass, w_scale, sample_plot_idx=0):
    
    multidataset_gen_dict = {}
    anim_dict_list_train = []
    anim_dict_list_test = []
    
    sample_idx = 0
    for tau_u, tau_y in zip(tau_u_list, tau_y_list):
        
        for seed, bias in zip(seed_list, bias_list):
        
            dataset_gen_dict, anim_dict_train, anim_dict_test = generate_instance_dataset(f=f, h=h,
                                                theta_f=theta_f, theta_h=theta_h, x0=x0, tau_u=tau_u, tau_y=tau_y,
                                                min_cons_hard_x_step=min_cons_hard_x_step, max_cons_hard_x_step=max_cons_hard_x_step,
                                                min_cons_hard_y_step=min_cons_hard_y_step, max_cons_hard_y_step=max_cons_hard_y_step,
                                                u_train=u_train, u_test=u_test, y_train=y_train, y_test=y_test,
                                                len_train_data=len_train_data, len_test_data=len_test_data, seed=seed, bias=bias,
                                                clamp_ub=clamp_ub, clamp_lb=clamp_lb,
                                                w_mass=w_mass, w_scale=w_scale)
            
            multidataset_gen_dict[sample_idx] = (dataset_gen_dict, (tau_u, tau_y, seed, bias))
            anim_dict_list_train.append(anim_dict_train)
            anim_dict_list_test.append(anim_dict_test)

            sample_idx = sample_idx + 1
    
    produce_animation = True

    if produce_animation:
        
        def init():
            plt.clf()
            return plt
        
        def animate_train_test(frame):
            
            
            color_1 = "#228B22"
            # color_1 = "#006400"
            color_2 = "#FF4500"
            color_3 = "k"
            
            dpi = 900
            fs = 8
            lw_rr = 0.4
            pad = 1
            rcParams['font.family'] = 'Times New Roman'
            rcParams.update({
            'font.size': 8,                # General font size
            'axes.titlesize': 8,           # Title of axes
            'axes.labelsize': 8,           # Labels of axes
            'xtick.labelsize': 8,          # x-axis tick labels
            'ytick.labelsize': 8,          # y-axis tick labels
            'legend.fontsize': 8,          # Legend font size
            'figure.titlesize': 8          # Figure title font size
            })
            
            anim_dict_train = anim_dict_list_train[frame]
            y_train_branch = anim_dict_train["y_train_branch"]
            y_test_branch = anim_dict_train["y_test_branch"]
            u_train_branch = anim_dict_train["u_train_branch"]
            u_test_branch = anim_dict_train["u_test_branch"]

            dataset_gen_dict = multidataset_gen_dict[frame][0]

            u_gen = dataset_gen_dict["u_train"]
            y_pred_gen = dataset_gen_dict["y_train"]
            data_title = r"Train Dataset"

            plt.subplot(2, 2, 1)
            plt.cla()
            plt.plot(y_pred_gen, linestyle="-", color=color_3, linewidth=1.75*lw_rr, zorder=3)
            plt.plot(y_train_branch, linestyle="--", color=color_2, linewidth=1.5*lw_rr, zorder=1)
            plt.plot(y_test_branch, linestyle="--", color=color_1, linewidth=1.5*lw_rr, zorder=1)
            # plt.fill_between(range(len(y_pred_gen)), y_train_branch, y_test_branch, label="LTI Zone", color='gray', zorder=0, alpha=0.2)
            plt.ylabel('Output')
            plt.ylim([1, 11])
            plt.title(data_title, pad=pad)
            # plt.grid()
            plt.xticks([])

            plt.subplot(2, 2, 3)
            plt.cla()
            plt.plot(u_gen, linestyle="-", color=color_3, linewidth=1.75*lw_rr, zorder=3)
            plt.plot(u_train_branch, linestyle="--", color=color_2, linewidth=1.5*lw_rr, zorder=1)
            plt.plot(u_test_branch, linestyle="--", color=color_1, linewidth=1.5*lw_rr, zorder=1)
            # plt.fill_between(range(len(u_gen)), u_train_branch, u_test_branch, label="Designated zone for input generation", color='gray', zorder=0, alpha=0.2)
            plt.rcParams['figure.dpi'] = 900
            plt.rcParams['savefig.dpi'] = 900
            plt.ylabel('Input')
            plt.ylim([-0.15, 7.15])
            plt.xlabel("Time [step]")
            # plt.grid()
            
            anim_dict_test = anim_dict_list_test[frame]
            y_train_branch_t = anim_dict_test["y_train_branch"]
            y_test_branch_t = anim_dict_test["y_test_branch"]
            u_train_branch_t = anim_dict_test["u_train_branch"]
            u_test_branch_t = anim_dict_test["u_test_branch"]

            u_gen_t = dataset_gen_dict["u_test"]
            y_pred_gen_t = dataset_gen_dict["y_test"]
            data_title_t = r"Test Dataset"

            plt.subplot(2, 2, 2)
            plt.cla()
            plt.plot(y_pred_gen_t, linestyle="-", color=color_3, linewidth=1.75*lw_rr, zorder=3)
            plt.plot(y_train_branch_t, linestyle="--", color=color_2, linewidth=1.5*lw_rr, zorder=1)
            plt.plot(y_test_branch_t, linestyle="--", color=color_1, linewidth=1.5*lw_rr, zorder=1)
            # plt.fill_between(range(len(y_pred_gen)), y_train_branch, y_test_branch, label="LTI Zone", color='gray', zorder=0, alpha=0.2)
            plt.ylim([1, 11])
            # plt.grid()
            plt.title(data_title_t, pad=pad)
            plt.xticks([])
            plt.yticks([])

            plt.subplot(2, 2, 4)
            plt.cla()
            plt.plot(u_gen_t, linestyle="-", color=color_3, linewidth=1.75*lw_rr, zorder=3)
            plt.plot(u_train_branch_t, linestyle="--", color=color_2, linewidth=1.5*lw_rr, zorder=1)
            plt.plot(u_test_branch_t, linestyle="--", color=color_1, linewidth=1.5*lw_rr, zorder=1)
            # plt.fill_between(range(len(u_gen)), u_train_branch, u_test_branch, label="Designated zone for input generation", color='gray', zorder=0, alpha=0.2)
            plt.ylim([-0.15, 7.15])
            plt.xlabel("Time [step]")
            # plt.grid()
            plt.yticks([])
            
            legend_item1 = mlines.Line2D([], [], color="#228B22", label="Test-Real", linestyle='--')
            legend_item2 = mlines.Line2D([], [], color="#FF4500", label="Train-Real", linestyle='--')
            legend_item3 = mlines.Line2D([], [], color="k", label="Synthetic")

            # Add the custom legend to the plot
            plt.figlegend(handles=[legend_item1, legend_item2, legend_item3], loc="upper left", frameon=False, ncol=3)
            
            return plt
        
        dpi = 900
        cw = 3.5
        ar = 0.7
        # fig_anim_train_test = plt.figure(figsize = (12,8), dpi=dpi)
        # anim_test = animation.FuncAnimation(fig_anim_train_test, animate_train_test, init_func=init, frames=len(anim_dict_list_test), interval=1000, blit=False)
        # anim_test.save('gifs/monte_carlo_datageneration_train_test.gif', writer='imagemagick', fps=4)
        plt.figure(figsize = (cw,cw*ar), dpi=dpi)
        fig = animate_train_test(sample_plot_idx)
        # legend_item1 = mlines.Line2D([], [], color="#228B22", label="Real Test Dataset", linestyle='--')
        # legend_item2 = mlines.Line2D([], [], color="#FF4500", label="Real Training Dataset", linestyle='--')
        # legend_item3 = mlines.Line2D([], [], color="k", label="Synthetic Dataset")

        # # Add the custom legend to the plot
        # plt.figlegend(handles=[legend_item1, legend_item2, legend_item3], loc="upper left", frameon=False)
        plt.savefig('plots/MCSamplePlot.pdf', format='pdf', bbox_inches="tight")

    return multidataset_gen_dict


#Recieving Input/Output Data for Cascaded Tanks
folder = os.path.join(r'C:\Users\milad.banitalebi\Desktop\SLIMPEC Project\Task  1.1\Implementation\Datasets\CascadedTanks')
file = os.path.join(folder, "dataBenchmark.csv")
dataset = pd.read_csv(file)
dataset = dataset.rename(
    columns={
        "uEst": "u_train",
        "uVal": "u_test",
        "yEst": "y_train",
        "yVal": "y_test",
    }
)

u_train = torch.tensor(dataset["u_train"], dtype=torch.float32)
y_train = torch.tensor(dataset["y_train"], dtype=torch.float32)
u_test = torch.tensor(dataset["u_test"], dtype=torch.float32)
y_test = torch.tensor(dataset["y_test"], dtype=torch.float32)

# Fitted Paramters for Model (2) Worden et al. (2018)
theta_f = [0.1925, 0.2429, 0.1697, 0.0444, 0.1642]
theta_h = [0]
x0 = torch.tensor([[5.6431, 5.5580, 5.5580, 0]])
min_cons_hard_x_step = torch.tensor([[0 , 0 , 0, -float("inf")]])
max_cons_hard_x_step = torch.tensor([[10, 10, 10, +float("inf")]])
min_cons_hard_y_step = torch.tensor([[0]])
max_cons_hard_y_step = torch.tensor([[10]])

# Dynamic Paramters for Input and Ouput of the System
tau_u_max = 0.85
tau_y_max = 0.85
tau_u_min = 0.25
tau_y_min = 0.25
bias_max = 0.1
bias_min = 0

seed_mean = 42
num_samples = 5

tau_u_list = np.linspace(start=tau_u_min, stop=tau_u_max, num=num_samples)
tau_y_list = np.linspace(start=tau_y_min, stop=tau_y_max, num=num_samples)
seed_list = np.linspace(start=seed_mean - 20*int(num_samples/2), stop=seed_mean + 20*int(num_samples/2), num=num_samples, dtype=int)
bias_list = np.linspace(start=bias_min, stop=bias_max, num=num_samples)

print(f"tau_u_list: {tau_u_list}")
print(f"tau_y_list: {tau_y_list}")
print(f"seed_list: {seed_list}")
print(f"bias_list: {bias_list}")
# Generating Artifiical Train and Test Data
len_train_data = 1*1024
len_test_data = 1*1024

clamp_ub = 2
clamp_lb = -1

w_mass = 0.5
w_scale = 2

multidataset_gen_dict = generate_multi_dataset(f=f_cascaded_tanks, h=h_cascaded_tanks,
                                               theta_f=theta_f, theta_h=theta_h, x0=x0, tau_u_list=tau_u_list, tau_y_list=tau_y_list,
                                               min_cons_hard_x_step=min_cons_hard_x_step, max_cons_hard_x_step=max_cons_hard_x_step,
                                               min_cons_hard_y_step=min_cons_hard_y_step, max_cons_hard_y_step=max_cons_hard_y_step,
                                               u_train=u_train, u_test=u_test, y_train=y_train, y_test=y_test,
                                               len_train_data=len_train_data, len_test_data=len_test_data, seed_list=seed_list, bias_list=bias_list,
                                               clamp_ub=clamp_ub, clamp_lb=clamp_lb,
                                               w_mass=w_mass, w_scale=w_scale,
                                               sample_plot_idx=20)

with open('dicts_mc/multidataset_gen_dict.pkl', 'wb') as f:
    pk.dump(multidataset_gen_dict, f)
