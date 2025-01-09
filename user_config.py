import torch
from torch import nn as nn
import pandas as pd

# Model (1) From Worden et al. (2018)
# [USER-DEFINED] physical state function (f)
def f_cascaded_tanks(x_step, u_step, theta_f):

    x_step = torch.clip(x_step, min=0)
    
    x_1 = x_step[:, 0:1]
    x_2 = x_step[:, 1:2]

    theta_f_1 = theta_f[0]
    theta_f_3 = theta_f[1]
    theta_f_4 = theta_f[2]
    theta_f_5 = theta_f[3]

    dx_1 = -theta_f_1 * torch.sqrt(x_1) + theta_f_4 * u_step

    x_1_overflow = x_1 + dx_1
    for b in range(x_step.shape[0]):
        if x_1_overflow[b] <= 10:
            dx_2 = theta_f_1 * torch.sqrt(x_1) - theta_f_3 * torch.sqrt(x_2)
        elif x_1_overflow[b] > 10:
            dx_2 = theta_f_1 * torch.sqrt(x_1) - theta_f_3 * torch.sqrt(x_2) + theta_f_5 * u_step
        else:
            print(f"error in conditioning the model, {x_1}, || {x_2}")

    dx = torch.cat((dx_1, dx_2), 1)

    return dx

# [USER-DEFINED] physical output function (h)
def h_cascaded_tanks(x_sim, theta_h):
    y_sim = x_sim[:,1:2]

    return y_sim
#========================================================================================================================================================================================
# [USER-DEFINED] Model Parameters
n_y = 1                                                         #Number of outputs
n_u = 1                                                         #Number of Inputs
n_x = 2                                                         #Number of Physical States
n_x_b = 3                                                       #Number of Augmented Neural States
B = 1
# # --------------------------------------------------------
# Constraints From Model (1) Worden et al. (2018)
theta_f_initial_guess = [0.2, 0.2, 0.2, 0.1]                    #Initial Guess for Theta_f
theta_h_initial_guess = [0]                                     #Initial Guess for Theta_h
x0_phys_initial_guess = [5, 5]                                  #Initial Guess for Physical States
# # --------------------------------------------------------
x_b_0 = torch.zeros((1, n_x_b))                                 #Initial Guess for Augmented Neural States
# --------------------------------------------------------
# Constraints From Worden et al. (2018)
min_cons_hard_x_step = torch.zeros((B, n_x))
max_cons_hard_x_step = torch.ones((B, n_x)) * 10
min_cons_hard_y_step = torch.zeros((B, n_y))
max_cons_hard_y_step = torch.ones((B, n_y)) * 10
#========================================================================================================================================================================================
# USER-DEFINED] Input/Output Data
#Receiving Input/Output Data for Cascaded Tanks
dataset = pd.read_csv("dataBenchmark.csv")
dataset = dataset.rename(
    columns={
        "uEst": "u_train",
        "uVal": "u_test",
        "yEst": "y_train",
        "yVal": "y_test",
    }
)

# Using Benchmark Data for Training and Testing
u_train = torch.tensor(dataset["u_train"], dtype=torch.float32)
y_train = torch.tensor(dataset["y_train"], dtype=torch.float32)
u_val = torch.tensor(dataset["u_test"], dtype=torch.float32)[6:]
y_val = torch.tensor(dataset["y_test"], dtype=torch.float32)[6:]
u_test = torch.tensor(dataset["u_test"], dtype=torch.float32)[6:]
y_test = torch.tensor(dataset["y_test"], dtype=torch.float32)[6:]

#========================================================================================================================================================================================
# user config dictionary to pass 
user_params = {
    "f_physical_model": f_cascaded_tanks,
    "h_physical_model": h_cascaded_tanks,
    "n_x": n_x,
    "n_y": n_y,
    "n_u": n_u,
    "n_x_b": n_x_b,
    "theta_f_initial_guess": theta_f_initial_guess,
    "theta_h_initial_guess": theta_h_initial_guess,
    "x0_phys_initial_guess": x0_phys_initial_guess,
    "x_b_0": x_b_0,
    "u_train": u_train,
    "y_train": y_train,
    "u_val": u_val,
    "y_val": y_val,
    "u_test": u_test,
    "y_test": y_test,
    "min_cons_hard_x_step": min_cons_hard_x_step,
    "max_cons_hard_x_step": max_cons_hard_x_step,
    "min_cons_hard_y_step": min_cons_hard_y_step,
    "max_cons_hard_y_step": max_cons_hard_y_step
    }
