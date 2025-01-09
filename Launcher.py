
# ## SLIMPEC Task 1.1 Cascaded Tanks Benchmark
import torch
from torch import nn as nn
from Core import AugmentedModel
import default_config
import user_config
import time
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True  # Makes the results deterministic.
    torch.backends.cudnn.benchmark = False  # Disables the benchmark mode which can cause non-deterministic behavior.
    print(f"Seed set to: {seed}")


def launch_greyboxmodel(**kwargs):

    default_args =default_config.default_params

    if "lr_x0_phys" in kwargs:
        lr_x0_phys = kwargs["lr_x0_phys"]
    else:
        lr_x0_phys = default_args["lr_x0_phys"]

    if "lr_theta_f" in kwargs:
        lr_theta_f = kwargs["lr_theta_f"]
    else:
        lr_theta_f = default_args["lr_theta_f"]

    if "lr_theta_h" in kwargs:
        lr_theta_h = kwargs["lr_theta_h"]
    else:
        lr_theta_h = default_args["lr_theta_h"]

    if "lr_x0_phys_norm" in kwargs:
        lr_x0_phys_norm = kwargs["lr_x0_phys_norm"]
    else:
        lr_x0_phys_norm = default_args["lr_x0_phys_norm"]
    
    if "lr_theta_f_norm" in kwargs:
        lr_theta_f_norm = kwargs["lr_theta_f_norm"]
    else:
        lr_theta_f_norm = default_args["lr_theta_f_norm"]

    if "lr_theta_h_norm" in kwargs:
        lr_theta_h_norm = kwargs["lr_theta_h_norm"]
    else:
        lr_theta_h_norm = default_args["lr_theta_h_norm"]

    if "lr_f_b" in kwargs:
        lr_f_b = kwargs["lr_f_b"]
    else:
        lr_f_b = default_args["lr_f_b"]

    if "lr_g_b" in kwargs:
        lr_g_b = kwargs["lr_g_b"]
    else:
        lr_g_b = default_args["lr_g_b"]

    if "lr_h_b" in kwargs:
        lr_h_b = kwargs["lr_h_b"]
    else:
        lr_h_b = default_args["lr_h_b"]

    if "lr_x_b_0" in kwargs:
        lr_x_b_0 = kwargs["lr_x_b_0"]
    else:
        lr_x_b_0 = default_args["lr_x_b_0"]
    
    if "lr_x_f_0" in kwargs:
        lr_x_f_0 = kwargs["lr_x_f_0"]
    else:
        lr_x_f_0 = default_args["lr_x_f_0"]
    
    if "epochs_phys" in kwargs:
        epochs_phys = kwargs["epochs_phys"]
    else:
        epochs_phys = default_args["epochs_phys"]

    if "epochs_neural" in kwargs:
        epochs_neural = kwargs["epochs_neural"]
    else:
        epochs_neural = default_args["epochs_neural"]

    if "n_feat" in kwargs:
        n_feat = kwargs["n_feat"]
    else:
        n_feat = default_args["n_feat"]

    if "num_hidden_layers" in kwargs:
        num_hidden_layers = kwargs["num_hidden_layers"]
    else:
        num_hidden_layers = default_args["num_hidden_layers"]

    if "augmentation" in kwargs:
        augmentation = kwargs["augmentation"]
    else:
        augmentation = default_args["augmentation"]

    if "batch_size" in kwargs:
        batch_size = kwargs["batch_size"]
    else:
        batch_size = 0

    if "subseq_len_train" in kwargs:
        subseq_len_train = kwargs["subseq_len_train"]
    else:
        subseq_len_train = None
    
    if "subseq_len_val" in kwargs:
        subseq_len_val = kwargs["subseq_len_val"]
    else:
        subseq_len_val = None

    if "stride_train" in kwargs:
        stride_train = kwargs["stride_train"]
    else:
        stride_train = None
    
    if "stride_val" in kwargs:
        stride_val = kwargs["stride_val"]
    else:
        stride_val = None

    if "display_step" in kwargs:
        display_step = kwargs["display_step"]
    else:
        display_step = default_args["display_step"]

    if "include_f_b" in kwargs:
        include_f_b = kwargs["include_f_b"]
    else:
        include_f_b = default_args["include_f_b"]

    if "include_g_b" in kwargs:
        include_g_b = kwargs["include_g_b"]
    else:
        include_g_b = default_args["include_g_b"]

    if "include_h_b" in kwargs:
        include_h_b = kwargs["include_h_b"]
    else:
        include_h_b = default_args["include_h_b"]

    if "saved_model_file_name" in kwargs:
        saved_model_file_name = kwargs["saved_model_file_name"]
    else:
        saved_model_file_name = default_args["saved_model_file_name"]

    if "x_b_exploration_scale" in kwargs:
        x_b_exploration_scale = kwargs["x_b_exploration_scale"]
    else:
        x_b_exploration_scale = default_args["x_b_exploration_scale"]

    # Required arguments provided by user config file
    default_param_override_display = False
    if "f_physical_model" in kwargs:
        f_userdefined = kwargs["f_physical_model"]
        if default_param_override_display:
            override_arg = "f_physical_model"
            print(f" {override_arg} was overridden by the direct call of the function in the main call. current value for {override_arg} is {f_userdefined}.")
    else:
        f_userdefined = user_config.user_params["f_physical_model"]
        
    if "h_physical_model" in kwargs:
        h_userdefined = kwargs["h_physical_model"]
        if default_param_override_display:
            override_arg = "h_physical_model"
            print(f" {override_arg} was overridden by the direct call of the function in the main call. current value for {override_arg} is {h_userdefined}.")
    else:
        h_userdefined = user_config.user_params["h_physical_model"]

    if "n_x" in kwargs:
        n_x = kwargs["n_x"]
    else:
        n_x = user_config.user_params["n_x"]

    if "n_y" in kwargs:
        n_y = kwargs["n_y"]
    else:
        n_y = user_config.user_params["n_y"]

    if "n_u" in kwargs:
        n_u = kwargs["n_u"]
    else:
        n_u = user_config.user_params["n_u"]

    if "n_x_b" in kwargs:
        n_x_b = kwargs["n_x_b"]
    else:
        n_x_b = user_config.user_params["n_x_b"]

    if "theta_f_initial_guess" in kwargs:
        theta_f_initial_guess = kwargs["theta_f_initial_guess"]
    else:
        theta_f_initial_guess = user_config.user_params["theta_f_initial_guess"]

    if "theta_h_initial_guess" in kwargs:
        theta_h_initial_guess = kwargs["theta_h_initial_guess"]
    else:
        theta_h_initial_guess = user_config.user_params["theta_h_initial_guess"]

    if "x0_phys_initial_guess" in kwargs:
        x0_phys_initial_guess = kwargs["x0_phys_initial_guess"]
    else:
        x0_phys_initial_guess = user_config.user_params["x0_phys_initial_guess"]

    if "x_b_0" in kwargs:
        x_b_0 = kwargs["x_b_0"]
    else:
        x_b_0 = user_config.user_params["x_b_0"]

    if "Binary_state" in kwargs:
        Binary_state = kwargs["Binary_state"]
    else:
        Binary_state = torch.ones((n_x + n_x_b, n_x + n_x_b))
        
    if "Binary_output" in kwargs:
        Binary_output = kwargs["Binary_output"]
    else:
        Binary_output = torch.ones((n_y, n_x + n_x_b))

    if "u_train" in kwargs:
        u_train = kwargs["u_train"]
    else:
        u_train = user_config.user_params["u_train"]

    if "y_train" in kwargs:
        y_train = kwargs["y_train"]
    else:
        y_train = user_config.user_params["y_train"]

    if "u_val" in kwargs:
        u_val = kwargs["u_val"]
    else:
        u_val = user_config.user_params["u_val"]

    if "y_val" in kwargs:
        y_val = kwargs["y_val"]
    else:
        y_val = user_config.user_params["y_val"]
    
    if "u_test" in kwargs:
        u_test = kwargs["u_test"]
    else:
        u_test = user_config.user_params["u_test"]

    if "y_test" in kwargs:
        y_test = kwargs["y_test"]
    else:
        y_test = user_config.user_params["y_test"]

    if "lr_neuralnet" in kwargs:
        lr_neuralnet = kwargs["lr_neuralnet"]
    else:
        lr_neuralnet = {"f_b":lr_f_b, "g_b": lr_g_b, "h_b": lr_h_b}
    
    if "min_cons_hard_x_step" in kwargs:
        min_cons_hard_x_step = kwargs["min_cons_hard_x_step"]
    else:
        min_cons_hard_x_step = user_config.user_params["min_cons_hard_x_step"]

    if "max_cons_hard_x_step" in kwargs:
        max_cons_hard_x_step = kwargs["max_cons_hard_x_step"]
    else:
        max_cons_hard_x_step = user_config.user_params["max_cons_hard_x_step"]

    if "min_cons_hard_y_step" in kwargs:
        min_cons_hard_y_step = kwargs["min_cons_hard_y_step"]
    else:
        min_cons_hard_y_step = user_config.user_params["min_cons_hard_y_step"]

    if "max_cons_hard_y_step" in kwargs:
        max_cons_hard_y_step = kwargs["max_cons_hard_y_step"]
    else:
        max_cons_hard_y_step = user_config.user_params["max_cons_hard_y_step"]
        
    if default_param_override_display:
        for override_arg in kwargs:
            print(f"{override_arg} was overwritten by the direct call of the Launcher in the Main.py script.",
                    f"current value for {override_arg} is {kwargs[override_arg]}.")

    # Developing tools arguments  
    if "dev_tool_animate" in kwargs:
        dev_tool_animate = kwargs["dev_tool_animate"]
    else:
        dev_tool_animate = False

    if "init_std" in kwargs:
        init_std = kwargs["init_std"]
    else:
        init_std = 1e-8

    if "lr_mu_x" in kwargs:
        lr_mu_x = kwargs["lr_mu_x"]
    else:
        lr_mu_x = lr_x0_phys
    
    if "lr_sigma_x" in kwargs:
        lr_sigma_x = kwargs["lr_sigma_x"]
    else:
        lr_sigma_x = lr_x0_phys/10

    if "penalty_coef_dict" in kwargs:
        penalty_coef_dict = kwargs["penalty_coef_dict"]
    else:
        penalty_coef_dict = {
                            "mu_x_f":         1e-1,
                            "mu_x_b":         1e-1,
                            "sigma_x_f":      1e-1,
                            "sigma_x_b":      1e-1,
                            }
    if "neuron_dict" in kwargs:
        neuron_dict = kwargs["neuron_dict"]
    else:
        neuron_dict = {
            "f_b": [],
            "g_b": [], 
            "h_b": [],  
            }
        
    if "optim_alg_dict" in kwargs:
        optim_alg_dict = kwargs["optim_alg_dict"]
    else:
        optim_alg_dict = {
            "physical optimization algorithm": torch.optim.AdamW,
            "normal physical optimization algorithm": torch.optim.AdamW, 
            "neural optimization algorithm": torch.optim.AdamW,  
            }
        
    if "loss_dict" in kwargs:
        loss_dict = kwargs["loss_dict"]
    else:

        def l2_loss(y, y_sim, sim_obj=None):
            loss = torch.mean((y - y_sim)**2)
            return loss, loss

        loss_dict = {
            "physical loss": l2_loss,
            "normal physical loss": l2_loss, 
            "neural loss": l2_loss,  
            }
    
    if "penalty_loss_dict" in kwargs:
        penalty_loss_dict = kwargs["penalty_loss_dict"]
    else:

        def l2_loss_penalty(y, y_sim, sim_obj=None):
            loss = torch.mean((y - y_sim)**2)
            return loss

        penalty_loss_dict = {
                            "mu_x_f":         l2_loss_penalty,
                            "mu_x_b":         l2_loss_penalty,
                            "sigma_x_f":      l2_loss_penalty,
                            "sigma_x_b":      l2_loss_penalty,
                        }
    
    if "act_fn_dict" in kwargs:
        act_fn_dict = kwargs["act_fn_dict"]
    else:

        act_fn_dict = {
            "f_b": nn.Tanh(),
            "g_b": nn.Tanh(), 
            "h_b": nn.Tanh(),  
                    }
    
    if "bias_dict" in kwargs:
        bias_dict = kwargs["bias_dict"]
    else:

        bias_dict = {
                    "f_b": {"input layer": False, "hidden layer": False, "output layer": False},
                    "g_b": {"input layer": False, "hidden layer": False, "output layer": False},
                    "h_b": {"input layer": True,  "hidden layer": True,  "output layer": True}
                    }
        
    if "iv_est_dict" in kwargs:
        iv_est_dict = kwargs["iv_est_dict"]
        if "Static Neural Network" in iv_est_dict.keys() and subseq_len_train != subseq_len_val:
            print("ERROR: Using the Static Neural Network for estimating the initial values require that subseq_len_train == subseq_len_val")
            
    else:
        iv_est_dict = {"Physical Estimate": {}} 
    if "device" in kwargs:
        device = kwargs["device"]
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if "dataloader_num_workers" in kwargs:
        dataloader_num_workers = kwargs["dataloader_num_workers"]
    else:
        dataloader_num_workers = 1
        
    if "seed" in kwargs:
        seed = kwargs["seed"]
    else:
        seed = []
        
    if "gamma_power" in kwargs:
        gamma_power = kwargs["gamma_power"]
    else:
        gamma_power = 1
    
    if "display_gamma_condition_stat" in kwargs:
        display_gamma_condition_stat = kwargs["display_gamma_condition_stat"]
    else:
        display_gamma_condition_stat = False
    
    if "adaptive_gamma_eval" in kwargs:
        adaptive_gamma_eval = kwargs["adaptive_gamma_eval"]
    else:
        adaptive_gamma_eval = False
    
    # Setting the constant Seed for random operations
    if isinstance(seed, (int, float)):
        set_seed(seed)
    
    # Defining the Augmented Model
    model = AugmentedModel(
                            f_userdefined=f_userdefined,
                            h_userdefined=h_userdefined,
                            n_y=n_y,
                            n_x=n_x, 
                            n_x_b=n_x_b, 
                            n_u=n_u, 
                            u_train=u_train, 
                            y_train=y_train, 
                            u_val=u_val, 
                            y_val=y_val, 
                            u_test=u_test, 
                            y_test=y_test,
                            theta_f_initial_guess=theta_f_initial_guess,
                            theta_h_initial_guess=theta_h_initial_guess, 
                            x0_phys_initial_guess=x0_phys_initial_guess, 
                            x_b_0=x_b_0,
                            lr_x0_phys=lr_x0_phys, 
                            lr_theta_f=lr_theta_f, 
                            lr_theta_h=lr_theta_h,
                            lr_x0_phys_norm=lr_x0_phys_norm, 
                            lr_theta_f_norm=lr_theta_f_norm, 
                            lr_theta_h_norm=lr_theta_h_norm,
                            lr_neuralnet=lr_neuralnet, 
                            lr_x_b_0=lr_x_b_0, 
                            lr_x_f_0=lr_x_f_0,
                            epochs_phys=epochs_phys, 
                            epochs_neural=epochs_neural,
                            Binary_state=Binary_state, 
                            Binary_output=Binary_output,
                            batch_size=batch_size, 
                            save_file_name=saved_model_file_name,
                            n_feat=n_feat, 
                            num_hidden_layers=num_hidden_layers,
                            include_f_b=include_f_b,
                            include_g_b=include_g_b,
                            include_h_b=include_h_b,
                            display_step=display_step,
                            subseq_len_train=subseq_len_train,
                            subseq_len_val=subseq_len_val,
                            stride_train=stride_train,
                            stride_val=stride_val,
                            augmentation=augmentation,
                            min_cons_hard_x_step=min_cons_hard_x_step,
                            max_cons_hard_x_step=max_cons_hard_x_step,
                            min_cons_hard_y_step=min_cons_hard_y_step,
                            max_cons_hard_y_step=max_cons_hard_y_step,
                            x_b_exploration_scale=x_b_exploration_scale,
                            init_std=init_std,
                            lr_mu_x=lr_mu_x,
                            lr_sigma_x=lr_sigma_x,
                            penalty_coef_dict=penalty_coef_dict,
                            dev_tool_animate=dev_tool_animate,
                            optim_alg_dict=optim_alg_dict,
                            loss_dict=loss_dict,
                            penalty_loss_dict=penalty_loss_dict,
                            neuron_dict=neuron_dict,
                            act_fn_dict=act_fn_dict,
                            bias_dict = bias_dict,
                            device=device,
                            dataloader_num_workers=dataloader_num_workers,
                            seed = seed,
                            iv_est_dict=iv_est_dict,
                            gamma_power=gamma_power,
                            display_gamma_condition_stat=display_gamma_condition_stat,
                            adaptive_gamma_eval=adaptive_gamma_eval
                            )

    start_time = time.time()
    augmodel, y_pred_test = model.augmented_model_fit()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Augmented Modeling Program Execution Time: {execution_time} seconds")
    
    result_dict = model.save_results()
    y_pred_test_lin = []
    linmodel = []

    return y_pred_test, y_pred_test_lin, augmodel, result_dict, linmodel, model, execution_time

