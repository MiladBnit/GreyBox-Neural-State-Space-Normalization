# %% [markdown]
# ## SLIMPEC Task 1.1

# %% [markdown]
# Run the following cells to load data, create train/test sets, normalize the data

# %%
import pandas as pd
import numpy as np
import random
import torch
from torch import nn as nn
from Classes import (DataScaler,
                        FANSNeuralStateUpdate, FANSNeuralOutputUpdate,
                        FANSNeuralStateUpdate_Zero, FANSNeuralOutputUpdate_Zero,
                        PhysicalStateUpdate, PhysicalOutput,
                        PhysicalStateUpdateNormalized, PhysicalOutputNormalized,
                        PhysicalStateEstimator,
                        StateSpaceSimulator,
                        StateSpaceSimulatorAugmented, RobustStateSpaceSimulatorAugmented,
                        StateSpaceSimulatorAugmented_Non_normalized, RobustStateSpaceSimulatorAugmented_Non_normalized,
                        StateSpaceSimulatorAugmented_MANAM, RobustStateSpaceSimulatorAugmented_MANAM,
                        AugModel,
                        SubsequenceDataset, DataLoader)

from Functions import save_result, result_plot, training_progress_display
from Functions import save_checkpoint, load_checkpoint

# %%
class AugmentedModel():

    def __init__(self,      
                f_userdefined,
                h_userdefined,
                n_y,
                n_x, 
                n_x_b, 
                n_u, 
                u_train, 
                y_train,
                u_val,
                y_val, 
                u_test, 
                y_test,
                theta_f_initial_guess,
                theta_h_initial_guess, 
                x0_phys_initial_guess, 
                x_b_0,
                lr_x0_phys, 
                lr_theta_f, 
                lr_theta_h,
                lr_x0_phys_norm, 
                lr_theta_f_norm, 
                lr_theta_h_norm,
                lr_neuralnet, 
                lr_x_b_0, 
                lr_x_f_0,
                epochs_phys, 
                epochs_neural,
                Binary_state, 
                Binary_output,
                batch_size, 
                save_file_name,
                n_feat, 
                num_hidden_layers,
                include_f_b,
                include_g_b,
                include_h_b,
                display_step,
                subseq_len_train,
                subseq_len_val,
                stride_train,
                stride_val,
                augmentation,
                min_cons_hard_x_step,
                max_cons_hard_x_step,
                min_cons_hard_y_step,
                max_cons_hard_y_step,
                x_b_exploration_scale,
                init_std,
                lr_mu_x,
                lr_sigma_x,
                penalty_coef_dict,
                penalty_loss_dict,
                dev_tool_animate,
                neuron_dict,
                optim_alg_dict,
                loss_dict,
                act_fn_dict,
                bias_dict,
                device,
                dataloader_num_workers,
                seed,
                iv_est_dict,
                gamma_power,
                display_gamma_condition_stat,
                adaptive_gamma_eval):

        self.gamma_power = gamma_power
        self.display_gamma_condition_stat = display_gamma_condition_stat
        self.device = device
        self.dataloader_num_workers = dataloader_num_workers
        self.n_y = n_y                                                      # Number of outputs
        self.n_x = n_x                                                      # Number of Physical States
        self.n_x_b = n_x_b                                                  # Number of Augmented Neural States
        self.n_u = n_u                                                      # Number of Inputs

        self.u_train = u_train                                              # Training Input
        self.u_val = u_val                                                  # Validation Input
        self.u_test = u_test                                                # Test Input
        self.y_train = y_train                                              # Training Output
        self.y_val = y_val                                                  # Validation Output
        self.y_test = y_test                                                # Test Output

        self.theta_f_initial_guess = theta_f_initial_guess                  # Initial Guess for Theta_f
        self.theta_h_initial_guess = theta_h_initial_guess                  # Initial Guess for Theta_h
        self.x0_phys_initial_guess = x0_phys_initial_guess                  # Initial Guess for Physical States

        self.lr_x0_phys = lr_x0_phys                                        # Learning Rate for Initial value of Un-normalized Physical States
        self.lr_theta_f = lr_theta_f                                        # Learning Rate for Theta_f for Un-normalized Optimization
        self.lr_theta_h = lr_theta_h                                        # Learning Rate for Theta_h for Un-normalized Optimization
        self.lr_neuralnet = lr_neuralnet                                    # Learning Rate for Neural Network
        self.lr_x0_phys_norm = lr_x0_phys_norm                              # Learning Rate for normalized Physical States
        self.lr_theta_f_norm = lr_theta_f_norm                              # Learning Rate for Theta_f for normalized Optimization
        self.lr_theta_h_norm = lr_theta_h_norm                              # Learning Rate for Theta_h for normalized Optimization
        self.lr_x_b_0 = lr_x_b_0                                            # Learning Rate for Initial value of Augmented Neural States
        self.lr_x_f_0 =lr_x_f_0                                             # Learning Rate for Initial value of Augmented Physical States
        self.lr_mu_x = lr_mu_x                                              # Learning Rate for Initial value of mu_x
        self.lr_sigma_x = lr_sigma_x                                        # Learning Rate for Initial value of Sigma_x

        self.epochs_phys = epochs_phys                                      # Number of Training Epochs for Un-normalized Optimization
        self.epochs_neural = epochs_neural                                  # Number of Training Epochs for Neural Training

        self.n_feat = n_feat                                                # Number of features for Neural Layers
        self.num_hidden_layers = num_hidden_layers                          # Number of Hidden Layers (0: Shallow Neural Network)

        self.include_f_b = include_f_b                                      # Whether to include f_b or not             
        self.include_g_b = include_g_b                                      # Whether to include g_b or not
        self.include_h_b = include_h_b                                      # Whether to include h_b or not
        self.x_b_0 = x_b_0.to(self.device)                                  # Initial Guess for Augmented Neural States

        self.Binary_state = Binary_state                                    # Binary Matrix for States
        self.Binary_output = Binary_output                                  # Binary Matrix for Outputs
        
        self.f_userdefined = f_userdefined                                  # User-Defined Physical State Function (f)
        self.h_userdefined = h_userdefined                                  # User-Defined Physical Output Function (h)

        self.subseq_len_train = subseq_len_train                            # Subsequence length for Batch Samples for Training Data
        self.subseq_len_val = subseq_len_val                                # Subsequence length for Batch Samples for Validation Data
        self.stride_train = stride_train                                    # Stride of the batch dataset (data repeatability in the batched form) for the Training Data
        self.stride_val = stride_val                                        # Stride of the batch dataset (data repeatability in the batched form) for the Validation Data
        if batch_size == 0:
            self.training_mode = 0                                          # Training Mode: 0 for pure training/ 1 for mini-batch
            print("Note: training mode set to pure (no mini-batch)")
        else:
            self.training_mode = 1
            self.batch_size = batch_size

        if self.training_mode == 0:
            self.subseq_len_train = u_train.shape[0]
            self.subseq_len_val = u_val.shape[0]
            self.batch_size = 1
            self.stride_train = 1
            self.stride_val = 1
            print("Note: mini-batch sequence length set to sample length")
            print("Note: mini-batch sequence length is equal to sample length -> setting training mode to one iteration per epoch")
            self.drop_condition = False
        else:
            self.drop_condition = True

        x_sim_phys_mean = []
        x_sim_phys_std = []

        u_mean = torch.mean(self.u_train)
        u_std = torch.std(self.u_train)
        y_mean = torch.mean(self.y_train)
        y_std = torch.std(self.y_train)

        datascaler = DataScaler(x_mean=x_sim_phys_mean,
                                x_std=x_sim_phys_std,
                                u_mean=u_mean,
                                u_std=u_std,
                                y_mean=y_mean,
                                y_std=y_std,
                                device="cpu")
        
        self.datascaler = datascaler
        self.u_train = u_train.view(-1, n_u)
        self.y_train = y_train.view(-1, n_y)
        self.u_val = u_val.view(-1, n_u)
        self.y_val = y_val.view(-1, n_y)
        self.u_test = u_test.view(-1, n_u)
        self.y_test = y_test.view(-1, n_y)

        self.u_train_norm = datascaler.zscore_normalize(self.u_train, self.datascaler.u_mean, self.datascaler.u_std)
        self.u_val_norm = datascaler.zscore_normalize(self.u_val, self.datascaler.u_mean, self.datascaler.u_std)
        self.u_test_norm = datascaler.zscore_normalize(self.u_test, self.datascaler.u_mean, self.datascaler.u_std)
        self.y_train_norm = datascaler.zscore_normalize(self.y_train, self.datascaler.y_mean, self.datascaler.y_std)
        self.y_val_norm = datascaler.zscore_normalize(self.y_val, self.datascaler.y_mean, self.datascaler.y_std)
        self.y_test_norm = datascaler.zscore_normalize(self.y_test, self.datascaler.y_mean, self.datascaler.y_std)

        self.min_cons_hard_x_step = min_cons_hard_x_step.to(self.device)
        self.max_cons_hard_x_step = max_cons_hard_x_step.to(self.device)
        self.min_cons_hard_y_step = min_cons_hard_y_step.to(self.device)
        self.max_cons_hard_y_step = max_cons_hard_y_step.to(self.device)
        self.x_b_exploration_scale = torch.tensor(x_b_exploration_scale, device=self.device)

        self.augmentation = augmentation                                    # Defines the Augmentation Method of the model (Robust or Exact)
        self.init_std = init_std

        self.penalty_coef_dict = penalty_coef_dict
        self.penalty_loss_dict = penalty_loss_dict
        self.neuron_dict = neuron_dict
        self.optim_alg_dict = optim_alg_dict
        self.loss_dict = loss_dict
        self.act_fn_dict = act_fn_dict
        self.bias_dict = bias_dict
        self.iv_est_dict = iv_est_dict

        self.dev_tool_animate = dev_tool_animate
        self.save_file_name = save_file_name                                # Name of the saved Model
        self.display_step = display_step                                    # Training Progress Display Steps  
        
        validation_gen = torch.Generator()
        if isinstance(seed, (int, float)):
            validation_gen.manual_seed(seed)
        
        train_dataset = SubsequenceDataset(self.u_train, self.y_train, subseq_len=self.subseq_len_train, stride=self.stride_train)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_condition,
                                num_workers=self.dataloader_num_workers, pin_memory=False)
        
        val_dataset = SubsequenceDataset(self.u_val, self.y_val, subseq_len=self.subseq_len_val, stride=self.stride_val)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_condition,
                                num_workers=self.dataloader_num_workers, pin_memory=False,
                                generator=validation_gen)
        
        train_norm_dataset = SubsequenceDataset(self.u_train_norm, self.y_train_norm, subseq_len=self.subseq_len_train, stride=self.stride_train)
        self.train_norm_loader = DataLoader(train_norm_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_condition,
                                    num_workers=self.dataloader_num_workers)
        
        val_norm_dataset = SubsequenceDataset(self.u_val_norm, self.y_val_norm, subseq_len=self.subseq_len_val, stride=self.stride_val)
        self.val_norm_loader = DataLoader(val_norm_dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_condition,
                                            num_workers=self.dataloader_num_workers,
                                            generator=validation_gen)
        
        self.adaptive_gamma_eval = adaptive_gamma_eval

    def train_phys(self, simulator_phys, theta_f, theta_h, x0_phys_batch, optimizer_phys, train_loader):
        simulator_phys.train()
        loss_phys_total_train = 0
        batch_pop = 0
        for batch_idx, (u_train_batch, y_train_batch) in enumerate(train_loader):
            u_train_batch, y_train_batch = u_train_batch.to(self.device), y_train_batch.to(self.device)
            optimizer_phys.zero_grad()
            x_sim_phys_batch, y_sim_phys_batch = simulator_phys(x0_phys_batch,
                                                                u_train_batch,
                                                                theta_f=theta_f,
                                                                theta_h=theta_h,
                                                                min_cons_hard_x_step=self.min_cons_hard_x_step,
                                                                max_cons_hard_x_step=self.max_cons_hard_x_step,
                                                                min_cons_hard_y_step=self.min_cons_hard_y_step,
                                                                max_cons_hard_y_step=self.max_cons_hard_y_step)
            
            loss_phys_train_batch, loss_phys_train_nominal_batch = self.loss_dict["physical loss"](y_train_batch, y_sim_phys_batch, sim_obj=simulator_phys)
            loss_phys_train_batch.backward()
            optimizer_phys.step()
            batch_pop = batch_pop + u_train_batch.shape[1]
            loss_phys_total_train = loss_phys_total_train + loss_phys_train_nominal_batch * u_train_batch.shape[1]
        return loss_phys_total_train/batch_pop, theta_f, theta_h

    def evaluate_phys(self, simulator_phys, theta_f, theta_h, x0_phys_batch, val_loader):
        simulator_phys.eval()
        loss_phys_total_val = 0
        batch_pop = 0
        with torch.no_grad():
            for batch_idx, (u_val_batch, y_val_batch) in enumerate(val_loader):
                u_val_batch, y_val_batch = u_val_batch.to(self.device), y_val_batch.to(self.device)
                x_val_phys_batch, y_val_phys_batch = simulator_phys(x0_phys_batch,
                                                                    u_val_batch,
                                                                    theta_f=theta_f,
                                                                    theta_h=theta_h,
                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step,
                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step,
                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step,
                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step)
                
                loss_phys_val_batch, loss_phys_val_nominal_batch = self.loss_dict["physical loss"](y_val_batch, y_val_phys_batch, sim_obj=simulator_phys)
                batch_pop = batch_pop + u_val_batch.shape[1]
                loss_phys_total_val = loss_phys_total_val + loss_phys_val_nominal_batch * u_val_batch.shape[1]
        return loss_phys_total_val/batch_pop

    def train_neural(self, simulator_augmented, theta_f_fitted, theta_h_fitted, x_b_0_batch, estimator, datascaler, optimizer_neural, train_loader, epoch_neural):
        simulator_augmented.train()
         
        loss_neural_total_train = 0
        batch_pop = 0
        for batch_idx, (u_train_norm_batch, y_train_norm_batch) in enumerate(train_loader):
            u_train_norm_batch, y_train_norm_batch = u_train_norm_batch.to(self.device), y_train_norm_batch.to(self.device)
            optimizer_neural.zero_grad()
            
            x_f_0_est_neural_batch = estimator(u=u_train_norm_batch, y=y_train_norm_batch, mean=datascaler.x_mean, std=datascaler.x_std)

            if self.augmentation == "Exact" or self.augmentation == "Exact - MANAM" or self.augmentation ==  "Exact - NonNormalized":
                x_f_sim_neural_batch, x_b_sim_neural_batch, y_sim_neural_batch = simulator_augmented(x_f_0=x_f_0_est_neural_batch, x_b_0=x_b_0_batch, u=u_train_norm_batch,
                                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm_batch,
                                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm_batch,
                                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm_batch,
                                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm_batch,
                                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub)
            elif self.augmentation == "Robust" or self.augmentation == "Robust - MANAM" or self.augmentation ==  "Robust - NonNormalized":
                x_f_sim_neural_batch, x_b_sim_neural_batch, y_sim_neural_batch, x_p_sim_neural_batch = simulator_augmented(x_f_0=x_f_0_est_neural_batch, x_b_0=x_b_0_batch, u=u_train_norm_batch,
                                                                                                        theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                                        min_cons_hard_x_step=self.min_cons_hard_x_step_norm_batch,
                                                                                                        max_cons_hard_x_step=self.max_cons_hard_x_step_norm_batch,
                                                                                                        min_cons_hard_y_step=self.min_cons_hard_y_step_norm_batch,
                                                                                                        max_cons_hard_y_step=self.max_cons_hard_y_step_norm_batch,
                                                                                                        min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                                        max_cons_hard_x_b_step=self.x_b_step_ub)
            
            mu_x_f_sim_neural_batch = torch.mean(x_f_sim_neural_batch, dim=1)
            mu_x_b_sim_neural_batch = torch.mean(x_b_sim_neural_batch, dim=1)
            std_x_f_sim_neural_batch = torch.std(x_f_sim_neural_batch, dim=1)
            std_x_b_sim_neural_batch = torch.std(x_b_sim_neural_batch, dim=1)

            penalty_mu = self.penalty_coef_dict["mu_x_f"] * self.penalty_loss_dict["mu_x_f"](mu_x_f_sim_neural_batch, 0) + self.penalty_coef_dict["mu_x_b"] * self.penalty_loss_dict["mu_x_b"](mu_x_b_sim_neural_batch, 0)
            penalty_std = self.penalty_coef_dict["sigma_x_f"] * self.penalty_loss_dict["sigma_x_f"](std_x_f_sim_neural_batch, 1) + self.penalty_coef_dict["sigma_x_b"] *  self.penalty_loss_dict["sigma_x_b"](std_x_b_sim_neural_batch, 1)
            penalty = penalty_mu + penalty_std
            
            loss_neural_train_batch, loss_neural_train_batch_nominal = self.loss_dict["neural loss"](y_train_norm_batch, y_sim_neural_batch, sim_obj=simulator_augmented)
            loss_neural_train_batch_penalized = loss_neural_train_batch + penalty
            loss_neural_train_batch_penalized.backward()
            optimizer_neural.step()
            
            batch_pop = batch_pop + u_train_norm_batch.shape[1]
            loss_neural_total_train = loss_neural_total_train + loss_neural_train_batch_nominal * u_train_norm_batch.shape[1]
            
            if batch_idx == 0:
                x_f_0_est_sim_neural = x_f_0_est_neural_batch
            
            if self.dev_tool_animate and self.batch_size == 1:
                self.x_f_sim_train_neural_history[epoch_neural] = x_f_sim_neural_batch
                self.x_b_sim_train_neural_history[epoch_neural] = x_b_sim_neural_batch
        
        return loss_neural_total_train/batch_pop, x_f_0_est_sim_neural

    def evaluate_neural(self, simulator_augmented, theta_f_fitted, theta_h_fitted, x_b_0_batch, estimator, datascaler, val_loader):
        simulator_augmented.eval()
        
        loss_neural_total_val = 0
        batch_pop = 0
        with torch.no_grad():
            for batch_idx, (u_val_norm_batch, y_val_norm_batch) in enumerate(val_loader):
                u_val_norm_batch, y_val_norm_batch = u_val_norm_batch.to(self.device), y_val_norm_batch.to(self.device)
                
                x_f_0_est_val_neural = estimator(u=u_val_norm_batch, y=y_val_norm_batch, mean=datascaler.x_mean, std=datascaler.x_std)
                
                if self.augmentation == "Exact" or self.augmentation ==  "Exact - NonNormalized":
                    x_f_sim_val_neural, x_b_sim_val_neural, y_sim_val_neural = simulator_augmented(x_f_0=x_f_0_est_val_neural, x_b_0=x_b_0_batch, u=u_val_norm_batch, 
                                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm_batch,
                                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm_batch,
                                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm_batch,
                                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm_batch,
                                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub)
                elif self.augmentation == "Robust" or self.augmentation ==  "Robust - NonNormalized":
                    x_f_sim_val_neural, x_b_sim_val_neural, y_sim_val_neural, x_p_sim_val_neural = simulator_augmented(x_f_0=x_f_0_est_val_neural, x_b_0=x_b_0_batch, u=u_val_norm_batch, 
                                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub)
                
                elif self.augmentation == "Exact - MANAM":
                    x_f_sim_val_neural, x_b_sim_val_neural, y_sim_val_neural = simulator_augmented(x_f_0=x_f_0_est_val_neural, x_b_0=x_b_0_batch, u=u_val_norm_batch, 
                                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm_batch,
                                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm_batch,
                                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm_batch,
                                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm_batch,
                                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub,
                                                                                                    adaptive_gamma_eval=self.adaptive_gamma_eval)
                elif self.augmentation == "Robust - MANAM":
                    x_f_sim_val_neural, x_b_sim_val_neural, y_sim_val_neural, x_p_sim_val_neural = simulator_augmented(x_f_0=x_f_0_est_val_neural, x_b_0=x_b_0_batch, u=u_val_norm_batch, 
                                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub,
                                                                                                    adaptive_gamma_eval=self.adaptive_gamma_eval)

                loss_neural_val_batch, loss_neural_val_nominal_batch = self.loss_dict["neural loss"](y_val_norm_batch, y_sim_val_neural, sim_obj=simulator_augmented)
                batch_pop = batch_pop + u_val_norm_batch.shape[1]
                loss_neural_total_val = loss_neural_total_val + loss_neural_val_nominal_batch * u_val_norm_batch.shape[1]
                
                if batch_idx == 0:
                    x_f_0_est_val_neural = x_f_0_est_val_neural
                    
        return loss_neural_total_val/batch_pop, x_f_0_est_val_neural
    
    def sim_evaluate_neural(self, simulator_augmented, theta_f_fitted, theta_h_fitted, datascaler, x_f_0_est_sim_neural, x_f_0_est_val_neural):
        with torch.no_grad():
            
            if self.augmentation == "Exact" or self.augmentation ==  "Exact - NonNormalized":
                x_f_sim_neural, x_b_sim_neural, y_sim_neural = simulator_augmented(x_f_0=x_f_0_est_sim_neural,
                                                                                x_b_0=self.x_b_0,
                                                                                u=self.u_train_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted,
                                                                                theta_h=theta_h_fitted,
                                                                                datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub)
                
                x_f_val_neural, x_b_val_neural, y_val_neural = simulator_augmented(x_f_0=x_f_0_est_val_neural,
                                                                                    x_b_0=self.x_b_0,
                                                                                    u=self.u_val_norm.view(1, -1, self.n_u),
                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub)
                x_p_sim_neural = x_f_sim_neural
                x_p_val_neural = x_f_val_neural
            
            elif self.augmentation == "Robust" or self.augmentation ==  "Robust - NonNormalized":
                x_f_sim_neural, x_b_sim_neural, y_sim_neural, x_p_sim_neural = simulator_augmented(x_f_0=x_f_0_est_sim_neural, x_b_0=self.x_b_0,
                                                                                u=self.u_train_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub)
                
                x_f_val_neural, x_b_val_neural, y_val_neural, x_p_val_neural = simulator_augmented(x_f_0=x_f_0_est_sim_neural, x_b_0=self.x_b_0,
                                                                                u=self.u_val_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub)
                
            elif self.augmentation == "Exact - MANAM":
                x_f_sim_neural, x_b_sim_neural, y_sim_neural = simulator_augmented(x_f_0=x_f_0_est_sim_neural,
                                                                                x_b_0=self.x_b_0,
                                                                                u=self.u_train_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted,
                                                                                theta_h=theta_h_fitted,
                                                                                datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub,
                                                                                adaptive_gamma_eval=self.adaptive_gamma_eval)
                
                x_f_val_neural, x_b_val_neural, y_val_neural = simulator_augmented(x_f_0=x_f_0_est_val_neural,
                                                                                    x_b_0=self.x_b_0,
                                                                                    u=self.u_val_norm.view(1, -1, self.n_u),
                                                                                    theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                    min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                    max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                    min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                    max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                    min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                    max_cons_hard_x_b_step=self.x_b_step_ub,
                                                                                    adaptive_gamma_eval=self.adaptive_gamma_eval)
                x_p_sim_neural = x_f_sim_neural
                x_p_val_neural = x_f_val_neural
            
            elif self.augmentation == "Robust - MANAM":
                x_f_sim_neural, x_b_sim_neural, y_sim_neural, x_p_sim_neural = simulator_augmented(x_f_0=x_f_0_est_sim_neural, x_b_0=self.x_b_0,
                                                                                u=self.u_train_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub,
                                                                                adaptive_gamma_eval=self.adaptive_gamma_eval)
                
                x_f_val_neural, x_b_val_neural, y_val_neural, x_p_val_neural = simulator_augmented(x_f_0=x_f_0_est_sim_neural, x_b_0=self.x_b_0,
                                                                                u=self.u_val_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub,
                                                                                adaptive_gamma_eval=self.adaptive_gamma_eval)
                
                
        return (x_f_sim_neural, x_b_sim_neural, y_sim_neural, x_p_sim_neural,
                x_f_val_neural, x_b_val_neural, y_val_neural, x_p_val_neural)
    
    
    def test_neural(self, simulator_augmented, theta_f_fitted, theta_h_fitted, datascaler, x_f_0, x_b_0):
        with torch.no_grad():
            if self.augmentation == "Exact" or self.augmentation == "Exact - MANAM" or self.augmentation ==  "Exact - NonNormalized":
                x_f_test_neural, x_b_test_neural, y_test_neural = simulator_augmented(x_f_0=x_f_0, x_b_0=x_b_0,
                                                                                u=self.u_test_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub)
                x_p_test_neural = x_f_test_neural
            
            elif self.augmentation == "Robust" or self.augmentation == "Robust - MANAM" or self.augmentation ==  "Robust - NonNormalized":
                x_f_test_neural, x_b_test_neural, y_test_neural, x_p_test_neural = simulator_augmented(x_f_0=x_f_0, x_b_0=x_b_0,
                                                                                u=self.u_test_norm.view(1, -1, self.n_u),
                                                                                theta_f=theta_f_fitted, theta_h=theta_h_fitted, datascaler=datascaler,
                                                                                min_cons_hard_x_step=self.min_cons_hard_x_step_norm,
                                                                                max_cons_hard_x_step=self.max_cons_hard_x_step_norm,
                                                                                min_cons_hard_y_step=self.min_cons_hard_y_step_norm,
                                                                                max_cons_hard_y_step=self.max_cons_hard_y_step_norm,
                                                                                min_cons_hard_x_b_step=self.x_b_step_lb,
                                                                                max_cons_hard_x_b_step=self.x_b_step_ub)
        return x_f_test_neural, x_b_test_neural, y_test_neural, x_p_test_neural
    
    def set_neural_components(self, Binary_f, Binary_g, Binary_h):
        
        if self.augmentation ==  "Exact - NonNormalized" or self.augmentation ==  "Robust - NonNormalized":
            f = PhysicalStateUpdate(self.f_userdefined, self.device)
            h = PhysicalOutput(self.h_userdefined, self.device)
        else:
            f = PhysicalStateUpdateNormalized(self.f_userdefined, self.device)
            h = PhysicalOutputNormalized(self.h_userdefined, self.device)
        
        f_b = FANSNeuralStateUpdate(n_x_f=self.n_x_f, n_x_b=self.n_x_b, n_x_o=self.n_x_f,
                                    Binary_mat=Binary_f, n_u=self.n_u, n_feat=self.n_feat,
                                    num_hidden_layers=self.num_hidden_layers, init_std=self.init_std,
                                    neuron_list=self.neuron_dict["f_b"], act_fn=self.act_fn_dict["f_b"], include_bias=self.bias_dict["f_b"],
                                    device=self.device)
        
        g_b = FANSNeuralStateUpdate(n_x_f=self.n_x_f, n_x_b=self.n_x_b, n_x_o=self.n_x_b,
                                    Binary_mat=Binary_g, n_u=self.n_u, n_feat=self.n_feat,
                                    num_hidden_layers=self.num_hidden_layers, init_std=self.init_std,
                                    neuron_list=self.neuron_dict["g_b"], act_fn=self.act_fn_dict["g_b"], include_bias=self.bias_dict["g_b"],
                                    device=self.device)
        
        h_b = FANSNeuralOutputUpdate(n_x_f=self.n_x_f, n_x_b=self.n_x_b, n_x_o=self.n_y,
                                    Binary_mat=Binary_h, n_feat=self.n_feat,
                                    num_hidden_layers=self.num_hidden_layers, init_std=self.init_std,
                                    neuron_list=self.neuron_dict["h_b"], act_fn=self.act_fn_dict["h_b"], include_bias=self.bias_dict["h_b"],
                                    device=self.device)
        
        return f, h, f_b, g_b, h_b
    
    
    def set_simulator_augmented(self, f, h, f_b, g_b, h_b, datascaler):
        optimizer_neural_params = []
        if self.augmentation == "Exact":
            simulator_augmented = StateSpaceSimulatorAugmented(f=f, f_b=f_b, g_b=g_b, h=h, h_b=h_b, device=self.device)
            
        elif self.augmentation == "Robust":
            simulator_augmented = RobustStateSpaceSimulatorAugmented(f=f, f_b=f_b, g_b=g_b, h=h, h_b=h_b, device=self.device)
            
        elif self.augmentation == "Exact - NonNormalized":
            simulator_augmented = StateSpaceSimulatorAugmented_Non_normalized(f=f, f_b=f_b, g_b=g_b, h=h, h_b=h_b, device=self.device)
            
        elif self.augmentation == "Robust - NonNormalized":
            simulator_augmented = RobustStateSpaceSimulatorAugmented_Non_normalized(f=f, f_b=f_b, g_b=g_b, h=h, h_b=h_b, device=self.device)
            
        elif self.augmentation == "Exact - MANAM":
            simulator_augmented = StateSpaceSimulatorAugmented_MANAM(f=f, f_b=f_b, g_b=g_b, h=h, h_b=h_b, device=self.device,
                                                                        gamma_power=self.gamma_power,
                                                                        display_gamma_condition_stat=self.display_gamma_condition_stat)
            
            datascaler.x_mean = datascaler.x_mean.requires_grad_(True)
            datascaler.x_std = datascaler.x_std.requires_grad_(True)
            optimizer_neural_params.append({"params": datascaler.x_mean, "lr":self.lr_mu_x})
            optimizer_neural_params.append({"params": datascaler.x_std, "lr":self.lr_sigma_x})
            
        elif self.augmentation == "Robust - MANAM":
            simulator_augmented = RobustStateSpaceSimulatorAugmented_MANAM(f=f, f_b=f_b, g_b=g_b, h=h, h_b=h_b, device=self.device,
                                                                           gamma_power=self.gamma_power,
                                                                           display_gamma_condition_stat=self.display_gamma_condition_stat)
            
            datascaler.x_mean = datascaler.x_mean.requires_grad_(True)
            datascaler.x_std = datascaler.x_std.requires_grad_(True)
            optimizer_neural_params.append({"params": datascaler.x_mean, "lr":self.lr_mu_x})
            optimizer_neural_params.append({"params": datascaler.x_std, "lr":self.lr_sigma_x})
            
        return simulator_augmented, optimizer_neural_params
    
    
    def set_optimizer_neural(self, optimizer_neural_params, f_b, g_b, h_b, estimator):
        optimizer_neural_params.append({"params": estimator.parameters(), "lr":self.lr_x_f_0})
        if self.include_f_b:
            optimizer_neural_params.append({"params": f_b.parameters(), "lr": self.lr_neuralnet["f_b"]})
        if self.include_g_b:
            optimizer_neural_params.append({"params": g_b.parameters(), "lr": self.lr_neuralnet["g_b"]})
        if self.include_h_b:
            optimizer_neural_params.append({"params": h_b.parameters(), "lr": self.lr_neuralnet["h_b"]})
        optimizer_neural =  self.optim_alg_dict["neural optimization algorithm"](optimizer_neural_params)
        return optimizer_neural
    
    
    def set_estimator(self):
        if "Physical Estimate" in self.iv_est_dict.keys():
            estimator = PhysicalStateEstimator(x_f_0_phys=self.x0_phys, n_x_f=self.n_x_f, n_x_b=self.n_x_b, batch_size=self.batch_size)
        
        estimator = estimator.to(self.device)
        
            
        return estimator
    
    def physical_model_fit(self):
        # Setting up the dataloader
        train_loader = self.train_loader
        val_loader = self.val_loader
        
        # Creating Physical-Component Objects
        f_phys = PhysicalStateUpdate(self.f_userdefined, self.device)
        h_phys = PhysicalOutput(self.h_userdefined, self.device)
        
        # Creating Physical-Simulator
        simulator_phys = StateSpaceSimulator(f=f_phys,
                                            h=h_phys,
                                            device=self.device)

        theta_f = torch.tensor(self.theta_f_initial_guess, dtype=torch.float32, device=self.device).requires_grad_(True)
        theta_h = torch.tensor(self.theta_h_initial_guess, dtype=torch.float32, device=self.device).requires_grad_(True)
        x0_phys = torch.reshape(torch.tensor(self.x0_phys_initial_guess, dtype=torch.float32, device=self.device), (1, self.n_x)).requires_grad_(True)

        optimizer_phys = self.optim_alg_dict["physical optimization algorithm"](
            [{"params": x0_phys, "lr": self.lr_x0_phys},
            {"params": theta_f, "lr": self.lr_theta_f},
            {"params": theta_h, "lr": self.lr_theta_h}]
            )

        loss_history_train_phys = []
        loss_history_val_phys = []
        x0_phys_batch = x0_phys.repeat(self.batch_size, 1)
        
        print("|||||||||||||||||||||||||||||||||Physical Model Training Phase|||||||||||||||||||||||||||||||||")
        print(f"Training: Batch size: {self.batch_size} - Number of Parameter Update Per Epoch: {len(self.train_loader)}")
        print(f"Validation: Batch size: {self.batch_size} - Number of Parameter Update Per Epoch: {len(self.val_loader)}")    
        for epoch_phys in range(self.epochs_phys):
            
            loss_phys_total_train, theta_f, theta_h = self.train_phys(simulator_phys=simulator_phys,
                                                                    theta_f=theta_f, theta_h=theta_h, x0_phys_batch=x0_phys_batch,
                                                                    optimizer_phys=optimizer_phys, train_loader=train_loader)
            if (epoch_phys+1) % self.display_step == 0:
                
                with torch.no_grad():
                    loss_phys_total_val = self.evaluate_phys(simulator_phys = simulator_phys,
                                                            theta_f = theta_f, theta_h = theta_h,
                                                            x0_phys_batch = x0_phys_batch,
                                                            val_loader = val_loader)
                    
                    train_rmse_phys = torch.sqrt(loss_phys_total_train).item()
                    val_rmse_phys = torch.sqrt(loss_phys_total_val).item()
                    
                    loss_history_train_phys, loss_history_val_phys =  training_progress_display(train_loss=train_rmse_phys,
                                                                                                val_loss=val_rmse_phys,
                                                                                                loss_history_train=loss_history_train_phys,
                                                                                                loss_history_val=loss_history_val_phys,
                                                                                                epoch=epoch_phys, epochs=self.epochs_phys, display_val=True)
        
        simulator_phys.move_to_device(torch.device('cpu'))
        
        x_sim_phys, y_sim_phys = simulator_phys(x0_phys.to("cpu"),
                                                self.u_train.view(1, -1, self.n_u),
                                                theta_f=theta_f.to("cpu"),
                                                theta_h=theta_h.to("cpu"),
                                                min_cons_hard_x_step=self.min_cons_hard_x_step.to("cpu"),
                                                max_cons_hard_x_step=self.max_cons_hard_x_step.to("cpu"),
                                                min_cons_hard_y_step=self.min_cons_hard_y_step.to("cpu"),
                                                max_cons_hard_y_step=self.max_cons_hard_y_step.to("cpu"))
        
        _, y_val_phys = simulator_phys(x0_phys.to("cpu"),
                                        self.u_val.view(1, -1, self.n_u),
                                        theta_f=theta_f.to("cpu"),
                                        theta_h=theta_h.to("cpu"),
                                        min_cons_hard_x_step=self.min_cons_hard_x_step.to("cpu"),
                                        max_cons_hard_x_step=self.max_cons_hard_x_step.to("cpu"),
                                        min_cons_hard_y_step=self.min_cons_hard_y_step.to("cpu"),
                                        max_cons_hard_y_step=self.max_cons_hard_y_step.to("cpu"))
        
        _, y_test_phys = simulator_phys(x0_phys.to("cpu"),
                                        self.u_test.view(1, -1, self.n_u),
                                        theta_f=theta_f.to("cpu"),
                                        theta_h=theta_h.to("cpu"),
                                        min_cons_hard_x_step=self.min_cons_hard_x_step.to("cpu"),
                                        max_cons_hard_x_step=self.max_cons_hard_x_step.to("cpu"),
                                        min_cons_hard_y_step=self.min_cons_hard_y_step.to("cpu"),
                                        max_cons_hard_y_step=self.max_cons_hard_y_step.to("cpu"))
        
        print('Estimated Values from the un-normalized optimization:')  
        print(f'Estimated Value for x0: {x0_phys} \t Estimated Value for theta_f: {theta_f}')
        print(f'Estimated Value for theta_h: {theta_h}')
        
        # Detaching the Parameters
        theta_f = theta_f.detach()
        theta_h = theta_h.detach()
        x0_phys = x0_phys.detach()
        x_sim_phys = x_sim_phys.detach()
        
        self.datascaler.x_mean = torch.mean(x_sim_phys , 1)
        self.datascaler.x_std = torch.std(x_sim_phys , 1)
        self.datascaler.move_to_device(self.device)
        # x0_phys_norm = self.datascaler.zscore_normalize(x0_phys, self.datascaler.x_mean, self.datascaler.x_std)
        
        # Retrieving the normalized hard constraints on states and outputs
        self.min_cons_hard_x_step_norm = self.datascaler.zscore_normalize(self.min_cons_hard_x_step, self.datascaler.x_mean, self.datascaler.x_std)
        self.max_cons_hard_x_step_norm = self.datascaler.zscore_normalize(self.max_cons_hard_x_step, self.datascaler.x_mean, self.datascaler.x_std)
        self.min_cons_hard_y_step_norm = self.datascaler.zscore_normalize(self.min_cons_hard_y_step, self.datascaler.y_mean, self.datascaler.y_std)
        self.max_cons_hard_y_step_norm = self.datascaler.zscore_normalize(self.max_cons_hard_y_step, self.datascaler.y_mean, self.datascaler.y_std)

        self.y_sim_phys = y_sim_phys.detach()
        self.y_val_phys =  y_val_phys.detach()
        self.y_test_phys =  y_test_phys.detach()
        self.loss_history_train_phys = loss_history_train_phys
        self.loss_history_val_phys = loss_history_val_phys
        
        return theta_f, theta_h, x0_phys, simulator_phys, y_val_phys
    
    def augmented_model_fit(self):
        
        loss_history_train_neural = []
        loss_history_val_neural = []
        
        theta_f, theta_h, x0_phys, _, _ = self.physical_model_fit()
        self.x0_phys = x0_phys
        
        # Setting up the dataloader
        train_norm_loader = self.train_norm_loader
        val_norm_loader = self.val_norm_loader
        
        self.n_x_f = self.n_x
        theta_f_fitted = theta_f
        theta_h_fitted = theta_h
        
        x_b_0_batch = self.x_b_0.repeat(self.batch_size, 1)
        
        Binary_f = self.Binary_state[:self.n_x_f, :]
        Binary_g = self.Binary_state[self.n_x_f:, :]
        Binary_h = self.Binary_output
        
        f, h, f_b, g_b, h_b = self.set_neural_components(Binary_f=Binary_f, Binary_g=Binary_g, Binary_h=Binary_h)
        
        simulator_augmented, optimizer_neural_params = self.set_simulator_augmented(f=f,
                                                                                    h=h,
                                                                                    f_b=f_b,
                                                                                    g_b=g_b,
                                                                                    h_b=h_b,
                                                                                    datascaler=self.datascaler)
        
        estimator = self.set_estimator()
        
        optimizer_neural = self.set_optimizer_neural(optimizer_neural_params, f_b, h_b, g_b, estimator)
        
        self.datascaler.move_to_device(self.device)
        
        print("|||||||||||||||||||||||||||||||||Augmented Model Training Phase|||||||||||||||||||||||||||||||||")
        print(f"Training: Batch size: {self.batch_size} - Number of Parameter Update Per Epoch: {len(self.train_norm_loader)}")
        print(f"Validation: Batch size: {self.batch_size} - Number of Parameter Update Per Epoch: {len(self.val_norm_loader)}")
        
        self.x_b_step_lb = -self.x_b_exploration_scale
        self.x_b_step_ub = +self.x_b_exploration_scale
        self.min_cons_hard_x_step_norm_batch = self.min_cons_hard_x_step_norm.repeat(self.batch_size, 1)
        self.max_cons_hard_x_step_norm_batch = self.max_cons_hard_x_step_norm.repeat(self.batch_size, 1)
        self.min_cons_hard_y_step_norm_batch = self.min_cons_hard_y_step_norm.repeat(self.batch_size, 1)
        self.max_cons_hard_y_step_norm_batch = self.max_cons_hard_y_step_norm.repeat(self.batch_size, 1)
        
        self.x_f_sim_train_neural_history = []
        self.x_b_sim_train_neural_history = []
        
        if self.dev_tool_animate and self.batch_size == 1:
            self.x_f_sim_train_neural_history = torch.empty([self.epochs_neural, self.batch_size, self.y_train.size(0) ,self.n_x_f])
            self.x_b_sim_train_neural_history = torch.empty([self.epochs_neural, self.batch_size, self.y_train.size(0) ,self.n_x_b])
        
        for epoch_neural in range(self.epochs_neural):
            # print(f"epoch_neural: {epoch_neural}")
            loss_neural_total_train, x_f_0_est_sim_neural_init_batch = self.train_neural(simulator_augmented=simulator_augmented,
                                                            theta_f_fitted=theta_f_fitted,
                                                            theta_h_fitted=theta_h_fitted,
                                                            x_b_0_batch=x_b_0_batch,
                                                            estimator=estimator,
                                                            datascaler=self.datascaler,
                                                            optimizer_neural=optimizer_neural,
                                                            train_loader=train_norm_loader,
                                                            epoch_neural=epoch_neural)
            
            
            if (epoch_neural+1) % self.display_step == 0:
                with torch.no_grad():
                    loss_neural_total_val, x_f_0_est_val_neural_init_batch = self.evaluate_neural(simulator_augmented=simulator_augmented,
                                                                theta_f_fitted=theta_f_fitted,
                                                                theta_h_fitted=theta_h_fitted,
                                                                x_b_0_batch=x_b_0_batch,
                                                                estimator=estimator,
                                                                datascaler=self.datascaler,
                                                                val_loader=val_norm_loader)
                    
                    train_rmse_neural_ref = torch.sqrt(loss_neural_total_train)*(self.datascaler.y_std).item()
                    val_rmse_neural_ref = torch.sqrt(loss_neural_total_val)*(self.datascaler.y_std).item()    
                    
                    loss_history_train_neural, loss_history_val_neural =  training_progress_display(train_loss=train_rmse_neural_ref,
                                                                                                    val_loss=val_rmse_neural_ref,
                                                                                                    loss_history_train=loss_history_train_neural,
                                                                                                    loss_history_val=loss_history_val_neural,
                                                                                                    epoch=epoch_neural, epochs=self.epochs_neural)
        
        # Setting the estimated initial value of states using the estimator
        x_f_0_est_sim_neural = x_f_0_est_sim_neural_init_batch[0:1, :]
        x_f_0_est_val_neural = x_f_0_est_val_neural_init_batch[0:1, :]
        
        # Moving data/model to the cpu for the whole range simulation, validation and test
        self.min_cons_hard_x_step_norm = self.min_cons_hard_x_step_norm.to('cpu')
        self.max_cons_hard_x_step_norm = self.max_cons_hard_x_step_norm.to('cpu')
        self.min_cons_hard_y_step_norm = self.min_cons_hard_y_step_norm.to('cpu')
        self.max_cons_hard_y_step_norm = self.max_cons_hard_y_step_norm.to('cpu')
        self.x_b_step_lb = self.x_b_step_lb.to('cpu')
        self.x_b_step_ub = self.x_b_step_ub.to('cpu')
        self.x_b_0 = self.x_b_0.to('cpu')
        self.datascaler.move_to_device(torch.device('cpu'))
        theta_f_fitted = theta_f_fitted.to('cpu')
        theta_h_fitted = theta_h_fitted.to('cpu')
        x_f_0_est_sim_neural = x_f_0_est_sim_neural.to('cpu')
        x_f_0_est_val_neural = x_f_0_est_val_neural.to('cpu')
        simulator_augmented.move_to_device(torch.device('cpu'))
        
        (x_f_sim_neural, x_b_sim_neural, y_sim_neural, x_p_sim_neural,
        x_f_val_neural, x_b_val_neural, y_val_neural, x_p_val_neural) = self.sim_evaluate_neural(simulator_augmented=simulator_augmented,
                                                                                                theta_f_fitted=theta_f_fitted,
                                                                                                theta_h_fitted=theta_h_fitted,
                                                                                                datascaler=self.datascaler,
                                                                                                x_f_0_est_sim_neural=x_f_0_est_sim_neural,
                                                                                                x_f_0_est_val_neural=x_f_0_est_val_neural)

        
        x_f_test_neural, x_b_test_neural, y_test_neural, x_p_test_neural = self.test_neural(simulator_augmented=simulator_augmented,
                                                                                            theta_f_fitted=theta_f_fitted,
                                                                                            theta_h_fitted=theta_h_fitted,
                                                                                            datascaler=self.datascaler,
                                                                                            x_f_0=x_f_0_est_sim_neural,
                                                                                            x_b_0=self.x_b_0)
        
        
        self.y_sim_phys=self.y_sim_phys.to('cpu').detach()
        self.y_val_phys=self.y_val_phys.to('cpu').detach()
        self.y_test_phys=self.y_test_phys.to('cpu').detach()
        
        self.y_sim_neural = y_sim_neural.detach()
        self.y_val_neural = y_val_neural.detach()
        self.y_test_neural = y_test_neural.detach()
        self.x_b_sim_neural = x_b_sim_neural.detach()
        self.x_b_val_neural = x_b_val_neural.detach()
        self.x_b_test_neural = x_b_test_neural.detach()
        self.x_f_sim_neural = x_f_sim_neural.detach()
        self.x_f_val_neural = x_f_val_neural.detach()
        self.x_f_test_neural = x_f_test_neural.detach()
        self.loss_history_train_neural = [element.to("cpu").detach() for element in loss_history_train_neural]
        self.loss_history_val_neural = [element.to("cpu").detach() for element in loss_history_val_neural]
        
        self.x_f_test_neural_denorm = self.datascaler.zscore_denormalize(x_f_test_neural, self.datascaler.x_mean, self.datascaler.x_std)
        self.x_p_test_neural_denorm = self.datascaler.zscore_denormalize(x_p_test_neural, self.datascaler.x_mean, self.datascaler.x_std)
        self.y_test_neural_denorm = self.datascaler.zscore_denormalize(y_test_neural, self.datascaler.y_mean, self.datascaler.y_std)
        self.y_val_neural_denorm = self.datascaler.zscore_denormalize(self.y_val_neural, self.datascaler.y_mean, self.datascaler.y_std)
        
        augmodel = AugModel(simulator=simulator_augmented,
                            x_f_0=x_f_0_est_sim_neural,
                            x_b_0=self.x_b_0,
                            theta_f=theta_f_fitted,
                            theta_h=theta_h_fitted,
                            datascaler=self.datascaler,
                            min_cons_hard_x_b_step=self.x_b_step_lb,
                            max_cons_hard_x_b_step=self.x_b_step_ub,
                            min_cons_hard_x_step_norm=self.min_cons_hard_x_step_norm,
                            max_cons_hard_x_step_norm=self.max_cons_hard_x_step_norm,
                            min_cons_hard_y_step_norm=self.min_cons_hard_y_step_norm,
                            max_cons_hard_y_step_norm=self.max_cons_hard_y_step_norm,
                            augmentation=self.augmentation)
        
        return augmodel, self.y_test_neural_denorm
    
    def save_results(self):
        
        result_dict = save_result(datascaler=self.datascaler,
                                y_train=self.y_train,
                                y_val=self.y_val,
                                y_test=self.y_test,
                                y_sim_phys=self.y_sim_phys,
                                y_val_phys=self.y_val_phys,
                                y_test_phys=self.y_test_phys,
                                y_sim_neural=self.y_sim_neural,
                                y_val_neural=self.y_val_neural,
                                y_test_neural=self.y_test_neural,
                                x_b_sim_neural=self.x_b_sim_neural,
                                x_b_val_neural=self.x_b_val_neural,
                                x_b_test_neural=self.x_b_test_neural,
                                x_f_sim_neural=self.x_f_sim_neural,
                                x_f_val_neural=self.x_f_val_neural,
                                x_f_test_neural=self.x_f_test_neural)
        return result_dict
        
    def model_fit_eval(self):
        
        result_plot(datascaler=self.datascaler,
                    y_train=self.y_train,
                    y_val=self.y_val,
                    y_test=self.y_test,
                    y_sim_phys=self.y_sim_phys,
                    y_val_phys=self.y_val_phys,
                    y_test_phys=self.y_test_phys,
                    y_sim_neural=self.y_sim_neural,
                    y_val_neural=self.y_val_neural,
                    y_test_neural=self.y_test_neural,
                    x_b_sim_neural=self.x_b_sim_neural,
                    x_b_val_neural=self.x_b_val_neural,
                    x_b_test_neural=self.x_b_test_neural,
                    x_f_sim_neural=self.x_f_sim_neural,
                    x_f_val_neural=self.x_f_val_neural,
                    x_f_test_neural=self.x_f_test_neural)
