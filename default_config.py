default_params = {
    "batch_size": 0,                                    #Batch Size (0: one iteration per epoch - no mini batch)
    "lr_x0_phys": 1e-3,                                 #Learning Rate (Initial value of Un-normalized Physical States)
    "lr_theta_f": 1e-3,                                 #Learning Rate for Theta_f (Initial value of Un-normalized Optimization)
    "lr_theta_h": 1e-3,                                 #Learning Rate for Theta_h (Initial value of Un-normalized Optimization)
    "lr_x0_phys_norm": 1e-3,                            #Learning Rate for (Initial value of normalized Physical States)
    "lr_theta_f_norm": 1e-3,                            #Learning Rate for Theta_f (Initial value of normalized Optimization)
    "lr_theta_h_norm": 1e-3,                            #Learning Rate for Theta_h (Initial value of normalized Optimization)
    "lr_x_b_0": 1e-3,                                   #Learning Rate for the Initial value of Augmented Neural States
    "lr_x_f_0": 1e-3,                                   #Learning Rate for the Initial value of Augmented Physical States
    "lr_f_b": 1e-5,                                     #Learning Rate for f_b (Initial value of the neural network f_b)
    "lr_g_b": 3e-2,                                     #Learning Rate for g_b (Initial value of the neural network g_b)
    "lr_h_b": 3e-2,                                     #Learning Rate for h_b (Initial value of the neural network h_b)
    "epochs_phys": 100,                                 #Number of epochs for Un-normalized Optimization
    "epochs_phys_norm": 100,                            #Number of epochs for normalized Optimization
    "epochs_neural": 300,                               #Number of epochs for Neural Training (Augmented Model/Neural Only Model)
    "n_feat": 8,                                        #Number of features for Neural Layers
    "num_hidden_layers": 1,                             #Number of Hidden Layers (1: Shallow Neural Network)
    "display_step": 10,                                 #Steps taken to display the optimization/training progress
    "include_f_b": True,                                #whether to include f_b or not
    "include_g_b": True,                                #whether to include g_b or not
    "include_h_b": True,                                #whether to include b_b or not
    "saved_model_file_name": "training_model.pt",       #Name of the file to store the model to
    "x_b_exploration_scale": 10000,                     #Set the exploration bound which the neural states are limited to
    }