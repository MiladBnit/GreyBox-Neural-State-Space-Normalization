# %% [markdown]
# ## SLIMPEC Task 1.1 Cascaded Tanks Benchmark - Functions

# %% [markdown]
# #### Imports

# %%
import torch
from torch import nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rcParams
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np

# %% [markdown]
# ##### Result Storage:
def save_result(datascaler,
                y_train,
                y_val,
                y_test,
                y_sim_phys,
                y_val_phys,
                y_test_phys,
                y_sim_neural,
                y_val_neural,
                y_test_neural,
                x_b_sim_neural,
                x_b_val_neural,
                x_b_test_neural,
                x_f_sim_neural,
                x_f_val_neural,
                x_f_test_neural,
                ):
    
    r2score_train_phys = r2_score(y_train, torch.reshape(y_sim_phys, (y_train.size(0), -1)).detach().numpy())
    rmse_train_phys = root_mean_squared_error(y_train, torch.reshape(y_sim_phys, (y_train.size(0), -1)).detach().numpy())

    r2score_test_phys = r2_score(y_test, torch.reshape(y_test_phys, (y_test.size(0), -1)).detach().numpy())
    rmse_test_phys = root_mean_squared_error(y_test, torch.reshape(y_test_phys, (y_test.size(0), -1)).detach().numpy())

    y_sim_neural_denorm = datascaler.zscore_denormalize(y_sim_neural, datascaler.y_mean, datascaler.y_std)
    y_test_neural_denorm = datascaler.zscore_denormalize(y_test_neural, datascaler.y_mean, datascaler.y_std)

    y_sim_neural_denorm = y_sim_neural_denorm[0, :, :].detach().numpy()
    r2score_train_neural = r2_score(y_train, y_sim_neural_denorm)
    rmse_train_neural = root_mean_squared_error(y_train, y_sim_neural_denorm)

    y_test_neural_denorm = y_test_neural_denorm[0, :, :].detach().numpy()
    r2score_test_neural = r2_score(y_test, y_test_neural_denorm)
    rmse_test_neural = root_mean_squared_error(y_test, y_test_neural_denorm)
    
    
    result_dict = {
                "datascaler": datascaler,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "y_sim_phys": y_sim_phys,
                "y_val_phys": y_val_phys,
                "y_test_phys": y_test_phys,
                "y_sim_neural": y_sim_neural,
                "y_val_neural": y_val_neural,
                "y_test_neural": y_test_neural,
                "x_b_sim_neural": x_b_sim_neural,
                "x_b_val_neural": x_b_val_neural,
                "x_b_test_neural": x_b_test_neural,
                "x_f_sim_neural": x_f_sim_neural,
                "x_f_val_neural": x_f_val_neural,
                "x_f_test_neural": x_f_test_neural,
                "r2score_train_phys": r2score_train_phys,
                "r2score_train_neural" : r2score_train_neural,
                "r2score_test_phys" : r2score_test_phys,
                "r2score_test_neural" : r2score_test_neural,
                "rmse_train_phys" : rmse_train_phys,
                "rmse_train_neural" : rmse_train_neural,
                "rmse_test_phys" : rmse_test_phys,
                "rmse_test_neural" : rmse_test_neural
                }
    
    return result_dict
# %%
def result_plot(datascaler,
                y_train,
                y_val,
                y_test,
                y_sim_phys,
                y_val_phys,
                y_test_phys,
                y_sim_neural,
                y_val_neural,
                y_test_neural,
                x_b_sim_neural,
                x_b_val_neural,
                x_b_test_neural,
                x_f_sim_neural,
                x_f_val_neural,
                x_f_test_neural,
                method_name="",
                save_figs=True
                ):
    
    n_x_f = x_f_test_neural.shape[2]
    n_x_b = x_b_test_neural.shape[2]
    T_test = x_f_test_neural.shape[1]
    T_train = x_f_sim_neural.shape[1]
    
    y_sim_phys = torch.reshape(y_sim_phys, (y_train.size(0), -1)).detach().numpy()
    r2score_train_phys = r2_score(y_train, y_sim_phys)
    rmse_train_phys = root_mean_squared_error(y_train, y_sim_phys)

    y_test_phys = torch.reshape(y_test_phys, (y_test.size(0), -1)).detach().numpy()
    r2score_test_phys = r2_score(y_test, y_test_phys)
    rmse_test_phys = root_mean_squared_error(y_test, y_test_phys)
    
    y_val_phys = torch.reshape(y_val_phys, (y_val.size(0), -1)).detach().numpy()
    r2score_val_phys = r2_score(y_val, y_val_phys)
    rmse_val_phys = root_mean_squared_error(y_val, y_val_phys)

    y_sim_neural_denorm = datascaler.zscore_denormalize(y_sim_neural, datascaler.y_mean, datascaler.y_std)
    y_val_neural_denorm = datascaler.zscore_denormalize(y_val_neural, datascaler.y_mean, datascaler.y_std)
    y_test_neural_denorm = datascaler.zscore_denormalize(y_test_neural, datascaler.y_mean, datascaler.y_std)
    x_f_test_neural_denorm = datascaler.zscore_denormalize(x_f_test_neural, datascaler.x_mean, datascaler.x_std)
    x_f_sim_neural_denorm = datascaler.zscore_denormalize(x_f_sim_neural, datascaler.x_mean, datascaler.x_std)

    y_sim_neural_denorm = y_sim_neural_denorm[0, :, :].detach().numpy()
    r2score_train_neural = r2_score(y_train, y_sim_neural_denorm)
    rmse_train_neural = root_mean_squared_error(y_train, y_sim_neural_denorm)

    y_val_neural_denorm = y_val_neural_denorm[0, :, :].detach().numpy()
    r2score_val_neural = r2_score(y_val, y_val_neural_denorm)
    rmse_val_neural = root_mean_squared_error(y_val, y_val_neural_denorm)
    
    y_test_neural_denorm = y_test_neural_denorm[0, :, :].detach().numpy()
    r2score_test_neural = r2_score(y_test, y_test_neural_denorm)
    rmse_test_neural = root_mean_squared_error(y_test, y_test_neural_denorm)

    x_f_test_neural_denorm = x_f_test_neural_denorm[0, :, :].detach().numpy()
    x_f_sim_neural_denorm = x_f_sim_neural_denorm[0, :, :].detach().numpy()
    
    x_b_test_neural = torch.reshape(x_b_test_neural[:, :, :], (y_test.size(0), -1)).detach().numpy()
    x_b_sim_neural = torch.reshape(x_b_sim_neural[:, :, :], (y_train.size(0), -1)).detach().numpy()

    print(f"                        | Physical Model | Augmented Model")
    print("-------------------------------------------------------------------------------------------")
    print(f"R2 Score in Training    |     {r2score_train_phys:.3f}      |      {r2score_train_neural:.3f}")
    print("-------------------------------------------------------------------------------------------")
    print(f"R2 Score in Validation  |     {r2score_val_phys:.3f}      |      {r2score_val_neural:.3f}")
    print("-------------------------------------------------------------------------------------------")
    print(f"R2 Score in Testing     |     {r2score_test_phys:.3f}      |      {r2score_test_neural:.3f}")
    print("-------------------------------------------------------------------------------------------")
    print(f"RMSE in Training        |     {rmse_train_phys:.3f}      |      {rmse_train_neural:.3f}")
    print("-------------------------------------------------------------------------------------------")
    print(f"RMSE in Validation      |     {rmse_val_phys:.3f}      |      {rmse_val_neural:.3f}")
    print("-------------------------------------------------------------------------------------------")
    print(f"RMSE in Testing         |     {rmse_test_phys:.3f}      |      {rmse_test_neural:.3f}")

    alpha = 0.15
    color_1 = "#228B22"
    # color_1 = "#006400"
    color_2 = "#FF4500"
    color_3 = "k"
    fs = 8
    lw_rr = 0.4
    cw = 3.5
    ar = 0.5
    dpi = 600
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
    
    fig = plt.figure(num=1, figsize = (cw,cw*ar), dpi=dpi)
    plt.plot(y_test_neural_denorm, label=method_name , linestyle="-", linewidth=2*lw_rr, zorder=2, color=color_1)
    plt.plot(y_test_phys, label="Baseline Model", linestyle="-", linewidth=2*lw_rr, zorder=1, color=color_2)
    plt.plot(y_test, label="True Value", linestyle=":", linewidth=2*lw_rr, zorder=3, color=color_3)
    # plt.title('Models Performance on Test Data')
    plt.xlabel('Time [step]')
    plt.ylabel('Output')
    plt.grid(True, linewidth=0.5)
    # plt.tight_layout()
    # fig.legend(prop={'family': 'serif', 'size': legend_fs, 'style': 'italic'}, loc='upper left', bbox_to_anchor=(0, 0.95, 1, 0.2), mode='expand', borderaxespad=0, ncol=4)
    # fig.legend(loc='upper left', frameon=False, fontsize=fs , ncol=3, bbox_to_anchor=(0, 1.05))
    fig.legend(loc='upper left', frameon=False, fontsize=fs , ncol=3)
    # if save_figs:
        # plt.savefig('plots/Testing_Fit_Plot.pdf', format='pdf', dpi=dpi, bbox_inches='tight') 
    plt.show()

    # fig = plt.figure(num=2, figsize = (cw,cw*ar), dpi=300) 
    # plt.plot(y_sim_neural_denorm, label="Model Fit by " + method_name, linestyle="-", linewidth=2*lw_rr, zorder=3, color=color_1)
    # plt.plot(y_sim_phys, label="Model Fit by Baseline Model", linestyle="-", linewidth=2*lw_rr, zorder=2, color=color_2)
    # plt.plot(y_train, label="True Value", linestyle=":", linewidth=2*lw_rr, zorder=1, color=color_3)
    # # plt.title('Model Fit on Training Data')
    # plt.xlabel('Time [step]')
    # plt.ylabel('Output')
    # plt.grid(True, linewidth=0.5)
    # # fig.legend(prop={'family': 'serif', 'style': 'italic'}, loc='upper left', bbox_to_anchor=(0, 0.95, 1, 0.2), mode='expand', borderaxespad=0, ncol=4)
    # fig.legend(loc='upper left', frameon=False, fontsize=fs , ncol=3, bbox_to_anchor=(0, 1.05))
    # # if save_figs:
    #     # plt.savefig('plots/Training_Fit_Plot.pdf', format='pdf', dpi=dpi, bbox_inches='tight')
    # plt.show()

    xhat_1 = torch.reshape(x_f_sim_neural[:, :, :], (y_train.size(0), -1)).detach().numpy()[:, 0]
    xhat_2 = torch.reshape(x_f_sim_neural[:, :, :], (y_train.size(0), -1)).detach().numpy()[:, 1]
    z_1 = x_b_sim_neural[:, 0]
    z_2 = x_b_sim_neural[:, 1]
    z_3 = x_b_sim_neural[:, 2]
    
    mean_xhat_1 = np.mean(xhat_1, axis=0)
    mean_xhat_2 = np.mean(xhat_2, axis=0)
    mean_z_1 = np.mean(z_1, axis=0)
    mean_z_2 = np.mean(z_2, axis=0)
    mean_z_3 = np.mean(z_3, axis=0)
    
    std_xhat_1 = np.std(xhat_1, axis=0)
    std_xhat_2 = np.std(xhat_2, axis=0)
    std_z_1 = np.std(z_1, axis=0)
    std_z_2 = np.std(z_2, axis=0)
    std_z_3 = np.std(z_3, axis=0)
    
    fig, axes = plt.subplots(n_x_f + n_x_b, 1, sharex=True, dpi=dpi, figsize = (cw,cw*ar*(1.5)))

    axes[0].plot(xhat_1, linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[0].hlines(mean_xhat_1, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[0].fill_between(np.arange(T_train), mean_xhat_1 - std_xhat_1 , mean_xhat_1 + std_xhat_1, alpha=alpha, color=color_1, zorder=1)
    axes[0].set_ylabel(r'${\hat{x}}_1$')
    axes[0].grid(True, linewidth=0.5)
    
    axes[1].plot(xhat_2, linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[1].hlines(mean_xhat_2, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[1].fill_between(np.arange(T_train), mean_xhat_2 - std_xhat_2 , mean_xhat_2 + std_xhat_2, alpha=alpha, color=color_1, zorder=1)
    axes[1].set_ylabel(r'${\hat{x}}_2$')
    axes[1].grid(True, linewidth=0.5)
    axes[1].set_yticks([-1, 0, 1])
    
    axes[2].plot(z_1, linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[2].hlines(mean_z_1, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[2].fill_between(np.arange(T_train), mean_z_1 - std_z_1 , mean_z_1 + std_z_1, alpha=alpha, color=color_1, zorder=1)   
    axes[2].set_ylabel(r'${\hat{z}}_1$')
    axes[2].grid(True, linewidth=0.5)
    axes[2].set_yticks([-1, 0, 1])
    
    axes[3].plot(z_2, linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[3].hlines(mean_z_2, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[3].fill_between(np.arange(T_train), mean_z_2 - std_z_2 , mean_z_2 + std_z_2, alpha=alpha, color=color_1, zorder=1)   
    axes[3].set_ylabel(r'${\hat{z}}_2$')
    axes[3].grid(True, linewidth=0.5)
    axes[3].set_yticks([-1, 0, 1])
    
    axes[4].plot(z_3, label=r'state', linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[4].hlines(mean_z_3, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, label=r'$\mu_{\hat{x}_{a}}$', zorder=2)
    axes[4].fill_between(np.arange(T_train), mean_z_3 - std_z_3 , mean_z_3 + std_z_3, alpha=alpha, color=color_1, zorder=1, label=r'$\mu_{\hat{x}_{a}} \pm \sigma_{\hat{x}_{a}}$')   
    axes[4].set_ylabel(r'${\hat{z}}_3$')
    axes[4].set_xlabel('Time [step]')
    axes[4].grid(True, linewidth=0.5)
    axes[4].set_yticks([-1, 0, 1])
    
    # fig.suptitle('Normalized State Trajectory over Training Input')
    # fig.legend(prop={'family': 'serif', 'style': 'italic'},loc='upper left', bbox_to_anchor=(0, 0.95, 1, 0.2), mode='expand', borderaxespad=0, ncol=3)
    fig.legend(loc='upper left', frameon=False, fontsize=fs , ncol=3)
    # plt.tight_layout()
    # if save_figs:
        # plt.savefig('plots/NormalizedStates_Plot.pdf', format='pdf', dpi=dpi, bbox_inches='tight') 
    plt.show()
    

def result_plot_unnormalized(
                            y_train,
                            datascaler,
                            x_b_sim_neural,
                            x_f_sim_neural,
                            datascaler_r,
                            x_b_sim_neural_r,
                            x_f_sim_neural_r,
                            save_figs=False
                            ):
    
    n_x_f = x_f_sim_neural.shape[2]
    n_x_b = x_b_sim_neural.shape[2]
    T_train = x_f_sim_neural.shape[1]

    alpha = 0.15
    color_1 = "#228B22"
    color_2 = "#FF4500"
    color_3 = "k"
    
    fs = 8
    lw_rr = 0.4
    cw = 3.5
    ar = 0.5
    dpi = 600
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
    
    x_f_sim_neural_denorm = datascaler.zscore_denormalize(x_f_sim_neural, datascaler.x_mean, datascaler.x_std)
    x_b_sim_neural = torch.reshape(x_b_sim_neural[:, :, :], (y_train.size(0), -1)).detach().numpy()
    
    xhat_1 = torch.reshape(x_f_sim_neural_denorm[:, :, :], (y_train.size(0), -1)).detach().numpy()[:, 0]
    xhat_2 = torch.reshape(x_f_sim_neural_denorm[:, :, :], (y_train.size(0), -1)).detach().numpy()[:, 1]
    z_1 = x_b_sim_neural[:, 0]
    z_2 = x_b_sim_neural[:, 1]
    z_3 = x_b_sim_neural[:, 2]
    
    mean_xhat_1 = np.mean(xhat_1, axis=0)
    mean_xhat_2 = np.mean(xhat_2, axis=0)
    mean_z_1 = np.mean(z_1, axis=0)
    mean_z_2 = np.mean(z_2, axis=0)
    mean_z_3 = np.mean(z_3, axis=0)
    
    std_xhat_1 = np.std(xhat_1, axis=0)
    std_xhat_2 = np.std(xhat_2, axis=0)
    std_z_1 = np.std(z_1, axis=0)
    std_z_2 = np.std(z_2, axis=0)
    std_z_3 = np.std(z_3, axis=0)
    
    x_f_sim_neural_denorm_r = datascaler.zscore_denormalize(x_f_sim_neural_r, datascaler_r.x_mean, datascaler_r.x_std)
    x_b_sim_neural_r = torch.reshape(x_b_sim_neural_r[:, :, :], (y_train.size(0), -1)).detach().numpy()
    
    xhat_1_r = torch.reshape(x_f_sim_neural_denorm_r[:, :, :], (y_train.size(0), -1)).detach().numpy()[:, 0]
    xhat_2_r = torch.reshape(x_f_sim_neural_denorm_r[:, :, :], (y_train.size(0), -1)).detach().numpy()[:, 1]
    z_1_r = x_b_sim_neural_r[:, 0]
    z_2_r = x_b_sim_neural_r[:, 1]
    z_3_r = x_b_sim_neural_r[:, 2]
    
    mean_xhat_1_r = np.mean(xhat_1_r, axis=0)
    mean_xhat_2_r = np.mean(xhat_2_r, axis=0)
    mean_z_1_r = np.mean(z_1_r, axis=0)
    mean_z_2_r = np.mean(z_2_r, axis=0)
    mean_z_3_r = np.mean(z_3_r, axis=0)
    
    std_xhat_1_r = np.std(xhat_1_r, axis=0)
    std_xhat_2_r = np.std(xhat_2_r, axis=0)
    std_z_1_r = np.std(z_1_r, axis=0)
    std_z_2_r = np.std(z_2_r, axis=0)
    std_z_3_r = np.std(z_3_r, axis=0)
    
    fig, axes = plt.subplots(n_x_b, 2, sharex=True, dpi=dpi, figsize = (cw, cw*ar))
    # fig.text(0.3, 0.95, "Exact Variation", ha='center', fontsize=fs)
    # fig.text(0.725, 0.95, "Robust Variation", ha='center', fontsize=fs)
    
    axes[0, 0].plot(z_1, linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[0, 0].hlines(mean_z_1, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[0, 0].fill_between(np.arange(T_train), mean_z_1 - std_z_1 , mean_z_1 + std_z_1, alpha=alpha, color=color_1, zorder=1)   
    axes[0, 0].set_ylabel(r'$z_1$', fontsize = fs)
    # axes[0, 0].grid(True)
    
    axes[1, 0].plot(z_2, linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[1, 0].hlines(mean_z_2, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[1, 0].fill_between(np.arange(T_train), mean_z_2 - std_z_2 , mean_z_2 + std_z_2, alpha=alpha, color=color_1, zorder=1)   
    axes[1, 0].set_ylabel(r'$z_2$', fontsize = fs)
    # axes[1, 0].grid(True)
    
    axes[2, 0].plot(z_3, label=r'residual state', linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[2, 0].hlines(mean_z_3, 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, label=r'$\mu_{z}$', zorder=2)
    axes[2, 0].fill_between(np.arange(T_train), mean_z_3 - std_z_3 , mean_z_3 + std_z_3, alpha=alpha, color=color_1, zorder=1, label=r'$\mu_{z} \pm \sigma_{z}$')   
    axes[2, 0].set_ylabel(r'$z_3$', fontsize = fs, labelpad=7)
    axes[2, 0].set_xlabel('Time [step]', fontsize = fs)
    # axes[2, 0].grid(True)
    
    axes[0, 1].plot(z_1_r , linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[0, 1].hlines(mean_z_1_r , 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[0, 1].fill_between(np.arange(T_train), mean_z_1_r  - std_z_1_r  , mean_z_1_r  + std_z_1_r , alpha=alpha, color=color_1, zorder=1)   
    # axes[0, 1].grid(True)
    
    axes[1, 1].plot(z_2_r , linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[1, 1].hlines(mean_z_2_r , 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[1, 1].fill_between(np.arange(T_train), mean_z_2_r  - std_z_2_r  , mean_z_2_r  + std_z_2_r , alpha=alpha, color=color_1, zorder=1)   
    # axes[1, 1].grid(True)
    
    axes[2, 1].plot(z_3_r , linewidth=2*lw_rr, color=color_3, zorder=3)
    axes[2, 1].hlines(mean_z_3_r , 0, T_train-1, linestyle='--', linewidth=1.75*lw_rr, color=color_2, zorder=2)
    axes[2, 1].fill_between(np.arange(T_train), mean_z_3_r  - std_z_3_r  , mean_z_3_r + std_z_3_r , alpha=alpha, color=color_1, zorder=1)   
    axes[2, 1].set_xlabel('Time [step]', fontsize = fs)
    # axes[2, 1].grid(True)
    
    # fig.suptitle('Normalized State Trajectory over Training Input')
    # fig.legend(prop={'family': 'serif', 'style': 'italic'},loc='upper left', bbox_to_anchor=(0, 0.95, 1, 0.2), mode='expand', borderaxespad=0, ncol=3)
    fig.legend(loc='upper left', frameon=False, fontsize=fs , ncol=3, bbox_to_anchor=(0, 1.05))
    # plt.tight_layout()
    # if save_figs:
        # plt.savefig('plots/Unnormed_NormalizedStates_Plot.pdf', format='pdf', dpi=dpi, bbox_inches='tight') 
    plt.show()
    
# %%
def training_progress_display(train_loss, val_loss, loss_history_train, loss_history_val, epoch, epochs, display_val=True):
    loss_history_train.append(train_loss)
    loss_history_val.append(val_loss)
    print("=====================================================================================================")
    print(f"Epoch: {epoch+1:03d}/{epochs:03d}"f" | Training Loss (RMSE): {train_loss:.6f}")
    if display_val:
        print(f"Epoch: {epoch+1:03d}/{epochs:03d}"f" | Validation Loss (RMSE): {val_loss:.6f}")
    return loss_history_train, loss_history_val

# %%
def save_checkpoint(epoch, simulator, x0, optimizer, loss_train, loss_test, filename):
    checkpoint = {
        'epoch': epoch,
        'simulator_state_dict': simulator.state_dict(),
        'x0': x0,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': loss_train,
        'loss_test': loss_test
    }
    torch.save(checkpoint, filename)

def load_checkpoint(simulator, optimizer, filename):
    checkpoint = torch.load(filename)
    simulator.load_state_dict(checkpoint['simulator_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_train = checkpoint['loss_train']
    loss_test = checkpoint['loss_test']
    x0 = checkpoint['x0']

    return epoch, x0, loss_train, loss_test


# %%
def hard_constraint_projection(a, min_cons_hard, max_cons_hard):
    a = torch.clamp(a, min=min_cons_hard, max=max_cons_hard)
    return a
