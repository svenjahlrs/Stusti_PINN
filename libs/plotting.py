
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from libs.SSP import *
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')
plt.rc('font', size=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', titlesize=8)
plt.rc('legend', title_fontsize=6)
plt.rc('legend', fontsize=6)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

dyn_cyan = '#1b7491'
dyn_red = '#8d1f22'
dyn_pink = '#BC91B9'
dyn_grey = '#5b7382'
dyn_dark = '#0c333f'


def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def plotting_PINN_cut_t(x, t, Eta_true, Eta_pred, xt_train, t_is, path_save, format_save='pdf'):
    SSPe = SSP_2D(Eta_true, Eta_pred)
    MSE = np.mean(np.square(Eta_true- Eta_pred))

    fig, ax = newfig(1.1, 2)
    ax.axis('off')

    gs0 = gridspec.GridSpec(5, 2)
    gs0.update(top=1 - 0.06, bottom=0.4, left=0.15, right=0.8, wspace=0.3)
    ax = plt.subplot(gs0[0:2, :])
    h = ax.imshow(Eta_true.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t \, [\mathrm{s}]$')
    ax.set_ylabel('$x \, [\mathrm{m}]$')
    ax.set_title('true $ H(t,x)$', fontsize=9)

    ax = plt.subplot(gs0[3:, :])
    h = ax.imshow(Eta_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=np.nanmin(Eta_true), vmax=np.nanmax(Eta_true))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.plot(xt_train[:, 1], xt_train[:, 0], 'kx', label='data (%d points)' % (xt_train[:, 0].size), markersize=2, clip_on=False)
    ax.set_xlabel('$t \, [\mathrm{s}]$')
    ax.set_ylabel('$x \, [\mathrm{m}]$')
    leg = ax.legend(frameon=False, loc='best')
    ax.set_title('PINN $H(t,x)$ '+ f'SSP={np.round(SSPe, 4)}, MSE={np.round(MSE, 4)}', fontsize=9)

    gs1 = gridspec.GridSpec(5, 3)
    gs1.update(top=0.27, bottom=0, left=0.1, right=0.8, wspace=0.4, hspace=0.5)

    for i, t_i in enumerate(t_is):

        if i <3:

            if t_i <= np.size(t):
                ax = plt.subplot(gs1[0:2, i])
                ax.plot(x, Eta_true[t_i, :], 'b-', linewidth=0.8, label='exact')
                ax.plot(x, Eta_pred[t_i, :], 'r--', linewidth=0.8, label='prediction')
                ax.set_xlabel('$x \, [\mathrm{m}]$', fontsize=8)
                ax.set_title('$t = %.2f \, \mathrm{s}$' % (t[t_i]), fontsize=8)
                ax.set_ylim([np.nanmin(Eta_true) - 0.1 * np.nanmin(Eta_true), np.nanmax(Eta_true) + 0.1 * np.nanmax(Eta_true)])
                ax.tick_params(axis='both', labelsize=8)

                if i ==0:
                    ax.set_ylabel('$H(t,x) \, [\mathrm{m}]$', fontsize=8)
                    ax.legend(loc='upper center', bbox_to_anchor=(1.9, 1.65), ncol=5, frameon=True, fontsize=8)

        else:
            if t_i <= np.size(t):
                ax = plt.subplot(gs1[3:, i-3])
                ax.plot(x, Eta_true[t_i, :], 'b-', linewidth=0.8, label='exact')
                ax.plot(x, Eta_pred[t_i, :], 'r--', linewidth=0.8, label='prediction')
                ax.set_xlabel('$x \, [\mathrm{m}]$', fontsize=8)
                ax.set_title('$t = %.2f \, \mathrm{s}$' % (t[t_i]), fontsize=8)
                ax.set_ylim([np.nanmin(Eta_true) - 0.1 * np.nanmin(Eta_true), np.nanmax(Eta_true) + 0.1 * np.nanmax(Eta_true)])
                ax.tick_params(axis='both', labelsize=8)

            if i == 3:
                ax.set_ylabel('$H(t,x) \, [\mathrm{m}]$', fontsize=8)

    plt.savefig(f'{path_save}.{format_save}', bbox_inches='tight', pad_inches=0)


def plotting_PINN_cut_x_wavetank(x, t, Eta_true, Eta_pred, xt_train, x_is, path_save,epoch='end', name_plot='$H(t,x)$', format_save='pdf'):
    '''scliced for wavetank'''
    SSPs = np.zeros_like(x_is, dtype=float)
    MSEs = np.zeros_like(x_is, dtype=float)
    for i, x_i in enumerate(x_is):
        SSPs[i] = SSP(Eta_true[:, x_i], Eta_pred[:, x_i])
        MSEs[i] = np.mean(np.square(Eta_true[:, x_i]- Eta_pred[:, x_i]))

    fig, ax = newfig(1.1, 2)
    ax.axis('off')

    gs0 = gridspec.GridSpec(5, 2 , height_ratios=[1, 1, 0.7, 1, 1])
    gs0.update(top=1 - 0.07, bottom=0.4, left=0.12, right=0.8, wspace=0.25)
    ax = plt.subplot(gs0[0:2, :])
    ax.spines['top'].set_color('none')
    # schummeln
    # plt_Eta_true = np.hstack((Eta_true, Eta_true[:,-1][:,None]))
    ax.spines['bottom'].set_color('none')
    h = ax.imshow(Eta_true.T, interpolation='nearest', cmap='viridis', extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax, label=name_plot)
    ax.set_xlabel('$t \, [\mathrm{s}]$')
    ax.set_ylabel('$x \, [\mathrm{m}]$')
    ax.set_title(f'true {name_plot}', fontsize=9)

    ax = plt.subplot(gs0[3:, :])
    h = ax.imshow(Eta_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=np.nanmin(Eta_true), vmax=np.nanmax(Eta_true))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax, label=name_plot)
    ax.plot(xt_train[:, 1], xt_train[:, 0], 'x', label='data (%d points)' % (xt_train[:, 0].size), markersize=5, c=dyn_pink,
            clip_on=False)
    ax.set_xlabel('$t \, [\mathrm{s}]$')
    ax.set_ylabel('$x \, [\mathrm{m}]$')
    leg = ax.legend(frameon=True, loc='best')
    ax.set_title(f'PINN {name_plot} epoch {epoch}: '+ f'SSP={np.round(np.mean(SSPs), 4)}, MSE={np.round(np.mean(MSEs), 6)}', fontsize=9)

    plot_columns = int(np.ceil(len(x_is)/2))
    gs1 = gridspec.GridSpec(5, plot_columns)
    gs1.update(top=0.27, bottom=0, left=0.12, right=0.8, wspace=0.3, hspace=0.5)


    for i, x_i in enumerate(x_is):

        if i <plot_columns:
            dif = np.abs(np.nanmax(Eta_true)- np.nanmin(Eta_true))

            if x_i <= np.size(x):
                ax = plt.subplot(gs1[0:2, i])
                ax.plot(t, Eta_true[:, x_i], c=dyn_cyan, linestyle='-', linewidth=0.8, label='true envelope')
                ax.plot(t, Eta_pred[:, x_i], c=dyn_red, linestyle='--', linewidth=0.8, label='PINN prediction')
                ax.set_xlabel('$t \, [\mathrm{s}]$', fontsize=8, labelpad=-0.07)
                ax.set_title('$x ='+f'{np.round(np.squeeze(x[x_i]),1)}' +'\, \mathrm{m}$: '+ f'SSP = {np.round(SSPs[i],3)}', fontsize=8)
                ax.set_xlim([np.min(t), np.max(t)])
                #ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.set_ylim([np.nanmin(Eta_true)-0.1*dif, np.nanmax(Eta_true)+0.1*dif])
                ax.tick_params(axis='both', labelsize=8)


                if i ==0:
                    ax.set_ylabel(f'{name_plot}'+' $\, [\mathrm{m}]$', fontsize=8)
                    ax.legend(loc='upper center', bbox_to_anchor=(1.8*plot_columns/3, 1.65), ncol=5, frameon=True, fontsize=8)

        else:
            if x_i <= np.size(x):
                ax = plt.subplot(gs1[3:, i-plot_columns])
                ax.plot(t, Eta_true[:, x_i], c=dyn_cyan, linestyle='-', linewidth=0.8, label='True envelope')
                ax.plot(t, Eta_pred[:, x_i], c=dyn_red, linestyle='--', linewidth=0.8, label='PINN prediction')
                ax.set_xlabel('$t \, [\mathrm{s}]$', fontsize=8, labelpad=-0.07)
                ax.set_title('$x =' + f'{np.round(np.squeeze(x[x_i]), 1)}' + '\, \mathrm{m}$: ' + f'SSP = {np.round(SSPs[i], 3)}', fontsize=8)
                ax.set_xlim([np.min(t), np.max(t)])
                #ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.set_ylim([np.nanmin(Eta_true)-0.1*dif, np.nanmax(Eta_true)+0.1*dif])
                ax.tick_params(axis='both', labelsize=8)

            if i ==plot_columns:
                ax.set_ylabel(f'{name_plot}'+' $\, [\mathrm{m}]$', fontsize=8)

    plt.savefig(f'{path_save}_epoch_{epoch}.{format_save}', bbox_inches='tight', pad_inches=0)


def plotting_PINN_cut_x(x, t, Eta_true, Eta_pred, xt_train, x_is, path_save, name_plot='$H(t,x)$', format_save='pdf'):
    SSPe = SSP_2D(Eta_true, Eta_pred)
    MSE = np.mean(np.square(Eta_true - Eta_pred))

    fig, ax = newfig(1.1, 2)
    ax.axis('off')

    gs0 = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 0.7, 1, 1])
    gs0.update(top=1 - 0.07, bottom=0.4, left=0.12, right=0.8, wspace=0.25)
    ax = plt.subplot(gs0[0:2, :])
    h = ax.imshow(Eta_true.T, interpolation='nearest', cmap='viridis', extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax, label=name_plot)
    ax.set_xlabel('$t \, [\mathrm{s}]$')
    ax.set_ylabel('$x \, [\mathrm{m}]$')
    ax.set_title(f'true {name_plot}', fontsize=9)

    ax = plt.subplot(gs0[3:, :])
    h = ax.imshow(Eta_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto',
                  vmin=np.nanmin(Eta_true), vmax=np.nanmax(Eta_true))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax, label=name_plot)
    ax.plot(xt_train[:, 1], xt_train[:, 0], 'x', label='data (%d points)' % (xt_train[:, 0].size), markersize=3, c=dyn_pink,
            clip_on=False)
    ax.set_xlabel('$t \, [\mathrm{s}]$')
    ax.set_ylabel('$x \, [\mathrm{m}]$')
    leg = ax.legend(frameon=False, loc='best')
    ax.set_title(f'PINN {name_plot} ' + f'SSP={np.round(SSPe, 4)}, MSE={np.round(MSE, 6)}', fontsize=9)

    plot_columns = int(np.ceil(len(x_is) / 2))
    gs1 = gridspec.GridSpec(5, plot_columns)
    gs1.update(top=0.27, bottom=0, left=0.12, right=0.9, wspace=0.4, hspace=0.5)

    for i, x_i in enumerate(x_is):

        if i < plot_columns:
            dif = np.abs(np.nanmax(Eta_true) - np.nanmin(Eta_true))

            if x_i <= np.size(x):
                ax = plt.subplot(gs1[0:2, i])
                ax.plot(t, Eta_true[:, x_i], c=dyn_cyan, linestyle='-', linewidth=0.8, label='true envelope')
                ax.plot(t, Eta_pred[:, x_i], c=dyn_red, linestyle='--', linewidth=0.8, label='PINN prediction')
                ax.set_xlabel('$t \, [\mathrm{s}]$', fontsize=8, labelpad=-0.07)
                ax.set_title('$x = %.2f \, \mathrm{m}$' % (x[x_i]), fontsize=8)
                ax.set_xlim([np.min(t), np.max(t)])
                ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.set_ylim([np.nanmin(Eta_true) - 0.1 * dif, np.nanmax(Eta_true) + 0.1 * dif])
                ax.tick_params(axis='both', labelsize=8)

                if i == 0:
                    ax.set_ylabel(f'{name_plot}' + ' $\, [\mathrm{m}]$', fontsize=8)
                    ax.legend(loc='upper center', bbox_to_anchor=(1.8 * plot_columns / 3, 1.65), ncol=5, frameon=True, fontsize=8)

        else:
            if x_i <= np.size(x):
                ax = plt.subplot(gs1[3:, i - plot_columns])
                ax.plot(t, Eta_true[:, x_i], c=dyn_cyan, linestyle='-', linewidth=0.8, label='True envelope')
                ax.plot(t, Eta_pred[:, x_i], c=dyn_red, linestyle='--', linewidth=0.8, label='PINN prediction')
                ax.set_xlabel('$t \, [\mathrm{s}]$', fontsize=8, labelpad=-0.07)
                ax.set_title('$x = %.2f \, \mathrm{m}$' % (x[x_i]), fontsize=8)
                ax.set_xlim([np.min(t), np.max(t)])
                ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.set_ylim([np.nanmin(Eta_true) - 0.1 * dif, np.nanmax(Eta_true) + 0.1 * dif])
                ax.tick_params(axis='both', labelsize=8)

            if i == plot_columns:
                ax.set_ylabel(f'{name_plot}' + ' $\, [\mathrm{m}]$', fontsize=8)

    plt.savefig(f'{path_save}.{format_save}', bbox_inches='tight', pad_inches=0)


def plotting_PINN_wavetank(x, t, Eta_pred, xt_train, x_is, M_is, name_save):

    fig, ax = newfig(1.1, 2)
    ax.axis('off')

    max = np.nanmax(np.nanmax(np.abs(M_is)))
    print(max)

    gs0 = gridspec.GridSpec(5, 2)
    gs0.update(top=1 - 0.06, bottom=0.5, left=0.15, right=0.8, wspace=0.3)

    ax = plt.subplot(gs0[3:, :])
    h = ax.imshow(Eta_pred.T, interpolation='nearest', cmap='viridis',
                  extent=[np.nanmin(t), np.nanmax(t), np.nanmin(x), np.nanmax(x)], origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.plot(xt_train[:, 1], xt_train[:, 0], 'kx', label='Data (%d points)' % (xt_train[:, 0].size), markersize=2,
            clip_on=False)
    ax.set_xlabel('$t [s]$')
    ax.set_ylabel('$x [m]$')
    leg = ax.legend(frameon=False, loc='best')
    ax.set_title('PINN $\eta(t,x)$ ', fontsize=10)

    gs1 = gridspec.GridSpec(5, 3)
    gs1.update(top=0.35, bottom=0, left=0.1, right=0.9, wspace=0.5, hspace=0.5)

    for i, (x_i, M_i) in enumerate(zip(x_is, M_is)):

        if i <3:

            if x_i <= np.size(x):
                SSPe = SSP(M_i, Eta_pred[:, x_i])
                MSE = np.mean(np.square(M_i - Eta_pred[:, x_i]))

                ax = plt.subplot(gs1[0:2, i])
                ax.plot(t, M_i, 'b-', linewidth=1, label='Exact')
                ax.plot(t, Eta_pred[:, x_i], 'r--', linewidth=1, label='Prediction')
                ax.set_xlabel('$t [s]$', fontsize=8, labelpad=-0.07)
                ax.set_ylim([-max, max])
                ax.set_title('$x = %.1f m, SSP= %.3f $' % (x[x_i], SSPe), fontsize=7)
                ax.tick_params(axis='both', labelsize=8)


                if i ==0:
                    ax.set_ylabel('$\eta(t,x)$', fontsize=8)
                    ax.legend(loc='upper center', bbox_to_anchor=(1.5, 1.6), ncol=5, frameon=False, fontsize=9)

        else:
            if x_i <= np.size(x):
                SSPe = SSP(M_i, Eta_pred[:, x_i])
                MSE = np.mean(np.square(M_i - Eta_pred[:, x_i]))

                ax = plt.subplot(gs1[3:, i - 3])
                ax.plot(t, M_i, 'b-', linewidth=1, label='Exact')
                ax.plot(t, Eta_pred[:, x_i], 'r--', linewidth=1, label='Prediction')
                ax.set_xlabel('$t [s]$', fontsize=8, labelpad=-0.07)
                ax.set_ylim([-max, max])
                ax.set_title('$x = %.1f m, SSP= %.3f $' % (x[x_i], SSPe), fontsize=7)
                ax.tick_params(axis='both', labelsize=8)

            if i == 3:
                ax.set_ylabel('$\eta(t,x)$', fontsize=8)

    plt.savefig(f'./figures/{name_save}.pdf')


def plotting_losscurve(path_loss: str, path_save, format_save='pdf', figsize=(6, 5), **kwargs):

    df = pd.read_csv(path_loss)

    plt.figure(figsize=figsize)
    plt.plot(df.MSE_u_ts[1:], label='$\mathrm{MSE_{u,ts}}$', linewidth=0.8)
    plt.plot(df.MSE_v_ts[1:], label='$\mathrm{MSE_{v,ts}}$', linewidth=0.8)
    plt.plot(df.MSE_u_dir[1:], label='$\mathrm{MSE_{u,dirichlet}}$', linewidth=0.8)
    plt.plot(df.MSE_v_dir[1:], label='$\mathrm{MSE_{v,dirichlet}}$', linewidth=0.8)
    plt.plot(df.MSE_u_new[1:], label='$\mathrm{MSE_{u,newman}}$', linewidth=0.8)
    plt.plot(df.MSE_v_new[1:], label='$\mathrm{MSE_{v,newman}}$', linewidth=0.8)
    plt.plot(df.MSE_u_pde[1:], label='$\mathrm{MSE_{u,pde}}$', linewidth=0.8)
    plt.plot(df.MSE_v_pde[1:], label='$\mathrm{MSE_{v,pde}}$', linewidth=0.8)
    if 'ymax' in kwargs:
        plt.ylim([0.0000000001, kwargs['ymax']])

    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs', fontsize=8)
    plt.ylabel('loss', fontsize=8)
    plt.yscale('log')
    plt.grid()
    plt.yticks([0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
    plt.tick_params(axis='both', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{path_save}/loss.{format_save}')


def plotting_losscurve_noBCs(path_loss: str, path_save, format_save='pdf', figsize=(6, 5), **kwargs):

    df = pd.read_csv(path_loss)

    plt.figure(figsize=figsize)
    plt.plot(df.MSE_u_ts[1:], label='$\mathrm{MSE_{u,ts}}$', linewidth=0.8)
    plt.plot(df.MSE_v_ts[1:], label='$\mathrm{MSE_{v,ts}}$', linewidth=0.8)
    plt.plot(df.MSE_u_pde[1:], label='$\mathrm{MSE_{u,pde}}$', linewidth=0.8)
    plt.plot(df.MSE_v_pde[1:], label='$\mathrm{MSE_{v,pde}}$', linewidth=0.8)
    if 'ymax' in kwargs:
        plt.ylim([0.0000000001, kwargs['ymax']])

    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs', fontsize=8)
    plt.ylabel('loss', fontsize=8)
    plt.yscale('log')
    plt.grid()
    plt.yticks([0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1])
    plt.tick_params(axis='both', labelsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{path_save}/loss.{format_save}')

def plotting_losscurve_lossweight(path_loss: str, path_save, format_save='pdf', figsize=(6, 8), **kwargs):

    df = pd.read_csv(path_loss + '.csv')
    dfw = pd.read_csv(path_loss + '_weights.csv')

    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(df.MSE_u_ts[1:], label='$\mathrm{MSE_{u,ts}}$', linewidth=0.8)
    plt.plot(df.MSE_v_ts[1:], label='$\mathrm{MSE_{v,ts}}$', linewidth=0.8)
    plt.plot(df.MSE_u_pde[1:], label='$\mathrm{MSE_{u,pde}}$', linewidth=0.8)
    plt.plot(df.MSE_v_pde[1:], label='$\mathrm{MSE_{v,pde}}$', linewidth=0.8)
    plt.plot(df.MSE_u_dir[1:], label='$\mathrm{MSE_{u,dirichlet}}$', linewidth=0.8)
    plt.plot(df.MSE_v_dir[1:], label='$\mathrm{MSE_{v,dirichlet}}$', linewidth=0.8)
    plt.plot(df.MSE_u_new[1:], label='$\mathrm{MSE_{u,newman}}$', linewidth=0.8)
    plt.plot(df.MSE_v_new[1:], label='$\mathrm{MSE_{v,newman}}$', linewidth=0.8)
    if 'ymax' in kwargs:
        plt.ylim([0.000000001, kwargs['ymax']])
    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs', fontsize=8)
    plt.ylabel('loss', fontsize=8)
    plt.yscale('log')
    plt.grid()
    plt.tick_params(axis='both', labelsize=8)
    plt.legend(fontsize=8)

    plt.subplot(2, 1, 2)
    plt.plot(dfw.w_MSE_u_ts[1:], label='$\mathrm{w_{u,ts}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_v_ts[1:], label='$\mathrm{w_{v,ts}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_u_pde[1:], label='$\mathrm{w_{u,pde}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_v_pde[1:], label='$\mathrm{w_{v,pde}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_u_dir[1:], label='$\mathrm{w_{u,dirichlet}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_v_dir[1:], label='$\mathrm{w_{v,dirichlet}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_u_new[1:], label='$\mathrm{w_{u,newman}}$', linewidth=0.8)
    plt.plot(dfw.w_MSE_v_new[1:], label='$\mathrm{w_{v,newman}}$', linewidth=0.8)
    plt.ylim([0, 1.1])
    if 'xmax' in kwargs:
        plt.xlim([0, kwargs['xmax']])
    plt.xlabel('epochs', fontsize=8)
    plt.ylabel('weight', fontsize=8)
    plt.grid()
    plt.tick_params(axis='both', labelsize=8)
    plt.legend(fontsize=8)



    plt.tight_layout()
    plt.savefig(f'{path_save}_loss.{format_save}')


def plotting_loss_weights_2D_SA(xs, ts, lambdas, name, epoch, path_save, figsize=(5, 2), format_save='png'):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    ax.set_title(f'self-adaption weights for {name}, epoch: {epoch}')
    pnt = ax.scatter(ts, xs, s=0.1, c=lambdas, cmap='jet')
    plt.xlabel('$t \, [\mathrm{s}]$')
    plt.ylabel('$x \, [\mathrm{m}]$')
    cbar = plt.colorbar(pnt)
    cbar.set_label("$\lambda_i$")
    plt.tight_layout()
    plt.savefig(f'{path_save}/{name}_epoch_{epoch}.{format_save}')