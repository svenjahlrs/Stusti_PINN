import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
from scipy.interpolate import griddata
from libs.plotting import *
from libs.wave_tools import *
from libs.PINN_NLSE_classes import PINN_NLSE_SA, PINN_NLSE_SA_noBCs
import time
from pyDOE import lhs
from pathlib import Path
from scipy.signal import hilbert
import pandas as pd
import os

dyn_cyan = '#1b7491'
dyn_red = '#8d1f22'
dyn_pink = '#BC91B9'
dyn_grey = '#5b7382'
dyn_dark = '#0c333f'


def main(Exact_solution, true_solution='bc'):
    # domain bounds
    lower_b = np.array([np.min(x), np.min(t)])  # lower boundary: [X_min, T_min]
    upper_b = np.array([np.max(x), np.max(t)])  # upper boundary: [X_max, T_max]

    # collocation points
    Exact_u = np.real(Exact_solution)  # real part of analytical solution u(x,t)
    Exact_v = np.imag(Exact_solution)  # imaginary part of analytical solution v(x,t)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)  # |h(x,t)|

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # array with discretization flattened (N_x*N_t x 2 entries)

    id_b = np.random.choice(x.shape[0], N_b, replace=False)
    id_ts = np.random.choice(t.shape[0], N0, replace=False)

    #points where we have exact boundary conditions as time-series (true solutions = ts)
    xt_ts = np.vstack((np.hstack((X[:, 0:1], T[:, 0:1]))[id_ts, :], np.hstack((X[:, -1:], T[:, -1:]))[id_ts, :]))  # [X_min, T_min to T_max]
    u_ts = np.vstack((Exact_u[:, 0:1][id_ts, :], Exact_u[:, -1:][id_ts, :]))  # real part of exact solution u(t, x=x_0) on x-boundary
    v_ts = np.vstack((Exact_v[:, 0:1][id_ts, :], Exact_v[:, -1:][id_ts, :]))  # imaginary part of exact solution u(t, x=x_0) = 0 on x-boundary


    # # points where we have exact boundary conditions as time-series (true solutions = ts)
    # xt_ts = np.hstack((X[:, 0:1], T[:, 0:1]))[id_ts, :] # [X_min, T_min to T_max]
    # u_ts = Exact_u[:, 0:1][id_ts, :] # real part of exact solution u(t, x=x_0) on x-boundary
    # v_ts = Exact_v[:, 0:1][id_ts, :] # imaginary part of exact solution u(t, x=x_0) = 0 on x-boundary
    #
    # # points for periodic boundaries (in time domain!)
    # xt_bl = np.hstack((X[0:1, :].T, T[0:1, :].T))[id_b, :]
    # xt_bu = np.hstack((X[-1:, :].T, T[-1:, :].T))[id_b, :]


    # collocation points inside the domain
    xt_pde = lower_b + (upper_b - lower_b) * lhs(2, N_f)  # random locations inside x-t domain? lhs(2, Nf): latin hypercube sampling (values between 0 and 1 )

    dict_collpoints = {'xt_ts': xt_ts, 'u_ts': u_ts, 'v_ts': v_ts, 'xt_pde': xt_pde, 'X': X, 'T': T}

    model = PINN_NLSE_SA_noBCs(dic=dict_collpoints, l_w_dic=dict_loss_weights_params, neurons_lay=layers, lower_bound=lower_b, upper_bound=upper_b,
                       ome_p=omega_p, depth=d, hs_LBFGS=hist_size, epAdam=epochsAdam, epLBFGS=epochsLBFGS, lr_LBFGS=lr_LBFGS, lr_Adam=lr_Adam, name_for_save=name_save)

    start_time = time.time()
    model.train(plot_each=5000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)

    u_pred, v_pred = model.predict(X_star[:, 0:1], X_star[:, 1:2])
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')
    A_pred = U_pred + 1j * V_pred
    Traeger_pred = wavevector_to_envelope(A=A_pred, x_grid=x, t_grid=t, omega=omega_p, kp=k_p)

    # final plotting

    plotting_PINN_cut_x_wavetank(x=x, t=t, Eta_true=Exact_h, Eta_pred=H_pred,
                        xt_train=dict_collpoints['xt_ts'],
                        x_is=[0, 300, 400, 500, 600], path_save=model.FIGURE_PATH + 'abs', format_save='png', name_plot='$B(x,t)$')

    plotting_PINN_cut_x_wavetank(x=x, t=t, Eta_true=Exact_u, Eta_pred=U_pred,
                        xt_train=dict_collpoints['xt_ts'],
                        x_is=[0, 300, 400, 500, 600], path_save=model.FIGURE_PATH + 'real', format_save='png', name_plot='$U(x,t)$')

    plotting_PINN_cut_x_wavetank(x=x, t=t, Eta_true=Exact_v, Eta_pred=V_pred,
                        xt_train=dict_collpoints['xt_ts'],
                        x_is=[0, 300, 400, 500, 600], path_save=model.FIGURE_PATH + 'imag', format_save='png', name_plot='$V(x,t)$')

    plotting_losscurve_noBCs(path_loss=model.LOSS_PATH, path_save=model.FIGURE_PATH, format_save='png', xmax=epochsAdam + epochsLBFGS, ymax=10, figsize=(9, 5))

    # intemediate plotting
    collp = np.load(os.path.join(model.ERROR_PATH, 'collpoints_all_epochs.npz'))
    datas = [f for f in os.listdir(model.ERROR_PATH) if 'lambdas' in f]
    for dat in datas:
        npzfile = np.load(os.path.join(model.ERROR_PATH, dat))
        plotting_loss_weights_2D_SA(xs=collp['x_pde'], ts=collp['t_pde'], lambdas=npzfile['lambda_u_pde'], name='U-pde', epoch=npzfile['epoch'], path_save=model.FIGURE_PATH,
                                    format_save='png')
        plotting_loss_weights_2D_SA(xs=collp['x_pde'], ts=collp['t_pde'], lambdas=npzfile['lambda_v_pde'], name='V-pde', epoch=npzfile['epoch'], path_save=model.FIGURE_PATH,
                                    format_save='png')

    datas = [f for f in os.listdir(model.ERROR_PATH) if 'pred' in f]
    for dat in datas:
        npzfile = np.load(os.path.join(model.ERROR_PATH, dat))

        plotting_PINN_cut_x_wavetank(x=x, t=t, Eta_true=Exact_h,
                                     Eta_pred=griddata(X_star, np.sqrt(npzfile['u_pred'] ** 2 + npzfile['v_pred'] ** 2).flatten(), (X, T), method='cubic'),
                                     xt_train=dict_collpoints['xt_ts'], x_is=[0, 300, 400, 500, 600], path_save=model.FIGURE_PATH + 'abs', format_save='png', name_plot='$B(x,t)$', epoch=npzfile['epoch'])

        plotting_PINN_cut_x_wavetank(x=x, t=t, Eta_true=Exact_u, Eta_pred=griddata(X_star, npzfile['u_pred'].flatten(), (X, T), method='cubic'),
                                     xt_train=dict_collpoints['xt_ts'], x_is=[0, 300, 400, 500, 600], path_save=model.FIGURE_PATH + 'real', format_save='png', name_plot='$U(x,t)$', epoch=npzfile['epoch'])

        plotting_PINN_cut_x_wavetank(x=x, t=t, Eta_true=Exact_v, Eta_pred=griddata(X_star, npzfile['v_pred'].flatten(), (X, T), method='cubic'),
                                     xt_train=dict_collpoints['xt_ts'], x_is=[0, 300, 400, 500, 600], path_save=model.FIGURE_PATH + 'imag', format_save='png', name_plot='$V(x,t)$', epoch=npzfile['epoch'])
    del model

if __name__ == "__main__":


    data_path = '/mnt/c/Users/svenj/Documents/PINN_NLS/Data/waves_data'
    data_path = '/home/svenja/DeployedProjects/PINN_NLS/Data/waves_data'

    d = 1  # water depth

    epochsAdam = 20000
    epochsLBFGS = 0

    hist_size = 30  # number of previous iterations that are used to approximate the inverse Hessian matrix for LBFGS optimizer.
    lr_LBFGS = 1  # learning rate for LBFGS optimizer
    # lr_Adam = 0.001

    np.random.seed(123)
    samples = np.random.randint(0, 1260, 20)
    samples = np.linspace(0, 1260, 20)
    samples[-1] = 1258


    for lr_Adam in [ 0.0001, 0.00001]:
        for divider_fact in [0.25, 0.5, 1, 2]:
            for ws in [397, 663]:
                ws = int(ws)
                np.random.seed(1234)
                torch.manual_seed(1234)

                N0 = 1200  # random initial data points on h(0, x)
                N_b = 200  # random collocation points for enforcing periodic boundaries
                N_f = 20000  # random collocation points inside the domain

                layers = [2, 200, 200, 200, 200, 2]  # nodes per layer of MLP

                # define domain
                t = np.expand_dims(np.linspace(0, 60, 1200), axis=1)
                x = np.expand_dims(np.linspace(0, 6, 601), axis=1)
                X, T = np.meshgrid(x, t)

                eta0 = np.squeeze(pd.read_csv(Path(data_path, 'data_WB_short.csv'), skiprows=ws, nrows=1).values)  # boundary values at 0m
                eta3 = np.squeeze(pd.read_csv(Path(data_path, 'data_3m_short.csv'), skiprows=ws, nrows=1).values)  # boundary values at 3m
                eta4 = np.squeeze(pd.read_csv(Path(data_path, 'data_4m_short.csv'), skiprows=ws, nrows=1).values)  # boundary values at 4m
                eta5 = np.squeeze(pd.read_csv(Path(data_path, 'data_5m_short.csv'), skiprows=ws, nrows=1).values)  # boundary values at 5m
                eta6 = np.squeeze(pd.read_csv(Path(data_path, 'data_6m_short.csv'), skiprows=ws, nrows=1).values)  # boundary values at 6m
                params = np.squeeze(pd.read_csv(Path(data_path, 'params_eps_ome_gam_short.csv'), skiprows=ws, nrows=1).values)
                omega_p = params[1]
                eps = params[0]
                gam = params[2]
                # generate analytic coefficients
                k_p, C_g, lamb, mu, delta, nu = NLSE_coefficients_chabchoub(omega_p=omega_p, d=d)
                print(f'\n\nloaded sample {ws} with eps={eps}, omega={omega_p} and gam={gam}\n\n')

                # hilbert transform liefert [eta(t)+i*H(eta(t))]
                H_0 = hilbert(eta0)  # envelope at 0m
                H_3 = hilbert(eta3)  # envelope at 0m
                H_4 = hilbert(eta4)  # envelope at 0m
                H_5 = hilbert(eta5)  # envelope at 0m
                H_6 = hilbert(eta6)  # envelope at 6m

                # mit Trägerwellenanteil multiplizieren (positiv drehend, wie in Untersuchung herausgefunden)
                Psi_0 = convert_to_wavevector(H=H_0, x=0, t_inc=0.0500, omega=omega_p, kp=k_p)
                Psi_3 = convert_to_wavevector(H=H_3, x=3, t_inc=0.0500, omega=omega_p, kp=k_p)
                Psi_4 = convert_to_wavevector(H=H_4, x=4, t_inc=0.0500, omega=omega_p, kp=k_p)
                Psi_5 = convert_to_wavevector(H=H_5, x=5, t_inc=0.0500, omega=omega_p, kp=k_p)
                Psi_6 = convert_to_wavevector(H=H_6, x=6, t_inc=0.0500, omega=omega_p, kp=k_p)

                divider_true_data = divider_fact * (np.max([np.max(np.abs(Psi_3.real)), np.max(np.abs(Psi_3.imag))]))  # an der stelle die vorgegeben!

                Psi_0 = Psi_0 / divider_true_data
                Psi_3 = Psi_3 / divider_true_data
                Psi_4 = Psi_4 / divider_true_data
                Psi_5 = Psi_5 / divider_true_data
                Psi_6 = Psi_6 / divider_true_data

                Exact = np.zeros_like(X, dtype=complex)
                Exact[:] = np.nan + 1j * np.nan
                Exact[:, 0] = Psi_0
                Exact[:, 1] = Psi_0
                Exact[:, 2] = Psi_0
                Exact[:, 299] = Psi_3
                Exact[:, 300] = Psi_3
                Exact[:, 301] = Psi_3
                Exact[:, 399] = Psi_4
                Exact[:, 400] = Psi_4
                Exact[:, 401] = Psi_4
                Exact[:, 499] = Psi_5
                Exact[:, 500] = Psi_5
                Exact[:, 501] = Psi_5
                Exact[:, 594] = Psi_6
                Exact[:, 595] = Psi_6
                Exact[:, 596] = Psi_6
                Exact[:, 597] = Psi_6
                Exact[:, 598] = Psi_6
                Exact[:, 599] = Psi_6
                Exact[:, 600] = Psi_6

                extract_ts = 'bc'  # other option initial condition 'ic'

                dict_loss_weights_params = {'w_u_ts': np.sqrt(50), 'w_v_ts': np.sqrt(50),
                                            'w_u_pde': np.sqrt(1), 'w_v_pde': np.sqrt(1),
                                            'NLS_form': 'space_like', 'given_ic': extract_ts, 'data_loss_SA': True}


                name_save = f'two_eta_0_6m_tests_2/NLSE_space_wavetank_SA-PINN_epochs_{epochsAdam}_{epochsLBFGS}_lr_{lr_Adam}_bothAdam_divide_{divider_fact}_IW_1_50_sample_W_{ws}_eps_{np.round(params[0], 4)}_ome_{int(params[1])}_gam{int(params[2])}'

                main(Exact_solution=Exact, true_solution=extract_ts)



