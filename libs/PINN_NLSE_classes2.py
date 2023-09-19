from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
from scipy.interpolate import griddata
from libs.plotting import *
from libs.wave_tools import *
import time
from csv import writer
from os.path import exists
import os
from itertools import chain


def init_weight_bias(m):
    """initializes weights of a NN layer with Xavier distribution """
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def cdn(var: torch.Tensor):
    """function converts tensors on GPU back to numpy arrays on CPU"""
    return var.cpu().detach().numpy()


def tfn(var: np.ndarray):
    """generates torch tensor in float32 on GPU device"""
    return torch.from_numpy(var).float().to(device)


def write_csv_line(path: str, line):
    """ writes a new line to a csv file at path"""
    with open(path, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(line)


class Net_Sequential(nn.Module):
    def __init__(self, lb, ub, nodes):
        super(Net_Sequential, self).__init__()
        self.lb = lb
        self.ub = ub
        self.nodes = nodes
        self.layers = nn.ModuleList()
        for i in range(len(self.nodes) - 1):
            self.layers.append(nn.Linear(int(self.nodes[i]), int(self.nodes[i + 1])))
        self.tanh = nn.Tanh()

    def forward(self, *inp):  # inp is tuple either (x, t) or (x, t, z)
        inputs = torch.cat(inp, axis=1)
        inputs = torch.div(2.0 * (inputs - self.lb), self.ub - self.lb) - 1.0  # normalize
        for i in range(len(self.nodes) - 2):
            lay = self.layers[i]
            inputs = self.tanh(lay(inputs))
        lay = self.layers[i + 1]
        output = lay(inputs)  # no activation is used in output layer
        return output


class PINN_NLSE:
    def __init__(self, dic, l_w_dic, neurons_lay, lower_bound, upper_bound, ome_p, depth, hs_LBFGS, lr_LBFGS, lr_Adam, epAdam, epLBFGS, name_for_save):
        """
        class to implement and train a Physics-informed Neural Network for the hydrodynamic nonlinear Schrödinger equation according to Chabchoub2016
        (https://www.mdpi.com/2311-5521/1/3/23#FD6-fluids-01-00023) in time-like or space-like form with given initial or boundary conditions
        :param dic: dictionary containing the true-solution data points (xt_ts, u_ts, x_ts), the spatial or temporal boundary points (xt_bl, xt_bu) and the collocation points for the PDE (xtde)
        :param l_w_dic: dictionary containing the chosen loss weights for each loss components
        :param neurons_lay: list of neurons per layer for PINN
        :param lower_bound: of domain [x_min, t_min] or [x_min, t_min, z_min]
        :param upper_bound: of domain [x_max, t_max] or [x_max, t_max, z_max]
        :param ome_p: peak frequency for NLS
        :param depth: water depth [m] for dispersion
        :param hs_LBFGS: history size for the LBFGS optimizer
        :param lr_LBFGS: learning rate for the LBFGS optimizer
        :param epLBFGS: maximum of iterations for the LBFGS optimizer (already needed for initialization) is equal to max_it! So no training-for-loop needed later.
        :param epAdam: epochs of Adam optimizer (actually needed later, but also here for consistency
        :param name_for_save: string with the name for saving the model parameters and figures
        """

        self.l_w_dict = l_w_dic
        self.loss = None  # sum of all loss components
        self.best_loss = None  # to only save models which performance is better than a previously achieved performance
        self.loss_MSE_u_ts = None  # loss components of true-solution data points (u:real part, v:imaginary part)
        self.loss_MSE_v_ts = None
        self.loss_MSE_u_dir = None  # loss components of dirichlet boundary conditions
        self.loss_MSE_v_dir = None
        self.loss_MSE_u_new = None  # loss components of newman boundary conditions
        self.loss_MSE_v_new = None
        self.loss_MSE_u_pde = None  # loss components of PDE-residual
        self.loss_MSE_v_pde = None


        self.name_save = name_for_save
        self.omega_p = ome_p
        self.d = depth
        self.layers = neurons_lay  # number of nodes each layer [2, .., 2]
        self.hs = hs_LBFGS
        self.lr_LBFGS = lr_LBFGS
        self.lr_Adam = lr_Adam
        self.epochsAdam = epAdam
        self.epochsLBFGS = epLBFGS
        self.epoch = 0

        self.start_time = time.time()
        self.MODEL_PATH = f'models/{self.name_save}.pth'
        self.ERROR_PATH = f"errors/{self.name_save}/"
        if not os.path.isdir(self.ERROR_PATH):
            os.mkdir(self.ERROR_PATH)
        self.LOSS_PATH = os.path.join(self.ERROR_PATH, f"loss.csv")
        self.FIGURE_PATH = f"figures/{self.name_save}/"
        if not os.path.isdir(self.FIGURE_PATH):
            os.mkdir(self.FIGURE_PATH)

        # convert data to torch tensors
        self.lower_bound = tfn(lower_bound)
        self.upper_bound = tfn(upper_bound)

        # supervised training data
        self.x_ts = Variable(tfn(dic['xt_ts'][:, 0:1]), requires_grad=True)
        self.t_ts = Variable(tfn(dic['xt_ts'][:, 1:2]), requires_grad=True)
        self.u_ts = Variable(tfn(dic['u_ts']), requires_grad=False)
        self.v_ts = Variable(tfn(dic['v_ts']), requires_grad=False)

        # collocation points inside entire (x, t) domain for pde-residual
        self.x_pde = Variable(tfn(dic['xt_pde'][:, 0:1]), requires_grad=True)
        self.t_pde = Variable(tfn(dic['xt_pde'][:, 1:2]), requires_grad=True)

        # collocation points for periodic boundaries (bl=boundary lower, bu= boundary upper)
        self.x_bl = Variable(tfn(dic['xt_bl'][:, 0:1]), requires_grad=True)
        self.t_bl = Variable(tfn(dic['xt_bl'][:, 1:2]), requires_grad=True)

        self.x_bu = Variable(tfn(dic['xt_bu'][:, 0:1]), requires_grad=True)
        self.t_bu = Variable(tfn(dic['xt_bu'][:, 1:2]), requires_grad=True)

        #factor for normalization
        self.fact = np.max(np.sqrt((dic['u_ts']**2+ dic['v_ts']**2)))


        # Initialize NNs
        self.model = Net_Sequential(lb=self.lower_bound[0:2], ub=self.upper_bound[0:2], nodes=self.layers)

        self.model.apply(init_weight_bias)

        self.model.float().to(device)

        self.optimizerAdam = torch.optim.Adam(self.model.parameters(), lr=self.lr_Adam)
        self.scheduler = ReduceLROnPlateau(self.optimizerAdam, patience=1000, verbose=True)

        self.optimizerLBFGS = torch.optim.LBFGS(self.model.parameters(),
                                                lr=self.lr_LBFGS,
                                                max_iter=self.epochsLBFGS,
                                                history_size=self.hs,
                                                line_search_fn=None)

    def load_model(self, path: str):
        """function loads a model, epochs and optimizer if .pth file of previously trained model exists at path location"""
        ll = torch.load(path)
        self.model.load_state_dict(ll['net'])
        self.epoch = ll['epoch']
        self.optimizerAdam.load_state_dict(ll['optim_Adam'])
        self.optimizerLBFGS.load_state_dict(ll['optim_LBFGS'])
        print(f'\n\n\nLoaded model from epoch {self.epoch}: \n\n\n')

    def save_model(self, path: str):
        """ function saves a model, epochs and optimizer as a pth.file at path"""
        torch.save(
            {'net': self.model.state_dict(),
             'epoch': self.epoch,
             'optim_Adam': self.optimizerAdam.state_dict(),
             'optim_LBFGS': self.optimizerLBFGS.state_dict()},
            path)
        print(f'model checkpoint saved')

    def net_uv(self, x, t, doub=False):
        """
        function to only accomplish the forward path through th NN to predict u and v (as otherwise also the derivatives would be
        calculated during inference, which is not necessary (computational effort)
        :param x: x coordinates to make prediction at
        :param t: corresponding t coordinates
        :param doub: boolean whether to calculate with double-precision (float64) format (required for stable LBFGS, not for Adam)
        :return: prediction of u = real part and v = imaginary part at all requested points (x_i,t_i)
        """

        if doub:
            uv = self.model(x.double(), t.double())
        else:
            uv = self.model(x, t)

        u = uv[:, 0:1]  # real part u (None x 1)
        v = uv[:, 1:2]  # imaginary v (None x 1)

        return u, v

    def net_duv(self, x, t, doub=False):
        """
        function to accomplish the forward path through th NN to predict u and v and also calculate their derivatives
        :param x: x coordinates to make prediction at
        :param t: corresponding t coordinates
        :param doub: boolean whether to calculate with double-precision (float64) format (required for stable LBFGS, not for Adam)
        :return: first and second derivatives w.r.t. x and t.
        """

        u, v = self.net_uv(x=x, t=t, doub=doub)

        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]

        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_tt = torch.autograd.grad(v_t.sum(), t, create_graph=True)[0]

        return u_x, u_xx, u_t, u_tt, v_x, v_xx, v_t, v_tt

    def loss_PINN(self, doub=False):
        """function calculates the multi-component loss-function by first predicting the required real (u) and imaginary (v) solution and its derivatives and the known
        points (ts), upper and lower boundary (bu, bl) for dirichlet and newman conditions and inside the domain (pde). For calculating the NLS residual, the time-like
        or space-like hydrodynamic NLSE can be initialized. For the newman condition we either use derivatives in time or space direction depending on whether we use
        real initial conditions or actually boundary conditions as th true solution (ts)"""

        # prediction where we know exact solution
        u_pred, v_pred = self.net_uv(x=self.x_ts, t=self.t_ts, doub=doub)

        # prediction at lower and upper boundary
        u_bl, v_bl = self.net_uv(x=self.x_bl, t=self.t_bl, doub=doub)
        u_bl_x, _, u_bl_t, _, v_bl_x, _, v_bl_t, _ = self.net_duv(x=self.x_bl, t=self.t_bl, doub=doub)

        u_bu, v_bu = self.net_uv(x=self.x_bu, t=self.t_bu, doub=doub)
        u_bu_x, _, u_bu_t, _, v_bu_x, _, v_bu_t, _ = self.net_duv(x=self.x_bu, t=self.t_bu, doub=doub)

        # # prediction inside domain
        u_pde, v_pde = self.net_uv(x=self.x_pde, t=self.t_pde, doub=doub)
        u_pde_x, u_pde_xx, u_pde_t, u_pde_tt, v_pde_x, v_pde_xx, v_pde_t, v_pde_tt = self.net_duv(x=self.x_pde, t=self.t_pde, doub=doub)

        if self.l_w_dict['NLS_form'] == 'time_like':  # time-like NLS residuals inside domain (for hydrodynamic wavemaker problems)
            _, C_g, _, _, delta, nu = NLSE_coefficients_chabchoub(omega_p=self.omega_p, d=self.d)
            f_v_pred = u_pde_x + 1 / C_g * u_pde_t + delta * v_pde_tt + nu * (u_pde ** 2 + v_pde ** 2) * v_pde
            f_u_pred = v_pde_x - 1 / C_g * v_pde_t + delta * u_pde_tt + nu * (u_pde ** 2 + v_pde ** 2) * u_pde
        elif self.l_w_dict['NLS_form'] == 'space_like':  # space-like form of NLS residuals inside domain
            _, C_g, lamb, mu, _, _ = NLSE_coefficients_chabchoub(omega_p=self.omega_p, d=self.d)
            f_v_pred = u_pde_t + C_g * u_pde_x + lamb * v_pde_xx + mu * (u_pde ** 2 + v_pde ** 2) * v_pde
            f_u_pred = - v_pde_t - C_g * v_pde_x + lamb * u_pde_xx + mu * (u_pde ** 2 + v_pde ** 2) * u_pde
        else:
            print('unknown NLS_form')

        # error between true solution and predicted solution
        self.loss_MSE_u_ts = torch.mean(torch.square(self.u_ts - u_pred))
        self.loss_MSE_v_ts = torch.mean(torch.square(self.v_ts - v_pred))

        # error periodic dirichlet boundary
        self.loss_MSE_u_dir = torch.mean(torch.square(u_bu - u_bl))
        self.loss_MSE_v_dir = torch.mean(torch.square(v_bu - v_bl))

        # error periodic newman boundary
        if self.l_w_dict[
            'given_ic'] == 'bc':  # if we know the true-solution as a time-series at x=x_0 (thus have an "initial-boundary condition") the newman bc must be satisfied in t-direction
            self.loss_MSE_u_new = torch.mean(torch.square(u_bu_t - u_bl_t))
            self.loss_MSE_v_new = torch.mean(torch.square(v_bu_t - v_bl_t))
        elif self.l_w_dict[
            'given_ic'] == 'ic':  # if we know the true-solution as a snapshot along x at t=t_0 (thus have a "real initial condition") the newman bc must be satisfied in x-direction
            self.loss_MSE_u_new = torch.mean(torch.square(u_bu_x - u_bl_x))
            self.loss_MSE_v_new = torch.mean(torch.square(v_bu_x - v_bl_x))
        else:
            print('error in given_ic value!')

        # PDE-residual error
        self.loss_MSE_u_pde = torch.mean(torch.square(f_u_pred))
        self.loss_MSE_v_pde = torch.mean(torch.square(f_v_pred))

        self.loss = self.l_w_dict['w_u_ts'] * self.loss_MSE_u_ts + self.l_w_dict['w_v_ts'] * self.loss_MSE_v_ts + \
                    self.l_w_dict['w_u_dir'] * self.loss_MSE_u_dir + self.l_w_dict['w_v_dir'] * self.loss_MSE_v_dir + \
                    self.l_w_dict['w_u_new'] * self.loss_MSE_u_new + self.l_w_dict['w_v_new'] * self.loss_MSE_v_new + \
                    self.l_w_dict['w_u_pde'] * self.loss_MSE_u_pde + self.l_w_dict['w_v_pde'] * self.loss_MSE_v_pde

        return self.loss

    def callback(self):
        """function to extract the loss components at current epoch, print them to console and save to loss csv-file.
        moreover it checks if a current epoch's loss is better than observed before and the case saves model"""

        elapsed_time = time.time() - self.start_time

        # extract loss components for current epoch
        keys = ['time', 'epoch', 'loss', 'MSE_u_ts', 'MSE_v_ts', 'MSE_u_pde', 'MSE_v_pde',
                'MSE_u_dir', 'MSE_v_dir', 'MSE_u_new', 'MSE_v_new']
        vals = [np.round(elapsed_time, 3), self.epoch, cdn(self.loss), cdn(self.loss_MSE_u_ts), cdn(self.loss_MSE_v_ts),
                cdn(self.loss_MSE_u_pde), cdn(self.loss_MSE_v_pde), cdn(self.loss_MSE_u_dir), cdn(self.loss_MSE_v_dir),
                cdn(self.loss_MSE_u_new), cdn(self.loss_MSE_v_new)]

        # print to console
        print("".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys, vals)))

        # create and write csv for saving the loss in each epoch
        if not exists(self.LOSS_PATH):
            write_csv_line(path=self.LOSS_PATH, line=keys)
        write_csv_line(path=self.LOSS_PATH, line=vals)

        # check if best loss improved and save model
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.save_model(path=self.MODEL_PATH)

        # if self.loss > 100000:
        #     self.load_model(path=self.MODEL_PATH)

        # increase epochs counter and start time for next epoch
        self.epoch += 1
        self.start_time = time.time()

    def train(self):
        """ function to train (or load if trained before) the defined model using the Adam and LBFGS optimizer for a
        specified number of epochs (self.epochsAdam, self.epochsLBFGS)"""

        self.best_loss = self.loss_PINN()  # initialize best_loss value

        if exists(
                self.MODEL_PATH):  # load if this was trained before (as we save state dict also for LBFGS optimizer, we cannot load the Adam training epochs and further train with a different LBFGS-setup
            self.load_model(path=self.MODEL_PATH)
        else:
            # Optimizer 1: Adam
            print(f'\n\n\nAdam Optimizer for {self.epochsAdam} epochs: \n\n\n')
            for epoch in range(self.epochsAdam):
                if torch.is_grad_enabled():
                    self.optimizerAdam.zero_grad()
                    loss = self.loss_PINN()
                    loss.backward()
                self.optimizerAdam.step()
                self.scheduler.step(self.loss_MSE_u_ts)
                self.callback()

            # Optimizer 2: L-BFGS
            print(f'\n\n\nL-BFGS Optimizer for {self.epochsLBFGS} epochs: \n\n\n')

            def closure():
                if torch.is_grad_enabled():
                    self.optimizerLBFGS.zero_grad()
                    loss = self.loss_PINN()  # loss needs also float64 format
                    self.loss = loss
                    if loss.requires_grad:  # because every call to closure() does not need the calculation of a gradient inside LBFGS (specifically in line search)
                        loss.backward()
                    self.callback()
                return loss

            self.optimizerLBFGS.step(closure)  # the LBFGS optimizer needs a predefined closure
            self.load_model(path=self.MODEL_PATH)  # load best model in the end (not necessarily the last epoch)

    def predict(self, x: np.ndarray, t: np.ndarray, batch_size=1000):
        """
        function makes a prediction after the model is trained on (x_i, t_i) combinations
        :param: x: numpy array of x values (which will be transformed to tf:Tensor to make prediction)
        :param: t: numpy array of corresponding t values (which will be transformed to tf:Tensor to make prediction)
        :return: prediction of u = real part and v = imaginary part at all requested points (x_i,t_i) back-transformed to numpy
       """

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tfn(x), tfn(t)), batch_size=batch_size, shuffle=False)
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        i = 0

        for x, t in test_loader:

            try:
                u_batch, v_batch = self.net_uv(x.double(), t.double())  # if model requires float32 values (e.g. trained only on Adam)
            except RuntimeError:
                u_batch, v_batch = self.net_uv(x, t)  # if model requires float64 values (e.g if trained on LBFGS)

            u[i * batch_size:(i + 1) * batch_size] = cdn(u_batch)  # backtransformation to numpy
            v[i * batch_size:(i + 1) * batch_size] = cdn(v_batch)
            i += 1

        return u, v



class PINN_NLSE_SA:
    def __init__(self, dic, l_w_dic, neurons_lay, lower_bound, upper_bound, ome_p, depth, hs_LBFGS, lr_LBFGS, lr_Adam, epAdam, epLBFGS, name_for_save):
        """
        add SA: self-adaptive weights: https://arxiv.org/abs/2009.04544
        class to implement and train a Physics-informed Neural Network for the hydrodynamic nonlinear Schrödinger equation according to Chabchoub2016
        (https://www.mdpi.com/2311-5521/1/3/23#FD6-fluids-01-00023) in time-like or space-like form with given initial or boundary conditions
        :param dic: dictionary containing the true-solution data points (xt_ts, u_ts, x_ts), the spatial or temporal boundary points (xt_bl, xt_bu) and the collocation points for the PDE (xtde)
        :param l_w_dic: dictionary containing the chosen loss weights for each loss components
        :param neurons_lay: list of neurons per layer for PINN
        :param lower_bound: of domain [x_min, t_min] or [x_min, t_min, z_min]
        :param upper_bound: of domain [x_max, t_max] or [x_max, t_max, z_max]
        :param ome_p: peak frequency for NLS
        :param depth: water depth [m] for dispersion
        :param hs_LBFGS: history size for the LBFGS optimizer
        :param lr_LBFGS: learning rate for the LBFGS optimizer
        :param epLBFGS: maximum of iterations for the LBFGS optimizer (already needed for initialization) is equal to max_it! So no training-for-loop needed later.
        :param epAdam: epochs of Adam optimizer (actually needed later, but also here for consistency
        :param name_for_save: string with the name for saving the model parameters and figures
        """

        self.loss_scaled = None
        self.l_w_dict = l_w_dic
        self.loss = None  # sum of all loss components
        self.best_loss = None  # to only save models which performance is better than a previously achieved performance
        self.loss_MSE_u_ts = None  # loss components of true-solution data points (u:real part, v:imaginary part)
        self.loss_MSE_v_ts = None
        self.loss_MSE_u_dir = None  # loss components of dirichlet boundary conditions
        self.loss_MSE_v_dir = None
        self.loss_MSE_u_new = None  # loss components of newman boundary conditions
        self.loss_MSE_v_new = None
        self.loss_MSE_u_pde = None  # loss components of PDE-residual
        self.loss_MSE_v_pde = None

        self.name_save = name_for_save
        self.omega_p = ome_p
        self.d = depth
        self.layers = neurons_lay  # number of nodes each layer [2, .., 2]
        self.hs = hs_LBFGS
        self.lr = lr_LBFGS
        self.lrA = lr_Adam
        self.epochsAdam = epAdam
        self.epochsLBFGS = epLBFGS

        self.epoch = 0

        self.start_time = time.time()
        self.MODEL_PATH = f'models/{self.name_save}.pth'
        self.ERROR_PATH = f"errors/{self.name_save}/"
        if not os.path.isdir(self.ERROR_PATH):
            os.mkdir(self.ERROR_PATH)
        self.LOSS_PATH = os.path.join(self.ERROR_PATH, f"loss.csv")
        self.FIGURE_PATH = f"figures/{self.name_save}/"
        if not os.path.isdir(self.FIGURE_PATH):
            os.mkdir(self.FIGURE_PATH)

        # convert data to torch tensors
        self.lower_bound = tfn(lower_bound)
        self.upper_bound = tfn(upper_bound)

        # supervised training data
        self.x_ts = Variable(tfn(dic['xt_ts'][:, 0:1]), requires_grad=True)
        self.t_ts = Variable(tfn(dic['xt_ts'][:, 1:2]), requires_grad=True)
        self.u_ts = Variable(tfn(dic['u_ts']), requires_grad=False)
        self.v_ts = Variable(tfn(dic['v_ts']), requires_grad=False)


        # collocation points inside entire (x, t) domain for pde-residual
        self.x_pde = Variable(tfn(dic['xt_pde'][:, 0:1]), requires_grad=True)
        self.t_pde = Variable(tfn(dic['xt_pde'][:, 1:2]), requires_grad=True)

        # collocation points for periodic boundaries (bl=boundary lower, bu= boundary upper)
        self.x_bl = Variable(tfn(dic['xt_bl'][:, 0:1]), requires_grad=True)
        self.t_bl = Variable(tfn(dic['xt_bl'][:, 1:2]), requires_grad=True)

        self.x_bu = Variable(tfn(dic['xt_bu'][:, 0:1]), requires_grad=True)
        self.t_bu = Variable(tfn(dic['xt_bu'][:, 1:2]), requires_grad=True)

        # values for intermediate plotting
        self.X_star = np.hstack((dic['X'].flatten()[:, None], dic['T'].flatten()[:, None]))

        # stuff for self-adaptive PINNs
        self.lambda_u_dir = Variable(tfn(self.l_w_dict['w_u_dir'] * np.ones(shape=(self.x_bl.shape[0], self.x_bl.shape[1]))), requires_grad=True)
        self.lambda_v_dir = Variable(tfn(self.l_w_dict['w_v_dir'] * np.ones(shape=(self.x_bl.shape[0], self.x_bl.shape[1]))), requires_grad=True)
        self.lambda_u_new = Variable(tfn(self.l_w_dict['w_u_new'] * np.ones(shape=(self.x_bl.shape[0], self.x_bl.shape[1]))), requires_grad=True)
        self.lambda_v_new = Variable(tfn(self.l_w_dict['w_v_new'] * np.ones(shape=(self.x_bl.shape[0], self.x_bl.shape[1]))), requires_grad=True)
        self.lambda_u_pde = Variable(tfn(self.l_w_dict['w_u_pde'] * np.ones(shape=(self.x_pde.shape[0], self.x_pde.shape[1]))), requires_grad=True)
        self.lambda_v_pde = Variable(tfn(self.l_w_dict['w_v_pde'] * np.ones(shape=(self.x_pde.shape[0], self.x_pde.shape[1]))), requires_grad=True)

        self.g_u_dir = torch.square(self.lambda_u_dir)
        self.g_v_dir = torch.square(self.lambda_v_dir)
        self.g_u_new = torch.square(self.lambda_u_new)
        self.g_v_new = torch.square(self.lambda_v_new)
        self.g_u_pde = torch.square(self.lambda_u_pde)
        self.g_v_pde = torch.square(self.lambda_v_pde)

        if self.l_w_dict['data_loss_SA']:  # only do if true
            # usually no data SA weights as input data can be noisy und you dont like to learn the noise
            self.lambda_u_ts = Variable(tfn(self.l_w_dict['w_u_ts'] * np.ones(shape=(self.x_ts.shape[0], self.x_ts.shape[1]))), requires_grad=True)
            self.lambda_v_ts = Variable(tfn(self.l_w_dict['w_v_ts'] * np.ones(shape=(self.x_ts.shape[0], self.x_ts.shape[1]))), requires_grad=True)
            self.g_u_ts = torch.square(self.lambda_u_ts)
            self.g_v_ts = torch.square(self.lambda_v_ts)
            list_optimizer = [self.lambda_u_ts, self.lambda_v_ts, self.lambda_u_dir, self.lambda_v_dir, self.lambda_u_new, self.lambda_v_new, self.lambda_u_pde,
                              self.lambda_v_pde]
        else:
            # fixed weighting factors for the data loss
            self.lambda_u_ts = Variable(tfn(np.array(self.l_w_dict['w_u_ts'])))
            self.lambda_v_ts = Variable(tfn(np.array(self.l_w_dict['w_v_ts'])))
            self.g_u_ts = torch.square(self.lambda_u_ts)
            self.g_v_ts = torch.square(self.lambda_v_ts)
            list_optimizer = [self.lambda_u_dir, self.lambda_v_dir, self.lambda_u_new, self.lambda_v_new, self.lambda_u_pde, self.lambda_v_pde]

        # Initialize NNs
        self.model = Net_Sequential(lb=self.lower_bound[0:2], ub=self.upper_bound[0:2], nodes=self.layers)

        self.model.apply(init_weight_bias)

        self.model.float().to(device)

        self.optimizerAdam = torch.optim.Adam(self.model.parameters(), lr=self.lrA)  # optimizer for minimization
        self.optimizerAdam_w = torch.optim.Adam(list_optimizer, lr=0.001)  # optmizer for maximization
        self.optimizerLBFGS = torch.optim.LBFGS(self.model.parameters(),
                                                lr=self.lr,
                                                max_iter=self.epochsLBFGS,
                                                history_size=self.hs,
                                                line_search_fn=None)

    def load_model(self, path: str):
        """function loads a model, epochs and optimizer if .pth file of previously trained model exists at path location"""
        ll = torch.load(path)
        self.model.load_state_dict(ll['net'])
        self.epoch = ll['epoch']
        self.optimizerAdam.load_state_dict(ll['optim_Adam'])
        self.optimizerAdam_w.load_state_dict(ll['optim_Adam_w'])
        self.optimizerLBFGS.load_state_dict(ll['optim_LBFGS'])
        print(f'\n\n\nLoaded model from epoch {self.epoch}: \n\n\n')

    def save_model(self, path: str):
        """ function saves a model, epochs and optimizer as a pth.file at path"""
        torch.save(
            {'net': self.model.state_dict(),
             'epoch': self.epoch,
             'optim_Adam': self.optimizerAdam.state_dict(),
             'optim_Adam_w': self.optimizerAdam_w.state_dict(),
             'optim_LBFGS': self.optimizerLBFGS.state_dict()},
            path)
        print(f'model checkpoint saved')

    def net_uv(self, x, t):
        """
        function to only accomplish the forward path through th NN to predict u and v (as otherwise also the derivatives would be
        calculated during inference, which is not necessary (computational effort)
        :param x: x coordinates to make prediction at
        :param t: corresponding t coordinates
        :return: prediction of u = real part and v = imaginary part at all requested points (x_i,t_i)
        """

        uv = self.model(x, t)

        u = uv[:, 0:1]  # real part u (None x 1)
        v = uv[:, 1:2]  # imaginary v (None x 1)

        return u, v

    def net_duv(self, x, t):
        """
        function to accomplish the forward path through th NN to predict u and v and also calculate their derivatives
        :param x: x coordinates to make prediction at
        :param t: corresponding t coordinates
        :param doub: boolean whether to calculate with double-precision (float64) format (required for stable LBFGS, not for Adam)
        :return: first and second derivatives w.r.t. x and t.
        """

        u, v = self.net_uv(x=x, t=t)

        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]

        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_tt = torch.autograd.grad(v_t.sum(), t, create_graph=True)[0]

        return u_x, u_xx, u_t, u_tt, v_x, v_xx, v_t, v_tt

    def loss_PINN(self):
        """function calculates the multi-component loss-function by first predicting the required real (u) and imaginary (v) solution and its derivatives and the known
        points (ts), upper and lower boundary (bu, bl) for dirichlet and newman conditions and inside the domain (pde). For calculating the NLS residual, the time-like
        or space-like hydrodynamic NLSE can be initialized. For the newman condition we either use derivatives in time or space direction depending on whether we use
        real initial conditions or actually boundary conditions as th true solution (ts)"""

        # prediction where we know exact solution
        u_pred, v_pred = self.net_uv(x=self.x_ts, t=self.t_ts)

        # prediction at lower and upper boundary
        u_bl, v_bl = self.net_uv(x=self.x_bl, t=self.t_bl)
        u_bl_x, _, u_bl_t, _, v_bl_x, _, v_bl_t, _ = self.net_duv(x=self.x_bl, t=self.t_bl)

        u_bu, v_bu = self.net_uv(x=self.x_bu, t=self.t_bu)
        u_bu_x, _, u_bu_t, _, v_bu_x, _, v_bu_t, _ = self.net_duv(x=self.x_bu, t=self.t_bu)

        u_pde, v_pde = self.net_uv(x=self.x_pde, t=self.t_pde)
        u_pde_x, u_pde_xx, u_pde_t, u_pde_tt, v_pde_x, v_pde_xx, v_pde_t, v_pde_tt = self.net_duv(x=self.x_pde, t=self.t_pde)


        if self.l_w_dict['NLS_form'] == 'time_like':  # time-like NLS residuals inside domain (for hydrodynamic wavemaker problems)
            _, C_g, _, _, delta, nu = NLSE_coefficients_chabchoub(omega_p=self.omega_p, d=self.d)
            f_v_pred = u_pde_x + 1 / C_g * u_pde_t + delta * v_pde_tt + nu * (u_pde ** 2 + v_pde ** 2) * v_pde
            f_u_pred = - v_pde_x - 1 / C_g * v_pde_t + delta * u_pde_tt + nu * (u_pde ** 2 + v_pde ** 2) * u_pde
        elif self.l_w_dict['NLS_form'] == 'space_like':  # space-like form of NLS residuals inside domain
            _, C_g, lamb, mu, _, _ = NLSE_coefficients_chabchoub(omega_p=self.omega_p, d=self.d)
            f_v_pred = u_pde_t + C_g * u_pde_x + lamb * v_pde_xx + mu * (u_pde ** 2 + v_pde ** 2) * v_pde
            f_u_pred = - v_pde_t - C_g * v_pde_x + lamb * u_pde_xx + mu * (u_pde ** 2 + v_pde ** 2) * u_pde
        else:
            print('unknown NLS_form')

        # multi-component loss function
        # batchshaped (no mean!)
        # error between true solution and predicted solution
        MSE_u_ts = torch.square(self.u_ts - u_pred)
        MSE_v_ts = torch.square(self.v_ts - v_pred)
        MSE_u_dir = torch.square(u_bu - u_bl)
        MSE_v_dir = torch.square(v_bu - v_bl)
        # error periodic newman boundary
        if self.l_w_dict['given_ic'] == 'bc':  # if we know the true-solution as a time-series at x=x_0 (thus have an "initial-boundary condition") the newman bc must be satisfied in t-direction
            MSE_u_new = torch.square(u_bu_t - u_bl_t)
            MSE_v_new = torch.square(v_bu_t - v_bl_t)
        elif self.l_w_dict['given_ic'] == 'ic':  # if we know the true-solution as a snapshot along x at t=t_0 (thus have a "real initial condition") the newman bc must be satisfied in x-direction
            MSE_u_new = torch.square(u_bu_x - u_bl_x)
            MSE_v_new = torch.square(v_bu_x - v_bl_x)
        else:
            print('error in given_ic value!')
        # PDE-residual error
        MSE_u_pde = torch.square(f_u_pred)
        MSE_v_pde = torch.square(f_v_pred)

        # means (for callback)
        self.loss_MSE_u_ts = torch.mean(MSE_u_ts)
        self.loss_MSE_v_ts = torch.mean(MSE_v_ts)
        self.loss_MSE_u_dir = torch.mean(MSE_u_dir)
        self.loss_MSE_v_dir = torch.mean(MSE_v_dir)
        self.loss_MSE_u_new = torch.mean(MSE_u_new)
        self.loss_MSE_v_new = torch.mean(MSE_v_new)
        self.loss_MSE_u_pde = torch.mean(MSE_u_pde)
        self.loss_MSE_v_pde = torch.mean(MSE_v_pde)

        if self.l_w_dict['data_loss_SA']:  # only do if true
            # usually no data weights as input data can be noisy und you dont like to learn the noise
            self.g_u_ts = torch.square(self.lambda_u_ts)
            self.g_v_ts = torch.square(self.lambda_v_ts)
        self.g_u_dir = torch.square(self.lambda_u_dir)
        self.g_v_dir = torch.square(self.lambda_v_dir)
        self.g_u_new = torch.square(self.lambda_u_new)
        self.g_v_new = torch.square(self.lambda_v_new)
        self.g_u_pde = torch.square(self.lambda_u_pde)
        self.g_v_pde = torch.square(self.lambda_v_pde)

        # loss without scaling with g(lambdas) for callback
        self.loss = self.loss_MSE_u_ts + self.loss_MSE_v_ts + self.loss_MSE_u_pde + self.loss_MSE_v_pde #+ self.loss_MSE_u_dir + self.loss_MSE_v_dir + self.loss_MSE_u_new + self.loss_MSE_v_new

        # loss with scaling for minimization/maximization (mean applied after multiplication of the self-adaptive weights)
        self.loss_scaled = torch.mean(self.g_u_ts * MSE_u_ts) + torch.mean(self.g_v_ts * MSE_v_ts) + torch.mean(self.g_u_pde * MSE_u_pde) + torch.mean(self.g_v_pde * MSE_v_pde) #+ torch.mean(self.g_u_dir * MSE_u_dir) + torch.mean(self.g_v_dir * MSE_v_dir) + torch.mean(self.g_u_new * MSE_u_new) + torch.mean(self.g_v_new * MSE_v_new)

        return self.loss_scaled

    def callback(self, plot_each):
        """function to extract the loss components at current epoch, print them to console and save to loss csv-file.
        moreover it checks if a current epoch's loss is better than observed before and the case saves model"""

        elapsed_time = time.time() - self.start_time

        # extract loss components for current epoch
        keys = ['epoch', 'loss', 'loss_scaled', 'MSE_u_ts', 'MSE_v_ts', 'MSE_u_pde', 'MSE_v_pde',
                'MSE_u_dir', 'MSE_v_dir', 'MSE_u_new', 'MSE_v_new']
        vals = [self.epoch, cdn(self.loss), cdn(self.loss_scaled), cdn(self.loss_MSE_u_ts), cdn(self.loss_MSE_v_ts),
                cdn(self.loss_MSE_u_pde), cdn(self.loss_MSE_v_pde), cdn(self.loss_MSE_u_dir), cdn(self.loss_MSE_v_dir),
                cdn(self.loss_MSE_u_new), cdn(self.loss_MSE_v_new)]

        # print to console
        print(f'time: {np.round(elapsed_time, 3)} s')
        print("".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys, vals)))
        print(f'g(lam_u_data) - mean: {torch.mean(self.g_u_ts)} min: {torch.min(self.g_u_ts)}, max: {torch.max(self.g_u_ts)}')
        print(f'g(lam_v_data) - mean: {torch.mean(self.g_v_ts)} min: {torch.min(self.g_v_ts)}, max: {torch.max(self.g_v_ts)}')
        print(f'g(lam_u_pde) - mean: {torch.mean(self.g_u_pde)} min: {torch.min(self.g_u_pde)}, max: {torch.max(self.g_u_pde)}')
        print(f'g(lam_v_pde) - mean: {torch.mean(self.g_v_pde)} min: {torch.min(self.g_v_pde)}, max: {torch.max(self.g_v_pde)}')
        print(f'g(lam_u_dir) - mean: {torch.mean(self.g_u_dir)} min: {torch.min(self.g_u_dir)}, max: {torch.max(self.g_u_dir)}')
        print(f'g(lam_v_dir) - mean: {torch.mean(self.g_v_dir)} min: {torch.min(self.g_v_dir)}, max: {torch.max(self.g_v_dir)}')
        print(f'g(lam_u_new) - mean: {torch.mean(self.g_u_new)} min: {torch.min(self.g_u_new)}, max: {torch.max(self.g_u_new)}')
        print(f'g(lam_v_new) - mean: {torch.mean(self.g_v_new)} min: {torch.min(self.g_v_new)}, max: {torch.max(self.g_v_new)}')

        # create and write csv for saving the loss in each epoch
        if not exists(self.LOSS_PATH):
            write_csv_line(path=self.LOSS_PATH, line=keys)
        write_csv_line(path=self.LOSS_PATH, line=vals)

        # check if best loss improved and save model
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.save_model(path=self.MODEL_PATH)

        # plot predictions each plot_each epochs
        if self.epoch % plot_each == 0:
            u_pred, v_pred = self.predict(self.X_star[:, 0:1], self.X_star[:, 1:2])
            np.savez(os.path.join(self.ERROR_PATH, f'pred_epoch_{self.epoch}'), u_pred=u_pred, v_pred=v_pred, epoch=self.epoch),

            if self.epoch <= self.epochsAdam + 2:
                np.savez(os.path.join(self.ERROR_PATH, f'lambdas_epoch_{self.epoch}'), lambda_u_ts=cdn(self.lambda_u_ts),
                         lambda_v_ts=cdn(self.lambda_v_ts), lambda_u_pde=cdn(self.lambda_u_pde), lambda_v_pde=cdn(self.lambda_v_pde),
                         lambda_u_dir=cdn(self.lambda_u_dir), lambda_v_dir=cdn(self.lambda_v_dir), lambda_u_new=cdn(self.lambda_u_new),
                         lambda_v_new=cdn(self.lambda_v_new), epoch=self.epoch)

        if self.epoch == 0:
            np.savez(os.path.join(self.ERROR_PATH, f'collpoints_all_epochs'), x_ts=cdn(self.x_ts), t_ts=cdn(self.t_ts),
                     x_pde=cdn(self.x_pde), t_pde=cdn(self.t_pde), x_bu=cdn(self.x_bu), t_bu=cdn(self.t_bu),
                     x_bl=cdn(self.x_bl), t_bl=cdn(self.t_bl))

        # increase epochs counter and start time for next epoch
        self.epoch += 1
        self.start_time = time.time()

    def train(self, plot_each=1000):
        """ function to train (or load if trained before) the defined model using the Adam and LBFGS optimizer for a
        specified number of epochs (self.epochsAdam, self.epochsLBFGS)"""

        self.best_loss = self.loss_PINN()  # initialize best_loss value

        if exists(
                self.MODEL_PATH):  # load if this was trained before (as we save state dict also for LBFGS optimizer, we cannot load the Adam training epochs and further train with a different LBFGS-setup
            self.load_model(path=self.MODEL_PATH)
        else:
            # Optimizer 1: Adam
            print(f'\n\n\nAdam Optimizer for {self.epochsAdam} epochs: \n\n\n')

            def closure_pos():
                self.optimizerAdam.zero_grad()
                loss = self.loss_PINN()
                loss.backward(retain_graph=True)
                return loss

            def closure_neg():
                self.optimizerAdam_w.zero_grad()
                loss_n = -1 * self.loss_PINN()
                loss_n.backward(retain_graph=True)
                return loss_n

            for epoch in range(self.epochsAdam):
                if torch.is_grad_enabled():
                    self.optimizerAdam.step(closure_pos)
                    self.optimizerAdam_w.step(closure_neg)
                    self.callback(plot_each)

            # Optimizer 2: L-BFGS
            print(f'\n\n\nL-BFGS Optimizer for {self.epochsLBFGS} epochs: \n\n\n')

            def closure():
                if torch.is_grad_enabled():
                    self.optimizerLBFGS.zero_grad()
                    loss = self.loss_PINN()
                    if loss.requires_grad:  # because every call to closure() does not need the calculation of a gradient inside LBFGS (specifically in line search)
                        loss.backward()
                    self.callback(plot_each)
                return loss

            self.optimizerLBFGS.step(closure)  # the LBFGS optimizer needs a predefined closure
            self.load_model(path=self.MODEL_PATH)  # load best model in the end (not necessarily the last epoch)

    def predict(self, x: np.ndarray, t: np.ndarray, batch_size=1000):
        """
        function makes a prediction after the model is trained on (x_i, t_i) combinations
        :param: x: numpy array of x values (which will be transformed to tf:Tensor to make prediction)
        :param: t: numpy array of corresponding t values (which will be transformed to tf:Tensor to make prediction)
        :return: prediction of u = real part and v = imaginary part at all requested points (x_i,t_i) back-transformed to numpy
       """

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tfn(x), tfn(t)), batch_size=batch_size, shuffle=False)
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        i = 0

        for x, t in test_loader:

            try:
                u_batch, v_batch = self.net_uv(x.double(), t.double())  # if model requires float32 values (e.g. trained only on Adam)
            except RuntimeError:
                u_batch, v_batch = self.net_uv(x, t)  # if model requires float64 values (e.g if trained on LBFGS)

            u[i * batch_size:(i + 1) * batch_size] = cdn(u_batch)  # backtransformation to numpy
            v[i * batch_size:(i + 1) * batch_size] = cdn(v_batch)
            i += 1

        return u, v



class PINN_NLSE_SA_noBCs:
    def __init__(self, dic, l_w_dic, neurons_lay, lower_bound, upper_bound, ome_p, depth, hs_LBFGS, lr_LBFGS, lr_Adam, epAdam, epLBFGS, name_for_save):
        """
        add SA: self-adaptive weights: https://arxiv.org/abs/2009.04544
        class to implement and train a Physics-informed Neural Network for the hydrodynamic nonlinear Schrödinger equation according to Chabchoub2016
        (https://www.mdpi.com/2311-5521/1/3/23#FD6-fluids-01-00023) in time-like or space-like form with given initial or boundary conditions
        :param dic: dictionary containing the true-solution data points (xt_ts, u_ts, x_ts), the spatial or temporal boundary points (xt_bl, xt_bu) and the collocation points for the PDE (xtde)
        :param l_w_dic: dictionary containing the chosen loss weights for each loss components
        :param neurons_lay: list of neurons per layer for PINN
        :param lower_bound: of domain [x_min, t_min] or [x_min, t_min, z_min]
        :param upper_bound: of domain [x_max, t_max] or [x_max, t_max, z_max]
        :param ome_p: peak frequency for NLS
        :param depth: water depth [m] for dispersion
        :param hs_LBFGS: history size for the LBFGS optimizer
        :param lr_LBFGS: learning rate for the LBFGS optimizer
        :param epLBFGS: maximum of iterations for the LBFGS optimizer (already needed for initialization) is equal to max_it! So no training-for-loop needed later.
        :param epAdam: epochs of Adam optimizer (actually needed later, but also here for consistency
        :param name_for_save: string with the name for saving the model parameters and figures
        """

        self.loss_scaled = None
        self.l_w_dict = l_w_dic
        self.loss = None  # sum of all loss components
        self.best_loss = None  # to only save models which performance is better than a previously achieved performance
        self.loss_MSE_u_ts = None  # loss components of true-solution data points (u:real part, v:imaginary part)
        self.loss_MSE_v_ts = None
        self.loss_MSE_u_pde = None  # loss components of dirichlet boundary conditions
        self.loss_MSE_v_pde = None

        self.name_save = name_for_save
        self.omega_p = ome_p
        self.d = depth
        self.layers = neurons_lay  # number of nodes each layer [2, .., 2]
        self.hs = hs_LBFGS
        self.lr = lr_LBFGS
        self.lrA = lr_Adam
        self.epochsAdam = epAdam
        self.epochsLBFGS = epLBFGS

        self.epoch = 0

        self.start_time = time.time()
        self.MODEL_PATH = f'models/{self.name_save}.pth'
        self.ERROR_PATH = f"errors/{self.name_save}/"
        if not os.path.isdir(self.ERROR_PATH):
            os.mkdir(self.ERROR_PATH)
        self.LOSS_PATH = os.path.join(self.ERROR_PATH, f"loss.csv")
        self.FIGURE_PATH = f"figures/{self.name_save}/"
        if not os.path.isdir(self.FIGURE_PATH):
            os.mkdir(self.FIGURE_PATH)

        # convert data to torch tensors
        self.lower_bound = tfn(lower_bound)
        self.upper_bound = tfn(upper_bound)

        # supervised training data
        self.x_ts = Variable(tfn(dic['xt_ts'][:, 0:1]), requires_grad=True)
        self.t_ts = Variable(tfn(dic['xt_ts'][:, 1:2]), requires_grad=True)
        self.u_ts = Variable(tfn(dic['u_ts']), requires_grad=False)
        self.v_ts = Variable(tfn(dic['v_ts']), requires_grad=False)


        # collocation points inside entire (x, t) domain for pde-residual
        self.x_pde = Variable(tfn(dic['xt_pde'][:, 0:1]), requires_grad=True)
        self.t_pde = Variable(tfn(dic['xt_pde'][:, 1:2]), requires_grad=True)

        # values for intermediate plotting
        self.X_star = np.hstack((dic['X'].flatten()[:, None], dic['T'].flatten()[:, None]))

        # stuff for self-adaptive PINNs
        self.lambda_u_pde = Variable(tfn(self.l_w_dict['w_u_pde'] * np.ones(shape=(self.x_pde.shape[0], self.x_pde.shape[1]))), requires_grad=True)
        self.lambda_v_pde = Variable(tfn(self.l_w_dict['w_v_pde'] * np.ones(shape=(self.x_pde.shape[0], self.x_pde.shape[1]))), requires_grad=True)

        self.g_u_pde = torch.square(self.lambda_u_pde)
        self.g_v_pde = torch.square(self.lambda_v_pde)

        if self.l_w_dict['data_loss_SA']:  # only do if true
            # usually no data SA weights as input data can be noisy und you dont like to learn the noise
            self.lambda_u_ts = Variable(tfn(self.l_w_dict['w_u_ts'] * np.ones(shape=(self.x_ts.shape[0], self.x_ts.shape[1]))), requires_grad=True)
            self.lambda_v_ts = Variable(tfn(self.l_w_dict['w_v_ts'] * np.ones(shape=(self.x_ts.shape[0], self.x_ts.shape[1]))), requires_grad=True)
            self.g_u_ts = torch.square(self.lambda_u_ts)
            self.g_v_ts = torch.square(self.lambda_v_ts)
            list_optimizer = [self.lambda_u_ts, self.lambda_v_ts, self.lambda_u_pde, self.lambda_v_pde]
        else:
            # fixed weighting factors for the data loss
            self.lambda_u_ts = Variable(tfn(np.array(self.l_w_dict['w_u_ts'])))
            self.lambda_v_ts = Variable(tfn(np.array(self.l_w_dict['w_v_ts'])))
            self.g_u_ts = torch.square(self.lambda_u_ts)
            self.g_v_ts = torch.square(self.lambda_v_ts)
            list_optimizer = [self.lambda_u_pde, self.lambda_v_pde]

        # Initialize NNs
        self.model = Net_Sequential(lb=self.lower_bound[0:2], ub=self.upper_bound[0:2], nodes=self.layers)

        self.model.apply(init_weight_bias)

        self.model.float().to(device)

        self.optimizerAdam = torch.optim.Adam(self.model.parameters(), lr=self.lrA)  # optimizer for minimization
        self.optimizerAdam_w = torch.optim.Adam(list_optimizer, lr=0.001)  # optmizer for maximization
        self.optimizerLBFGS = torch.optim.LBFGS(self.model.parameters(),
                                                lr=self.lr,
                                                max_iter=self.epochsLBFGS,
                                                history_size=self.hs,
                                                line_search_fn=None)

    def load_model(self, path: str):
        """function loads a model, epochs and optimizer if .pth file of previously trained model exists at path location"""
        ll = torch.load(path)
        self.model.load_state_dict(ll['net'])
        self.epoch = ll['epoch']
        self.optimizerAdam.load_state_dict(ll['optim_Adam'])
        self.optimizerAdam_w.load_state_dict(ll['optim_Adam_w'])
        self.optimizerLBFGS.load_state_dict(ll['optim_LBFGS'])
        print(f'\n\n\nLoaded model from epoch {self.epoch}: \n\n\n')

    def save_model(self, path: str):
        """ function saves a model, epochs and optimizer as a pth.file at path"""
        torch.save(
            {'net': self.model.state_dict(),
             'epoch': self.epoch,
             'optim_Adam': self.optimizerAdam.state_dict(),
             'optim_Adam_w': self.optimizerAdam_w.state_dict(),
             'optim_LBFGS': self.optimizerLBFGS.state_dict()},
            path)
        print(f'model checkpoint saved')

    def net_uv(self, x, t):
        """
        function to only accomplish the forward path through th NN to predict u and v (as otherwise also the derivatives would be
        calculated during inference, which is not necessary (computational effort)
        :param x: x coordinates to make prediction at
        :param t: corresponding t coordinates
        :return: prediction of u = real part and v = imaginary part at all requested points (x_i,t_i)
        """

        uv = self.model(x, t)

        u = uv[:, 0:1]  # real part u (None x 1)
        v = uv[:, 1:2]  # imaginary v (None x 1)

        return u, v

    def net_duv(self, x, t):
        """
        function to accomplish the forward path through th NN to predict u and v and also calculate their derivatives
        :param x: x coordinates to make prediction at
        :param t: corresponding t coordinates
        :param doub: boolean whether to calculate with double-precision (float64) format (required for stable LBFGS, not for Adam)
        :return: first and second derivatives w.r.t. x and t.
        """

        u, v = self.net_uv(x=x, t=t)

        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t.sum(), t, create_graph=True)[0]

        v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
        v_tt = torch.autograd.grad(v_t.sum(), t, create_graph=True)[0]

        return u_x, u_xx, u_t, u_tt, v_x, v_xx, v_t, v_tt

    def loss_PINN(self):
        """function calculates the multi-component loss-function by first predicting the required real (u) and imaginary (v) solution and its derivatives and the known
        points (ts), upper and lower boundary (bu, bl) for dirichlet and newman conditions and inside the domain (pde). For calculating the NLS residual, the time-like
        or space-like hydrodynamic NLSE can be initialized. For the newman condition we either use derivatives in time or space direction depending on whether we use
        real initial conditions or actually boundary conditions as th true solution (ts)"""

        # prediction where we know exact solution
        u_pred, v_pred = self.net_uv(x=self.x_ts, t=self.t_ts)

        u_pde, v_pde = self.net_uv(x=self.x_pde, t=self.t_pde)
        u_pde_x, u_pde_xx, u_pde_t, u_pde_tt, v_pde_x, v_pde_xx, v_pde_t, v_pde_tt = self.net_duv(x=self.x_pde, t=self.t_pde)


        if self.l_w_dict['NLS_form'] == 'time_like':  # time-like NLS residuals inside domain (for hydrodynamic wavemaker problems)
            _, C_g, _, _, delta, nu = NLSE_coefficients_chabchoub(omega_p=self.omega_p, d=self.d)
            # f_v_pred = u_pde_x + 1 / C_g * u_pde_t + delta * v_pde_tt + nu * (u_pde ** 2 + v_pde ** 2) * v_pde
            # f_u_pred = - v_pde_x - 1 / C_g * v_pde_t + delta * u_pde_tt + nu * (u_pde ** 2 + v_pde ** 2) * u_pde
        elif self.l_w_dict['NLS_form'] == 'space_like':  # space-like form of NLS residuals inside domain
            _, C_g, lamb, mu, _, _ = NLSE_coefficients_chabchoub(omega_p=self.omega_p, d=self.d)
            f_v_pred = u_pde_t + C_g * u_pde_x + lamb * v_pde_xx + mu * (u_pde ** 2 + v_pde ** 2) * v_pde
            f_u_pred = - v_pde_t - C_g * v_pde_x + lamb * u_pde_xx + mu * (u_pde ** 2 + v_pde ** 2) * u_pde
        else:
            print('unknown NLS_form')

        # multi-component loss function
        # batchshaped (no mean!)
        # error between true solution and predicted solution
        MSE_u_ts = torch.square(self.u_ts - u_pred)
        MSE_v_ts = torch.square(self.v_ts - v_pred)

        # PDE-residual error
        MSE_u_pde = torch.square(f_u_pred)
        MSE_v_pde = torch.square(f_v_pred)

        # means (for callback)
        self.loss_MSE_u_ts = torch.mean(MSE_u_ts)
        self.loss_MSE_v_ts = torch.mean(MSE_v_ts)
        self.loss_MSE_u_pde = torch.mean(MSE_u_pde)
        self.loss_MSE_v_pde = torch.mean(MSE_v_pde)

        if self.l_w_dict['data_loss_SA']:  # only do if true
            # usually no data weights as input data can be noisy und you dont like to learn the noise
            self.g_u_ts = torch.square(self.lambda_u_ts)
            self.g_v_ts = torch.square(self.lambda_v_ts)
        self.g_u_pde = torch.square(self.lambda_u_pde)
        self.g_v_pde = torch.square(self.lambda_v_pde)

        # loss without scaling with g(lambdas) for callback
        self.loss = self.loss_MSE_u_ts + self.loss_MSE_v_ts + self.loss_MSE_u_pde + self.loss_MSE_v_pde

        # loss with scaling for minimization/maximization (mean applied after multiplication of the self-adaptive weights)
        self.loss_scaled = torch.mean(self.g_u_ts * MSE_u_ts) + torch.mean(self.g_v_ts * MSE_v_ts) + torch.mean(self.g_u_pde * MSE_u_pde) + torch.mean(self.g_v_pde * MSE_v_pde)

        return self.loss_scaled

    def callback(self, plot_each):
        """function to extract the loss components at current epoch, print them to console and save to loss csv-file.
        moreover it checks if a current epoch's loss is better than observed before and the case saves model"""

        elapsed_time = time.time() - self.start_time

        # extract loss components for current epoch
        keys = ['epoch', 'loss', 'loss_scaled', 'MSE_u_ts', 'MSE_v_ts', 'MSE_u_pde', 'MSE_v_pde']
        vals = [self.epoch, cdn(self.loss), cdn(self.loss_scaled), cdn(self.loss_MSE_u_ts), cdn(self.loss_MSE_v_ts),
                cdn(self.loss_MSE_u_pde), cdn(self.loss_MSE_v_pde)]

        # print to console
        print(f'time: {np.round(elapsed_time, 3)} s')
        print("".join(str(key) + ": " + str(value) + ", " for key, value in zip(keys, vals)))
        print(f'g(lam_u_data) - mean: {torch.mean(self.g_u_ts)} min: {torch.min(self.g_u_ts)}, max: {torch.max(self.g_u_ts)}')
        print(f'g(lam_v_data) - mean: {torch.mean(self.g_v_ts)} min: {torch.min(self.g_v_ts)}, max: {torch.max(self.g_v_ts)}')
        print(f'g(lam_u_pde) - mean: {torch.mean(self.g_u_pde)} min: {torch.min(self.g_u_pde)}, max: {torch.max(self.g_u_pde)}')
        print(f'g(lam_v_pde) - mean: {torch.mean(self.g_v_pde)} min: {torch.min(self.g_v_pde)}, max: {torch.max(self.g_v_pde)}')

        # create and write csv for saving the loss in each epoch
        if not exists(self.LOSS_PATH):
            write_csv_line(path=self.LOSS_PATH, line=keys)
        write_csv_line(path=self.LOSS_PATH, line=vals)

        # check if best loss improved and save model
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.save_model(path=self.MODEL_PATH)

        # plot predictions each plot_each epochs
        if self.epoch % plot_each == 0:
            u_pred, v_pred = self.predict(self.X_star[:, 0:1], self.X_star[:, 1:2])
            np.savez(os.path.join(self.ERROR_PATH, f'pred_epoch_{self.epoch}'), u_pred=u_pred, v_pred=v_pred, epoch=self.epoch),

            if self.epoch <= self.epochsAdam + 2:
                np.savez(os.path.join(self.ERROR_PATH, f'lambdas_epoch_{self.epoch}'), lambda_u_ts=cdn(self.lambda_u_ts),
                         lambda_v_ts=cdn(self.lambda_v_ts), lambda_u_pde=cdn(self.lambda_u_pde), lambda_v_pde=cdn(self.lambda_v_pde), epoch=self.epoch)

        if self.epoch == 0:
            np.savez(os.path.join(self.ERROR_PATH, f'collpoints_all_epochs'), x_ts=cdn(self.x_ts), t_ts=cdn(self.t_ts),
                     x_pde=cdn(self.x_pde), t_pde=cdn(self.t_pde))

        # increase epochs counter and start time for next epoch
        self.epoch += 1
        self.start_time = time.time()

    def train(self, plot_each=1000):
        """ function to train (or load if trained before) the defined model using the Adam and LBFGS optimizer for a
        specified number of epochs (self.epochsAdam, self.epochsLBFGS)"""

        self.best_loss = self.loss_PINN()  # initialize best_loss value

        if exists(
                self.MODEL_PATH):  # load if this was trained before (as we save state dict also for LBFGS optimizer, we cannot load the Adam training epochs and further train with a different LBFGS-setup
            self.load_model(path=self.MODEL_PATH)
        else:
            # Optimizer 1: Adam
            print(f'\n\n\nAdam Optimizer for {self.epochsAdam} epochs: \n\n\n')

            def closure_pos():
                self.optimizerAdam.zero_grad()
                loss = self.loss_PINN()
                loss.backward(retain_graph=True)
                return loss

            def closure_neg():
                self.optimizerAdam_w.zero_grad()
                loss_n = -1 * self.loss_PINN()
                loss_n.backward(retain_graph=True)
                return loss_n

            for epoch in range(self.epochsAdam):
                if torch.is_grad_enabled():
                    self.optimizerAdam.step(closure_pos)
                    self.optimizerAdam_w.step(closure_neg)
                    self.callback(plot_each)

            # Optimizer 2: L-BFGS
            print(f'\n\n\nL-BFGS Optimizer for {self.epochsLBFGS} epochs: \n\n\n')

            def closure():
                if torch.is_grad_enabled():
                    self.optimizerLBFGS.zero_grad()
                    loss = self.loss_PINN()
                    if loss.requires_grad:  # because every call to closure() does not need the calculation of a gradient inside LBFGS (specifically in line search)
                        loss.backward()
                    self.callback(plot_each)
                return loss

            self.optimizerLBFGS.step(closure)  # the LBFGS optimizer needs a predefined closure
            self.load_model(path=self.MODEL_PATH)  # load best model in the end (not necessarily the last epoch)

    def predict(self, x: np.ndarray, t: np.ndarray, batch_size=1000):
        """
        function makes a prediction after the model is trained on (x_i, t_i) combinations
        :param: x: numpy array of x values (which will be transformed to tf:Tensor to make prediction)
        :param: t: numpy array of corresponding t values (which will be transformed to tf:Tensor to make prediction)
        :return: prediction of u = real part and v = imaginary part at all requested points (x_i,t_i) back-transformed to numpy
       """

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tfn(x), tfn(t)), batch_size=batch_size, shuffle=False)
        u = np.zeros_like(x)
        v = np.zeros_like(x)
        i = 0

        for x, t in test_loader:

            try:
                u_batch, v_batch = self.net_uv(x.double(), t.double())  # if model requires float32 values (e.g. trained only on Adam)
            except RuntimeError:
                u_batch, v_batch = self.net_uv(x, t)  # if model requires float64 values (e.g if trained on LBFGS)

            u[i * batch_size:(i + 1) * batch_size] = cdn(u_batch)  # backtransformation to numpy
            v[i * batch_size:(i + 1) * batch_size] = cdn(v_batch)
            i += 1

        return u, v
