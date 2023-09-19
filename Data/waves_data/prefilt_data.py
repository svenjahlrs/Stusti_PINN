import numpy as np
from pathlib import Path
import random
import time
from csv import writer


def write_csv_line(path: str, line):
    """ writes a new line to a csv file at path"""
    with open(path, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(line)

np.random.seed(1234)
data_path = '/mnt/c/Users/svenj/Documents/PINN_NLS/Data/waves_data'

params = np.loadtxt(Path(data_path, 'params_eps_ome_gam.txt'))

eps_unique = np.unique(params[:, 0])
ome_unique = np.unique(params[:, 1])
gam_unique = np.unique(params[:, 2])

samples_per_com = 10
indices = []

eps_ome_gam = np.loadtxt(Path(data_path, 'params_eps_ome_gam.txt'))
eta0 = np.loadtxt(Path(data_path, 'data_WB.txt'))
print('loaded eta0')
eta3 = np.loadtxt(Path(data_path, 'data_3m.txt'))
print('loaded eta3')
eta4 = np.loadtxt(Path(data_path, 'data_4m.txt'))
print('loaded eta4')
eta5 = np.loadtxt(Path(data_path, 'data_5m.txt'))
print('loaded eta5')
eta6 = np.loadtxt(Path(data_path, 'data_6m.txt'))
print('loaded eta6')

for e in eps_unique:
    for w in ome_unique:
        for g in gam_unique:
            for i, par_comb in enumerate(params):
                if np.equal(par_comb, np.array([e, w, g])).all():
                    indices.append(i)

            ind_short =random.sample(indices, samples_per_com)
            print(f'indices ({len(indices)}) for eps={e}, ome={w}, gam={g}: {ind_short}')
            for ii in ind_short:
                tic = time.time()

                write_csv_line(path=(Path(data_path, 'params_eps_ome_gam_short.csv')), line=eps_ome_gam[ii, :])
                write_csv_line(path=(Path(data_path, 'data_WB_short.csv')), line=eta0[ii, :])
                write_csv_line(path=(Path(data_path, 'data_3m_short.csv')), line=eta3[ii, :])
                write_csv_line(path=(Path(data_path, 'data_4m_short.csv')), line=eta4[ii, :])
                write_csv_line(path=(Path(data_path, 'data_5m_short.csv')), line=eta5[ii, :])
                write_csv_line(path=(Path(data_path, 'data_6m_short.csv')), line=eta6[ii, :])
                print(time.time()-tic)
            indices = []

