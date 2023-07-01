import os
import glob
import argparse

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--number', help='from 0 to 7', default=0, type=int)
parser.add_argument('--parameter', help='0, 1 or 2', default=0, type=int)
parser.add_argument('--type', type=str,help='tracer type to be selected')
args = parser.parse_args()

base = '/global/cfs/cdirs/desi/survey/catalogs/main/mocks/FirstGenMocks/AbacusSummit/Y1/mock1/LSScats/'
data_dir = base + 'blinded/jmena/test_w0-*/LSScats/blinded/'

fn_list = sorted(glob.glob(os.path.join(data_dir, 'blinded_parameters_{}.csv'.format(args.type))))

folder = fn_list[args.number]

A = np.loadtxt(folder, delimiter=',', skiprows=1)[args.parameter]

print(A)
