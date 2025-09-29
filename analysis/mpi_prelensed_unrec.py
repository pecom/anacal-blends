import numpy as np
import pandas as pd
from astropy.table import Table, hstack, vstack, join
import scipy.stats as stats
import os, sys
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--cbin", type=int)
parser.add_argument("-q", "--quarter", type=int)
parser.add_argument("-p", "--pixels", type=int)

args = parser.parse_args()
cbin = int(args.cbin)
qbin = int(args.quarter)
pix = int(args.pixels)

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

sys.path.insert(1, f'{hdir}/codes/friendly/friendly')
sys.path.insert(1, f'{hdir}/codes/friendly')
from friendly.utils import FCatalog
from friendly.matchers.kdtree import FKDTree
from friendly.pruners.mag_diff import MagDiffPruner


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

ods_cols = [b+'_ab' for b in 'ugrizy']
OneDegSq = Table.read(os.environ['CATSIM_DIR'] + '/OneDegSq.fits')
pixel_scale = 0.2


def group2table(groups):
    ndx1_name = 'kdtree_idx1'
    ndx2_name = 'kdtree_idx2'

    ndx1s = []
    ndx2s = []
    blend_diff = []
    for g in groups:
        ndx1s.append(g.idx1)
        ndx2s.append(g.idx2)
        blend_diff.append(len(g.idx2) - len(g.idx1))

    blend_table = Table(data=[ndx1s, ndx2s, blend_diff], names=[ndx1_name, ndx2_name, 'blend_diff'], dtype=[object, object, int])
    blend_table['kdtree_idx1'] = blend_table['kdtree_idx1'].reshape((len(blend_table)))
    return blend_table

# def match_seed(seed, cdir='catalogs_bin2'):
#     print(f"Working on object {seed}")
#     match cdir:
#         case "catalogs_bin1":
#             input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_g_mode3.fits')
#         case "catalogs_bin2":
#             input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_r.fits')
#         case "catalogs_constant0":
#             input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_constant_mode0.fits')
#         case "catalogs_constant1":
#             input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_constant_mode1.fits')
#         case _:
#             print("Unclear input truth! Please fix!")
#             sys.exit()

def match_seed(seed, mode=1, pix=10):
    print(f"Working on object {seed}")
    match mode:
        case 0:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs_unlensed/input_catalog_{seed}_constant_mode0.fits')
            cdir = 'catalogs_constant0_unlensed'
        case 1:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs_unlensed/input_catalog_{seed}_i_mode3.fits')
            cdir = 'catalogs_bin1_unlensed'
        case 2:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs_unlensed/input_catalog_{seed}_i_mode1.fits')
            cdir = 'catalogs_bin2_unlensed'
        case _:
            print("Unclear input truth! Please fix!")
            sys.exit()


    input_truth.add_index('index')
    ods_phot = OneDegSq[input_truth['index']][ods_cols]
    phot_truth = hstack((input_truth, ods_phot))
    bright_filt = phot_truth['i_ab'] < 27
    phot_fc1 = FCatalog(phot_truth, 'index', columns=phot_truth.columns)
    phot_fc2 = FCatalog(phot_truth[bright_filt], 'index', columns=phot_truth.columns)

    candidate_boost_factor = pix
    lazy_kdtree = FKDTree({'search_rad': candidate_boost_factor})
    kdtree_groups_input, _ = lazy_kdtree(phot_fc1, phot_fc2, {'RA1': 'prelensed_image_x', 'DEC1': 'prelensed_image_y',
                                                              'RA2': 'prelensed_image_x', 'DEC2': 'prelensed_image_y'})

    g2l = [len(k.idx2) for k in kdtree_groups_input]
    g2l = np.array(g2l)

    input_truth['match_num'] = g2l

    full_cat = Table.read(f'{pdir}/anacal_blends/{cdir}/catalog_{seed}.fits')

    blend_lbl = np.zeros(len(full_cat))
    for i, dc in enumerate(full_cat['truth_index']):
        prelensed_lbl = input_truth.loc[dc]['match_num']
        if np.any(prelensed_lbl > 1):
            blend_lbl[i] = 1

    np.save(f'{pdir}/anacal_blends/unrec/{cdir}_{seed}_unrec_unlensed_{pix}pix.npy', blend_lbl)
    return None


send_ndxs = None

if rank==0:
    match qbin:
        case 0:
            ndxs = np.arange(10)
        case 1:
            ndxs = np.arange(10240)
        case 2:
            ndxs = np.arange(10240, 20480)
        case 3:
            ndxs = np.arange(20480, 30720)
        case 4:
            ndxs = np.arange(3072, 40960)
    split_ndxs = np.array_split(ndxs, size)
else:
    split_ndxs = None

split_ndxs = comm.scatter(split_ndxs, root=0)

print(f"Looking at {len(split_ndxs)} objects at {rank}")

for ndx in split_ndxs:
    # match_seed(ndx, cdir=f'catalogs_bin{cbin}')
    match_seed(ndx, mode=cbin, pix=pix)

print(f"Done with rank {rank}")
