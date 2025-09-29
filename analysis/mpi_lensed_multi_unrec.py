import numpy as np
import pandas as pd
from astropy.table import Table, hstack, vstack, join
import scipy.stats as stats
from scipy.spatial import KDTree
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

# sys.path.insert(1, f'{hdir}/codes/friendly/friendly')
# sys.path.insert(1, f'{hdir}/codes/friendly')
# from friendly.utils import FCatalog
# from friendly.matchers.kdtree import FKDTree
# from friendly.pruners.mag_diff import MagDiffPruner


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


def get_unrec_bl_multi_truth(truth, maglim = 27, zbin=1, npix=pix):
    bright_filt = truth['i_ab'] < maglim
    truth['is_blend'] = 0.

    # If I'm applying a shear to bin 1,
    # I want to select on the elements in bin 2
    match zbin:
        case 1:
            cat1_filt = truth['z'] > 1
            cat2_filt = truth['z'] < 1
        case 2:
            cat1_filt = truth['z'] < 1
            cat2_filt = truth['z'] > 1

    # phot_fc1 = FCatalog(truth[cat1_filt], 'index', columns=truth.columns)
    # phot_fc2 = FCatalog(truth[bright_filt & cat2_filt], 'index', columns=truth.columns)

    # candidate_boost_factor = npix
    # lazy_kdtree = FKDTree({'search_rad': candidate_boost_factor})
    # kdtree_groups_multi, _ = lazy_kdtree(phot_fc1, phot_fc2, {'RA1': 'image_x', 'DEC1': 'image_y',
    #                                                          'RA2': 'image_x', 'DEC2': 'image_y'})

    phot_f1 = truth[cat1_filt]
    phot_f2 = truth[bright_filt & cat2_filt]

    phot2_ndxs_all = phot_f2['index']

    coords1 = np.vstack((phot_f1['image_x'], phot_f1['image_y'])).T
    coords2 = np.vstack((phot_f2['image_x'], phot_f2['image_y'])).T

    otherbin_tree = KDTree(coords2)
    ndxs = otherbin_tree.query_ball_point(coords1, r=npix)

    for i, row in enumerate(phot_f1):
        if len(ndxs[i]) > 0:
            truth.loc[row['index']]['is_blend'] = 1
            phot2_ndxs = phot2_ndxs_all[ndxs[i]]
            truth.loc[phot2_ndxs]['is_blend'] = 1

    return ndxs

def match_seed(seed, mode=1):
    print(f"Working on object {seed}")
    match mode:
        case 1:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs_unlensed/input_catalog_{seed}_i_mode3.fits')
            # input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_g_mode3.fits')
            cdir = 'catalogs_bin1'
        case 2:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs_unlensed/input_catalog_{seed}_i_mode1.fits')
            # input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_g.fits')
            cdir = 'catalogs_bin2'


        case _:
            print("Unclear input truth! Please fix!")
            sys.exit()

    constant_input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_constant_mode0.fits')
    constant_cdir = 'catalogs_constant0'
    # input_truth.add_index('index')
    ods_phot = OneDegSq[input_truth['index']][ods_cols]
    multi_phot_truth = hstack((input_truth, ods_phot))
    multi_phot_truth.add_index('index')

    ods_phot = OneDegSq[constant_input_truth['index']][ods_cols]
    const_phot_truth = hstack((constant_input_truth, ods_phot))
    const_phot_truth.add_index('index')

    _ = get_unrec_bl_multi_truth(const_phot_truth, zbin=mode)
    _ = get_unrec_bl_multi_truth(multi_phot_truth, zbin=mode)

    full_cat = Table.read(f'{pdir}/anacal_blends/{cdir}/catalog_{seed}.fits')
    Ncat = len(full_cat)
    blend_lbl = np.zeros(Ncat)
    for i, dc in enumerate(full_cat['truth_index']):
        prelensed_lbl = multi_phot_truth.loc[dc]['is_blend']
        if np.any(prelensed_lbl > 0):
            
            blend_lbl[i] = 1
    np.save(f'{pdir}/anacal_blends/unrec/{cdir}_{seed}_unrec_lensed_multi_{pix}pix.npy', blend_lbl)

    print("Working on const table now...")
    const_detection = Table.read(f'{pdir}/anacal_blends/catalogs_constant0/catalog_{seed}.fits')
    Ncat = len(const_detection)
    blend_lbl = np.zeros(Ncat)
    for i, dc in enumerate(const_detection['truth_index']):
        prelensed_lbl = const_phot_truth.loc[dc]['is_blend']
        if np.any(prelensed_lbl > 0):
            blend_lbl[i] = 1
    np.save(f'{pdir}/anacal_blends/unrec/{constant_cdir}_{seed}_unrec_lensed_multi_bin{mode}_{pix}pix.npy', blend_lbl)
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
    # match_seed(ndx, cdir=f'catalogs_constant{cbin}')
    match_seed(ndx, cbin)

print(f"Done with rank {rank}")
