import numpy as np
import pandas as pd
from astropy.table import Table, hstack, vstack, join
import scipy.stats as stats
import os, sys
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--cbin", type=int)

args = parser.parse_args()
cbin = int(args.cbin)

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

# Given the truth information, label blends
def get_unrec_bl_multi_truth(truth, maglim = 27, zbin=1):
    bright_filt = truth['i_ab'] < maglim

    # If I'm applying a shear to bin 1,
    # I want to select on the elements in bin 2
    match zbin:
        case 1:
            cat1_filt = truth['z'] > 1
            cat2_filt = truth['z'] < 1
        case 2:
            cat1_filt = truth['z'] < 1
            cat2_filt = truth['z'] > 1

    # We look for objects around the first catalog
    # so apply filter based on the bin we want to isolate
    phot_fc1 = FCatalog(truth[cat1_filt], 'index', columns=truth.columns)
    phot_fc2 = FCatalog(truth[bright_filt & cat2_filt], 'index', columns=truth.columns)

    # Label blends within 10 pixels of the catalogs
    candidate_boost_factor = 10
    lazy_kdtree = FKDTree({'search_rad': candidate_boost_factor})
    
    kdtree_groups_multi, _ = lazy_kdtree(phot_fc1, phot_fc2, {'RA1': 'prelensed_image_x', 'DEC1': 'prelensed_image_y',
                                                              'RA2': 'image_x', 'DEC2': 'image_y'}) # Returns a list of Groups
    # Each Group has an idx1 and ix2
    # idx1 = The IDs of objects in catalog 1 that are within candidate_boost_factor
    #        of the object we care about (which is from catalog 1)
    # idx2 = The IDs of objects in catalog 2 that are within candidate_boost_factor
    #        of the object we care about (which is from catalog 1)

    # g2l == How many objects in catalog2 are within 10 pixels
    #        of each object from catalog 1
    g2l = [len(k.idx2) for k in kdtree_groups_multi]
    g2l = np.array(g2l) 
    truth['is_blend'] = 0

    # Anywhere we have g2l > 0 is a blend
    # so we label objects in both entries as a blend
    for i in np.where(g2l)[0]:
        group = kdtree_groups_multi[i]
        g1s = group.idx1
        g2s = group.idx2

        
        truth.loc[g1s]['is_blend'] = 1
        truth.loc[g2s]['is_blend'] = 1

    return g2l

def match_seed(seed, mode=1):
    print(f"Working on object {seed}")
    # Load in input catalog
    match mode:
        case 1:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_g_mode3.fits')
            cdir = 'catalogs_bin1'
        case 2:
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_g.fits')
            cdir = 'catalogs_bin2'


        case _:
            print("Unclear input truth! Please fix!")
            sys.exit()

    # Load the input for the constant shear
    constant_input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_constant_mode0.fits')
    constant_cdir = 'catalogs_constant0'

    # Add photometry to truth catalog
    ods_phot = OneDegSq[input_truth['index']][ods_cols]
    multi_phot_truth = hstack((input_truth, ods_phot))
    multi_phot_truth.add_index('index')

    ods_phot = OneDegSq[constant_input_truth['index']][ods_cols]
    const_phot_truth = hstack((constant_input_truth, ods_phot))
    const_phot_truth.add_index('index')

    # Label both catalogs as blends 
    _ = get_unrec_bl_multi_truth(const_phot_truth, zbin=mode)
    _ = get_unrec_bl_multi_truth(multi_phot_truth, zbin=mode)

    # Write to files
    full_cat = Table.read(f'{pdir}/anacal_blends/{cdir}/catalog_{seed}.fits')
    blend_lbl = np.zeros(len(full_cat))
    for i, dc in enumerate(full_cat['truth_index']):
        prelensed_lbl = multi_phot_truth.loc[dc]['is_blend']
        if np.any(prelensed_lbl > 0):
            blend_lbl[i] = 1
    np.save(f'{pdir}/anacal_blends/unrec/{cdir}_{seed}_unrec_prelensed_multi.npy', blend_lbl)

    const_detection = Table.read(f'{pdir}/anacal_blends/catalogs_constant0/catalog_{seed}.fits')
    blend_lbl = np.zeros(len(const_detection))
    for i, dc in enumerate(const_detection['truth_index']):
        prelensed_lbl = const_phot_truth.loc[dc]['is_blend']
        if np.any(prelensed_lbl > 0):
            blend_lbl[i] = 1
    np.save(f'{pdir}/anacal_blends/unrec/{constant_cdir}_{seed}_unrec_prelensed_multi_bin{mode}.npy', blend_lbl)
    return None


send_ndxs = None
if rank==0:
    const_ndxs = np.arange(40960)
    first_quarter = np.arange(10240)
    second_quarter = np.arange(10240, 20480)
    third_quarter = np.arange(20480, 30720)
    fourth_quarter = np.arange(30720, 40960)
    split_ndxs = np.array_split(fourth_quarter, size)
else:
    split_ndxs = None

split_ndxs = comm.scatter(split_ndxs, root=0)

print(f"Looking at {len(split_ndxs)} objects at {rank}")

for ndx in split_ndxs:
    # match_seed(ndx, cdir=f'catalogs_bin{cbin}')
    # match_seed(ndx, cdir=f'catalogs_constant{cbin}')
    match_seed(ndx, cbin)

print(f"Done with rank {rank}")
