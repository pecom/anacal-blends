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

def match_seed(seed, cdir='catalogs_bin2', band='i'):
    print(f"Working on object {seed}")
    # Find the input catalog
    match cdir:
        case "catalogs_bin1":
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_g_mode3.fits')
        case "catalogs_bin2":
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_r.fits')
        case "catalogs_constant0":
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_constant_mode0.fits')
        case "catalogs_constant1":
            input_truth = Table.read(f'{pdir}/anacal_blends/truth_inputs/input_catalog_{seed}_constant_mode1.fits')
        case _:
            print("Unclear input truth! Please fix!")
            sys.exit()

    # Add the photometry from OneDegSq
    ods_phot = OneDegSq[input_truth['index']][ods_cols]
    phot_truth = hstack((input_truth, ods_phot))
    # Set-up detection catalog
    if band=='i':
        full_cat = Table.read(f'{pdir}/anacal_blends/{cdir}/catalog_{seed}.fits')
    else:
        full_cat = Table.read(f'{pdir}/anacal_blends/{cdir}/{band}catalog_{seed}.fits')
    full_cat['ndx'] = np.arange(len(full_cat))
    phot_truth.add_index('index')
    full_cat.add_index('ndx')
    full_cat['i_mag'] = 30 - 2.5*np.log10(full_cat['flux'])
    full_cat['scaled_x1'] = full_cat['x1'] / 0.2 - 11900
    full_cat['scaled_x2'] = full_cat['x2'] / 0.2 - 11900

    # Using friendly to label blends:
    phot_fc = FCatalog(phot_truth, 'index', columns=phot_truth.columns)
    full_fc = FCatalog(full_cat, 'ndx', columns=full_cat.columns)

    # Label blends within 5 pixels fo the truth catalog
    candidate_boost_factor = 5

    lazy_kdtree = FKDTree({'search_rad': candidate_boost_factor})
    kdtree_groups_cc_hst, _ = lazy_kdtree(full_fc, phot_fc, {'RA1': 'scaled_x1', 'DEC1': 'scaled_x2', 'RA2': 'image_x', 'DEC2': 'image_y'})
    # Prune groups based on magnitudes:
    mprune = MagDiffPruner({'ground_mag_limit': 28, 'space_mag_limit': 30, 'delta_mag_limit': 3})
    pruned_groups = mprune(full_fc, phot_fc, {'ground_mag_name': 'i_mag', 'space_mag_name': 'i_ab'}, kdtree_groups_cc_hst)

    blend_table = group2table(pruned_groups)
    pd_table = blend_table.to_pandas()
    pd_table.to_parquet(f'{pdir}/anacal_blends/unrec/{cdir}_{seed}_unrec.pq')
    return None


send_ndxs = None
if rank==0:
    const_ndxs = np.arange(40960)
    first_half = np.arange(20480)
    second_half = np.arange(20480, 40960)
    split_ndxs = np.array_split(first_half, size)
else:
    split_ndxs = None

split_ndxs = comm.scatter(split_ndxs, root=0)

print(f"Looking at {len(split_ndxs)} objects at {rank}")

for ndx in split_ndxs:
    # match_seed(ndx, cdir=f'catalogs_bin{cbin}')
    match_seed(ndx, cdir=f'catalogs_constant{cbin}')

print(f"Done with rank {rank}")
