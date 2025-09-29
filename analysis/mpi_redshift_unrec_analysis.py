import numpy as np
import pandas as pd
from astropy.table import Table
import scipy.stats as stats
import os, sys
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zbin",)

args = parser.parse_args()
zbin = int(args.zbin)

rng = np.random.default_rng()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

def eR_fromcatalog(catalog, clean=False):
    if clean:
        clean_filt = np.logical_and(np.abs(catalog["fpfs_e1"]) < 0.3, np.abs(catalog["fpfs_e2"]) < 0.3)
        catalog = catalog[clean_filt]
    e1 = np.sum(catalog["wsel"] * catalog["fpfs_e1"])
    de1_dg1 = np.sum(catalog["dwsel_dg1"] * catalog["fpfs_e1"] + catalog["wsel"] * catalog["fpfs_de1_dg1"])

    e2 = np.sum(catalog["wsel"] * catalog["fpfs_e2"])
    de2_dg2 = np.sum(catalog["dwsel_dg2"] * catalog["fpfs_e2"] + catalog["wsel"] * catalog["fpfs_de2_dg2"])

    return e1, e2, de1_dg1, de2_dg2

def eR_fromcatalog_zbin(catalog, z_bin=0, clean=False):
    # z_bin = 1 --> Only getting shears from objects in [0,1]
    # z_bin = 2 --> Only getting shears from objects in [1,20]
    if clean:
        clean_filt = np.logical_and(np.abs(catalog["fpfs_e1"]) < 0.3, np.abs(catalog["fpfs_e2"]) < 0.3)
        catalog = catalog[clean_filt]
    if z_bin==1:
        z_filt = np.logical_and(catalog['redshift'] > 0, catalog['redshift'] < 1)
    elif z_bin==2:
        z_filt = np.logical_and(catalog['redshift'] > 1, catalog['redshift'] < 20)
    filt_cat = catalog[z_filt]

    e1 = np.sum(filt_cat["wsel"] * filt_cat["fpfs_e1"])
    de1_dg1 = np.sum(filt_cat["dwsel_dg1"] * filt_cat["fpfs_e1"] + filt_cat["wsel"] * filt_cat["fpfs_de1_dg1"])

    e2 = np.sum(filt_cat["wsel"] * filt_cat["fpfs_e2"])
    de2_dg2 = np.sum(filt_cat["dwsel_dg2"] * filt_cat["fpfs_e2"] + filt_cat["wsel"] * filt_cat["fpfs_de2_dg2"])

    return e1, e2, de1_dg1, de2_dg2

def eR_fromcatalog_zbin_unrec(catalog, unrec_filt, z_bin=0):
    # z_bin = 1 --> Only getting shears from objects in [0,1]
    # z_bin = 2 --> Only getting shears from objects in [1,20]

    if z_bin==1:
        z_filt = np.logical_and(catalog['redshift'] > 0, catalog['redshift'] < 1)
    elif z_bin==2:
        z_filt = np.logical_and(catalog['redshift'] > 1, catalog['redshift'] < 20)

    total_filt = np.logical_and(z_filt, unrec_filt)
    filt_cat = catalog[total_filt]

    e1 = np.sum(filt_cat["wsel"] * filt_cat["fpfs_e1"])
    de1_dg1 = np.sum(filt_cat["dwsel_dg1"] * filt_cat["fpfs_e1"] + filt_cat["wsel"] * filt_cat["fpfs_de1_dg1"])

    e2 = np.sum(filt_cat["wsel"] * filt_cat["fpfs_e2"])
    de2_dg2 = np.sum(filt_cat["dwsel_dg2"] * filt_cat["fpfs_e2"] + filt_cat["wsel"] * filt_cat["fpfs_de2_dg2"])

    return e1, e2, de1_dg1, de2_dg2

def load_matched_catalogs_zs(f_lambda, Ns, ddir='/pscratch/sd/p/pecom/anacal_blends/',
                             bin_dir_name='catalogs_bin', bin_num=1, const_dir_name='catalogs_constant0'):

    N = len(Ns)
    mode0_e1s = np.zeros(N)
    mode1_e1s = np.zeros(N)
    mode0_R1s = np.zeros(N)
    mode1_R1s = np.zeros(N)

    for j,i in enumerate(Ns):
        print(f"On {j} {i} out of {N} for process {rank}")
        file_name = f_lambda(i)

        # unrec_blends_bin = Table.read(ddir + f'unrec/{bin_dir_name}{bin_num}_{i}_unrec.pq')
        # unrec_filt_bin = np.array(unrec_blends_bin['blend_diff'] < 1)
        # unrec_blends_const = Table.read(ddir + f'unrec/{const_dir_name}_{i}_unrec.pq')
        # unrec_filt_const = np.array(unrec_blends_const['blend_diff'] < 1)

        # Nt = len(unrec_blends_bin)
        # random_throw = rng.integers(0, Nt, 10)
        # unrec_filt_bin = np.ones(Nt)
        # unrec_filt_bin[random_throw] = 0

        multiz_shear_bin = np.load(f'{pdir}/anacal_blends/unrec/{bin_dir_name}{bin_num}_{i}_unrec_lensed_multi.npy')
        multi_unrec = pd.read_parquet(f'{pdir}/anacal_blends/unrec/{bin_dir_name}{bin_num}_{i}_unrec.pq') 
        # unrec_filt is True if the object is good (not a blend)
        unrec_filt_bin = ~np.logical_and((multiz_shear_bin >= 1), (multi_unrec['blend_diff'] >= 1).to_numpy())
        print(f"Keeping {np.sum(unrec_filt_bin)} out of {len(unrec_filt_bin)} from {i}")
        const_shear_bin = np.load(f'{pdir}/anacal_blends/unrec/{const_dir_name}_{i}_unrec_lensed_multi_bin{bin_num}.npy')
        const_unrec = pd.read_parquet(f'{pdir}/anacal_blends/unrec/{const_dir_name}_{i}_unrec.pq')
        unrec_filt_const = ~np.logical_and((const_shear_bin >= 1), (const_unrec['blend_diff'] >=1).to_numpy())
        print(f"Keeping {np.sum(unrec_filt_const)} out of {len(unrec_filt_const)} from {i}")

        zbin_catalog = Table.read(ddir + bin_dir_name + str(bin_num) + "/" + file_name)
        const_catalog = Table.read(ddir + const_dir_name + "/" + file_name) 

        # mode0_values = eR_fromcatalog(const_catalog, clean=False)
        # mode1_values = eR_fromcatalog(zbin_catalog, clean=False)
#        mode0_values = eR_fromcatalog_zbin(const_catalog, z_bin=bin_num)
#        mode1_values = eR_fromcatalog_zbin(zbin_catalog, z_bin=bin_num)

        mode0_values = eR_fromcatalog_zbin_unrec(const_catalog, unrec_filt_const, z_bin=bin_num)
        mode1_values = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin, z_bin=bin_num)

        mode0_e1s[j] = mode0_values[0]
        mode1_e1s[j] = mode1_values[0]

        mode0_R1s[j] = mode0_values[2]
        mode1_R1s[j] = mode1_values[2]
    return mode0_e1s, mode1_e1s, mode0_R1s, mode1_R1s


send_ndxs = None
if rank==0:
    # full_catalogs = os.listdir('/pscratch/sd/p/pecom/anacal_blends/catalogs_constant0')
    # const_ndxs0 = [int(k[8:-5]) for k in full_catalogs if not 'old' in k]
    # full_catalogs = os.listdir('/pscratch/sd/p/pecom/anacal_blends/catalogs_bin1')
    # const_ndxs1 = [int(k[8:-5]) for k in full_catalogs if not 'old' in k]
    # const_ndxs = np.intersect1d(const_ndxs0, const_ndxs1)
    const_ndxs = np.arange(40960)
    const_ndxs = np.arange(10240)
    const_ndxs = np.arange(10)
    print(f"Looking at {len(const_ndxs)} matched simulations")
    split_ndxs = np.array_split(const_ndxs, size)
else:
    split_ndxs = None

split_ndxs = comm.scatter(split_ndxs, root=0)

print(f"Looking at {len(split_ndxs)} objects at {rank}")

catv1 = lambda x : f'catalog_{x}.fits'
matched_const = load_matched_catalogs_zs(catv1, split_ndxs, bin_num=zbin)
# print(f"On {rank} we have {matched_const[0]}")

matched_const = comm.gather(matched_const, root=0)


if rank==0:
    # matched_const_np = np.array(matched_const)
    matched_proper = np.zeros((4, len(const_ndxs)))
    for i in range(4):
        matched_proper[i] = np.concatenate([mcp[i] for mcp in matched_const])
    np.save(f'{hdir}/anacal_scripts/data/matched_redshift_bin{zbin}_lensed_no-unrec.npy', matched_proper)
    print(matched_proper)
