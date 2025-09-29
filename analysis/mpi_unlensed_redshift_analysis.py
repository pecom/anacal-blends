import numpy as np
import pandas as pd
from astropy.table import Table
import scipy.stats as stats
import os, sys
from mpi4py import MPI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zbin")
parser.add_argument("-u", "--unrec", action='store_true')
parser.add_argument("-t", "--test", action='store_true')
parser.add_argument("-p", "--pixel", type=int)

args = parser.parse_args()
zbin = int(args.zbin)
test = args.test
unrec= args.unrec
pix = int(args.pixel)

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
    # print(f'Looking at {len(filt_cat)} objects')

    e1 = np.sum(filt_cat["wsel"] * filt_cat["fpfs_e1"])
    de1_dg1 = np.sum(filt_cat["dwsel_dg1"] * filt_cat["fpfs_e1"] + filt_cat["wsel"] * filt_cat["fpfs_de1_dg1"])

    e2 = np.sum(filt_cat["wsel"] * filt_cat["fpfs_e2"])
    de2_dg2 = np.sum(filt_cat["dwsel_dg2"] * filt_cat["fpfs_e2"] + filt_cat["wsel"] * filt_cat["fpfs_de2_dg2"])

    return e1, e2, de1_dg1, de2_dg2

def load_matched_catalogs_zs(f_lambda, Ns, ddir='/pscratch/sd/p/pecom/anacal_blends/', load_unrec=True,
                             bin_dir_name='catalogs_bin', bin_num=1, const_dir_name='catalogs_constant0', pix=10):

    N = len(Ns)
    mode0_e1s = np.zeros(N)
    mode1_e1s = np.zeros(N)
    mode0_R1s = np.zeros(N)
    mode1_R1s = np.zeros(N)

    for j,i in enumerate(Ns):
        print(f"On {j} {i} out of {N} for process {rank}")
        file_name = f_lambda(i)

        ##########################################################
        # unrec_filt is True if the object is good (not a blend) #
        ##########################################################

        zbin_catalog = Table.read(ddir + bin_dir_name + str(bin_num) +"_unlensed/" + file_name)
        const_catalog = Table.read(ddir + const_dir_name + "_unlensed/" + file_name) 

        if load_unrec:
            multiz_shear_bin = np.load(f'{pdir}/anacal_blends/unrec/{bin_dir_name}{bin_num}_unlensed_{i}_unrec_unlensed_5pix.npy')
            # multiz_shear_bin = np.load(f'{pdir}/anacal_blends/unrec/{bin_dir_name}{bin_num}_unlensed_{i}_unrec_prelensed_multi_unlensed_{pix}pix.npy')
            unrec_filt_bin = (multiz_shear_bin < 1)

            const_shear_bin = np.load(f'{pdir}/anacal_blends/unrec/{const_dir_name}_unlensed_{i}_unrec_unlensed_5pix.npy')
            # const_shear_bin = np.load(f'{pdir}/anacal_blends/unrec/{const_dir_name}_{i}_unrec_prelensed_multi_unlensed_bin{bin_num}_{pix}pix.npy')
            unrec_filt_const = (const_shear_bin < 1)
        else:
            unrec_filt_bin = np.ones(len(zbin_catalog)).astype(bool)
            unrec_filt_const = np.ones(len(const_catalog)).astype(bool)

        print(f"We have {np.sum(unrec_filt_bin)} good objects.")

        mode0_values = eR_fromcatalog_zbin_unrec(const_catalog, unrec_filt_const, z_bin=bin_num)
        mode1_values = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin, z_bin=bin_num)

        mode0_e1s[j] = mode0_values[0]
        mode1_e1s[j] = mode1_values[0]

        mode0_R1s[j] = mode0_values[2]
        mode1_R1s[j] = mode1_values[2]
    return mode0_e1s, mode1_e1s, mode0_R1s, mode1_R1s


send_ndxs = None
if rank==0:

    const_ndxs = np.arange(40960)
    const_ndxs = np.arange(10240)
    if test:
        const_ndxs = np.arange(3)
    print(f"Looking at {len(const_ndxs)} matched simulations")
    split_ndxs = np.array_split(const_ndxs, size)
else:
    split_ndxs = None

split_ndxs = comm.scatter(split_ndxs, root=0)

print(f"Looking at {len(split_ndxs)} objects at {rank}")

catv1 = lambda x : f'catalog_{x}.fits'
matched_const = load_matched_catalogs_zs(catv1, split_ndxs, load_unrec=unrec, bin_num=zbin, pix=pix)
# print(f"On {rank} we have {matched_const[0]}")

matched_const = comm.gather(matched_const, root=0)


if rank==0:
    # matched_const_np = np.array(matched_const)
    matched_proper = np.zeros((4, len(const_ndxs)))
    for i in range(4):
        matched_proper[i] = np.concatenate([mcp[i] for mcp in matched_const])
    if test:
        if unrec:
            np.save(f'{hdir}/anacal_scripts/data/matched_redshift_bin{zbin}_unlensed_unrec_{pix}pix_test.npy', matched_proper)
        else:
            np.save(f'{hdir}/anacal_scripts/data/matched_redshift_bin{zbin}_unlensed_test.npy', matched_proper)
    else:
        if unrec:
            np.save(f'{hdir}/anacal_scripts/data/matched_redshift_bin{zbin}_unlensed_unrec_{pix}pix.npy', matched_proper)
        else:
            np.save(f'{hdir}/anacal_scripts/data/matched_redshift_bin{zbin}_unlensed.npy', matched_proper)
    print(matched_proper)
