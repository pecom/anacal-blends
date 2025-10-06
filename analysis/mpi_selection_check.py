import numpy as np
import pandas as pd
from astropy.table import Table, hstack
import scipy.stats as stats
import os, sys
from mpi4py import MPI
import argparse
from pickle import load
from numpy.lib.recfunctions import structured_to_unstructured
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("-z", "--zbin", type=int)
parser.add_argument("-q", "--quarter", type=int)

args = parser.parse_args()
zbin = int(args.zbin)
qbin = int(args.quarter)

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

rng = np.random.default_rng()

with open(f"{hdir}/data/anacal/unrecognized_blend_lensed.pkl", "rb") as f:
    clf = load(f)


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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

def get_data(seed, zbin):
    if zbin==0:
        mz_data = Table.read(f'{pdir}/anacal_blends/catalogs_constant0/catalog_{seed}.fits')
        g_data = Table.read(f'{pdir}/anacal_blends/catalogs_constant0/catalog_{seed}_bandg.fits')
        r_data = Table.read(f'{pdir}/anacal_blends/catalogs_constant0/catalog_{seed}_bandr.fits')

        full_data = hstack((mz_data, g_data, r_data))
    else:
        mz_data = Table.read(f'{pdir}/anacal_blends/catalogs_bin{zbin}/catalog_{seed}.fits')
        g_data = Table.read(f'{pdir}/anacal_blends/catalogs_bin{zbin}/gcatalog_{seed}.fits')
        r_data = Table.read(f'{pdir}/anacal_blends/catalogs_bin{zbin}/rcatalog_{seed}.fits')

        full_data = hstack((mz_data, g_data, r_data))

    return full_data



def imag_cut(seed, zbin=1, dg1=0, dg2=0):
    full_data = get_data(seed, zbin)

    dmi_dg1 = -2.5*full_data['dflux_dg1']/(full_data['flux']*np.log(10))
    dmi_dg2 = -2.5*full_data['dflux_dg2']/(full_data['flux']*np.log(10))


    i_mag = 30 - 2.5*np.log10(full_data['flux'])
    shear_imag = i_mag + dg1*dmi_dg1 + dg2*dmi_dg2

    return shear_imag

def load_matched_catalogs_zs(f_lambda, Ns, ddir='/pscratch/sd/p/pecom/anacal_blends/',
                             bin_dir_name='catalogs_bin', bin_num=1,
                             const_dir_name='catalogs_constant0'):

    N = len(Ns)
    mode0_e1s = np.zeros(N)
    mode1_e1s = np.zeros(N)
    mode0_R1s = np.zeros(N)
    mode1_R1s = np.zeros(N)

    for j,i in enumerate(Ns):
        print(f"On {j} {i} out of {N} for process {rank}")
        file_name = f_lambda(i)

        zbin_catalog = Table.read(ddir + bin_dir_name + str(bin_num) + "/" + file_name)
        const_catalog = Table.read(ddir + const_dir_name + "/" + file_name) 

        # unrec_filt is True if the object is good (not a blend)
        imags = imag_cut(i, bin_num)
        unrec_filt_bin = imags < 24
        Nbin = len(imags)

        const_imags = imag_cut(i, 0)
        unrec_filt_const = const_imags < 24
        Nconst = len(const_imags)


        print(f"Keeping {np.sum(unrec_filt_bin)} out of {len(unrec_filt_bin)} from {i}")
        print(f"Keeping {np.sum(unrec_filt_const)} out of {len(unrec_filt_const)} from {i}")


        mode0_values = eR_fromcatalog_zbin_unrec(const_catalog, unrec_filt_const, z_bin=bin_num)
        mode1_values = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin, z_bin=bin_num)


        ### Selection bias correction for binned sim:

        delta_g = 0.02
        
        imag_plus = imag_cut(i, bin_num, dg1=delta_g, dg2=0)
        unrec_filt_bin_plus = imag_plus < 24
        e1p, _, _, _ = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin_plus, z_bin=bin_num)

        imag_minus = imag_cut(i, bin_num, dg1=-1.*delta_g, dg2=0)
        unrec_filt_bin_minus = imag_minus < 24
        e1m, _, _, _ = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin_minus, z_bin=bin_num)

        R1_selection = (e1p - e1m)/(2.0 * delta_g)

        const_plus = imag_cut(i, 0, dg1=delta_g, dg2=0)
        unrec_filt_const_plus = const_plus < 24
        e1p, _, _, _ = eR_fromcatalog_zbin_unrec(const_catalog, unrec_filt_const_plus, z_bin=bin_num)

        const_minus = imag_cut(i, 0, dg1=-1.*delta_g, dg2=0)
        unrec_filt_const_minus = const_minus < 24
        e1m, _, _, _ = eR_fromcatalog_zbin_unrec(const_catalog, unrec_filt_const_minus, z_bin=bin_num)

        R1_selection_const = (e1p - e1m)/(2.0 * delta_g)
        # redshift_score_plus = score_rf(i, bin_num, dg1=0, dg2=delta_g)
        # unrec_filt_bin_plus = redshift_score_plus < rf_threshold
        # _, e2p, _, _ = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin_plus, z_bin=bin_num)

        # redshift_score_minus = score_rf(i, bin_num, dg1=0, dg2=-1.*delta_g)
        # unrec_filt_bin_minus = redshift_score_minus < rf_threshold
        # _, e2m, _, _ = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin_minus, z_bin=bin_num)

        # R2_selection = (e2p - e2m)/(2.0 * delta_g)

        R1_selection = 0
        R1_selection_const = 0


        mode0_e1s[j] = mode0_values[0]
        mode1_e1s[j] = mode1_values[0]

        mode0_R1s[j] = mode0_values[2] + R1_selection_const
        mode1_R1s[j] = mode1_values[2] + R1_selection
        print('-------------------------------')
        print(R1_selection_const, R1_selection)
        print('-------------------------------')
        # mode0_R1s[j] = mode0_values[2]
        # mode1_R1s[j] = mode1_values[2]
    return mode0_e1s, mode1_e1s, mode0_R1s, mode1_R1s


send_ndxs = None
if rank==0:
    match qbin:
        case 0:
            ndxs = np.arange(100)
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

catv1 = lambda x : f'catalog_{x}.fits'
matched_const = load_matched_catalogs_zs(catv1, split_ndxs, bin_num=zbin)
# print(f"On {rank} we have {matched_const[0]}")

matched_const = comm.gather(matched_const, root=0)


if rank==0:
    # matched_const_np = np.array(matched_const)
    matched_proper = np.zeros((4, len(ndxs)))
    for i in range(4):
        matched_proper[i] = np.concatenate([mcp[i] for mcp in matched_const])

    np.save(f'{hdir}/anacal_scripts/data/matched_imag_cut.npy', matched_proper)
    e1_0, e1_1, R1_0, R1_1 = matched_proper
    est_m = (np.sum(e1_1 - e1_0) / np.sum(R1_1 + R1_0)) / .02 - 1
    print(matched_proper)
    print('--------------------------------------------------------')
    print(est_m)
