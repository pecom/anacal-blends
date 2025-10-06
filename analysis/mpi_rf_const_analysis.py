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
parser.add_argument("-q", "--quarter", type=int)
parser.add_argument("-s", "--score", type=float)
parser.add_argument("-r", "--random", action='store_true')

args = parser.parse_args()
qbin = int(args.quarter)
score = float(args.score)
rand = args.random

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

def eR_fromcatalog_unrec(catalog, unrec_filt, clean=False):
    if clean:
        clean_filt = np.logical_and(np.abs(catalog["fpfs_e1"]) < 0.3, np.abs(catalog["fpfs_e2"]) < 0.3)
        catalog = catalog[clean_filt]

    catalog = catalog[unrec_filt]
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

def get_data(seed, mode):
    mz_data = Table.read(f'{pdir}/anacal_blends/catalogs_constant{mode}/catalog_{seed}.fits')
    g_data = Table.read(f'{pdir}/anacal_blends/catalogs_constant{mode}/catalog_{seed}_bandg.fits')
    r_data = Table.read(f'{pdir}/anacal_blends/catalogs_constant{mode}/catalog_{seed}_bandr.fits')

    full_data = hstack((mz_data, g_data, r_data))

    return full_data


def score_rf(seed, mode=1, dg1=0, dg2=0):
    full_data = get_data(seed, mode)

    dmi_dg1 = -2.5*full_data['dflux_dg1']/(full_data['flux']*np.log(10))
    dmi_dg2 = -2.5*full_data['dflux_dg2']/(full_data['flux']*np.log(10))

    dmg_dg1 = -2.5*full_data['g_dflux_dg1']/(full_data['g_flux']*np.log(10))
    dmg_dg2 = -2.5*full_data['g_dflux_dg2']/(full_data['g_flux']*np.log(10))

    dmr_dg1 = -2.5*full_data['r_dflux_dg1']/(full_data['r_flux']*np.log(10))
    dmr_dg2 = -2.5*full_data['r_dflux_dg2']/(full_data['r_flux']*np.log(10))

    dsize_dg1 = 2*(full_data['fpfs_e1']*full_data['fpfs_de1_dg1'] + full_data['fpfs_e2']*full_data['fpfs_de2_dg1'])
    dsize_dg2 = 2*(full_data['fpfs_e1']*full_data['fpfs_de1_dg2'] + full_data['fpfs_e2']*full_data['fpfs_de2_dg2'])

    i_mag = 30 - 2.5*np.log10(full_data['flux'])
    g_mag = 30 - 2.5*np.log10(full_data['g_flux'])
    r_mag = 30 - 2.5*np.log10(full_data['r_flux'])
    fpfs_size = full_data['fpfs_e1']**2 + full_data['fpfs_e2']**2

    full_data['i_mag'] = i_mag + dg1*dmi_dg1 + dg2*dmi_dg2
    full_data['g_mag'] = g_mag + dg1*dmg_dg1 + dg2*dmg_dg2
    full_data['r_mag'] = r_mag + dg1*dmr_dg1 + dg2*dmr_dg2
    full_data['fpfs_size'] = fpfs_size + dg1*dsize_dg1 + dg2*dsize_dg2 # Should be ellipticity
    full_data['m2_mod'] = full_data['fpfs_m2'] + dg1*full_data['fpfs_dm2_dg1'] + dg2*full_data['fpfs_dm2_dg2'] # Should be size

    # Not changing names in case it breaks things with the pre-trained RF....

    rf_photom = full_data[['g_mag', 'r_mag', 'i_mag', 'fpfs_size', 'm2_mod']].as_array()
    rf_photom = structured_to_unstructured(rf_photom)
    probs = clf.predict_proba(rf_photom)
    scores = probs[:,1]

    return scores

def load_matched_catalogs_zs(f_lambda, Ns, ddir='/pscratch/sd/p/pecom/anacal_blends/',
                             const_dir_name='catalogs_constant0', rf_threshold=0.5):

    N = len(Ns)
    mode0_e1s = np.zeros(N)
    mode1_e1s = np.zeros(N)
    mode0_R1s = np.zeros(N)
    mode1_R1s = np.zeros(N)

    for j,i in enumerate(Ns):
        print(f"On {j} {i} out of {N} for process {rank}")
        file_name = f_lambda(i)

        const_catalog0 = Table.read(ddir + 'catalogs_constant0' + "/" + file_name) 
        const_catalog1 = Table.read(ddir + 'catalogs_constant1' + "/" + file_name) 

        # unrec_filt is True if the object is good (not a blend)
        score1 = score_rf(i, mode=1)
        unrec_filt_bin = score1 < rf_threshold
        Nbin = len(score1)

        score0 = score_rf(i, mode=0)
        unrec_filt_const = score0 < rf_threshold
        Nconst = len(score0)

        if rand:
            rand_cost_bin = np.sum(~unrec_filt_bin)/Nbin
            rand_cost_const = np.sum(~unrec_filt_const)/Nconst
            print(f"Const 0 cost {rand_cost_bin} | Const 1 {rand_cost_const}")

            unrec_filt_bin = rng.choice([0,1], size=Nbin, p=[rand_cost_bin, 1-rand_cost_bin])
            unrec_filt_const = rng.choice([0,1], size=Nconst, p=[rand_cost_const, 1-rand_cost_const])


        print(f"Keeping {np.sum(unrec_filt_bin)} out of {len(unrec_filt_bin)} from {i} with random {rand}")
        print(f"Keeping {np.sum(unrec_filt_const)} out of {len(unrec_filt_const)} from {i} with random {rand}")


        mode0_values = eR_fromcatalog_unrec(const_catalog0, unrec_filt_const)
        mode1_values = eR_fromcatalog_unrec(const_catalog1, unrec_filt_bin)


        ### Selection bias correction for binned sim:
        if rand:
            R1_selection = 0
            R1_selection_const = 0
        else:
            delta_g = 0.02
            
            redshift_score_plus = score_rf(i, 1, dg1=delta_g, dg2=0)
            unrec_filt_bin_plus = redshift_score_plus < rf_threshold
            e1p, _, _, _ = eR_fromcatalog_unrec(const_catalog1, unrec_filt_bin_plus)

            redshift_score_minus = score_rf(i, 1, dg1=-1.*delta_g, dg2=0)
            unrec_filt_bin_minus = redshift_score_minus < rf_threshold
            e1m, _, _, _ = eR_fromcatalog_unrec(const_catalog1, unrec_filt_bin_minus)

            R1_selection = (e1p - e1m)/(2.0 * delta_g)

            const_score_plus = score_rf(i, 0, dg1=delta_g, dg2=0)
            unrec_filt_const_plus = const_score_plus < rf_threshold
            e1p, _, _, _ = eR_fromcatalog_unrec(const_catalog0, unrec_filt_const_plus)

            const_score_minus = score_rf(i, 0, dg1=-1.*delta_g, dg2=0)
            unrec_filt_const_minus = const_score_minus < rf_threshold
            e1m, _, _, _ = eR_fromcatalog_unrec(const_catalog0, unrec_filt_const_minus)

            R1_selection_const = (e1p - e1m)/(2.0 * delta_g)
        # redshift_score_plus = score_rf(i, bin_num, dg1=0, dg2=delta_g)
        # unrec_filt_bin_plus = redshift_score_plus < rf_threshold
        # _, e2p, _, _ = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin_plus, z_bin=bin_num)

        # redshift_score_minus = score_rf(i, bin_num, dg1=0, dg2=-1.*delta_g)
        # unrec_filt_bin_minus = redshift_score_minus < rf_threshold
        # _, e2m, _, _ = eR_fromcatalog_zbin_unrec(zbin_catalog, unrec_filt_bin_minus, z_bin=bin_num)

        # R2_selection = (e2p - e2m)/(2.0 * delta_g)


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

catv1 = lambda x : f'catalog_{x}.fits'
matched_const = load_matched_catalogs_zs(catv1, split_ndxs, rf_threshold=score)
# print(f"On {rank} we have {matched_const[0]}")

matched_const = comm.gather(matched_const, root=0)


if rank==0:
    # matched_const_np = np.array(matched_const)
    matched_proper = np.zeros((4, len(ndxs)))
    for i in range(4):
        matched_proper[i] = np.concatenate([mcp[i] for mcp in matched_const])
    if qbin==0:
        suffix='_test'
    else:
        suffix =''
    if rand:
        np.save(f'{hdir}/anacal_scripts/data/matched_const_lensed_RFscore{score}_rand{suffix}.npy', matched_proper)
    else:
        np.save(f'{hdir}/anacal_scripts/data/matched_const_lensed_RFscore{score}{suffix}.npy', matched_proper)
    e1_0, e1_1, R1_0, R1_1 = matched_proper
    est_m = (np.sum(e1_1 - e1_0) / np.sum(R1_1 + R1_0)) / .02 - 1
    print(matched_proper)
    print('--------------------------------------------------------')
    print(est_m)
