import numpy as np
from astropy.table import Table
import scipy.stats as stats
import os, sys
from mpi4py import MPI


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

def load_matched_catalogs(f_lambda, Ns, ddir='/pscratch/sd/p/pecom/anacal_blends/'):
    # f_lambda should be a function that takes an input (int) N
    # and returns the file_name string

    N = len(Ns)
    mode0_e1s = np.zeros(N)
    mode1_e1s = np.zeros(N)
    mode0_R1s = np.zeros(N)
    mode1_R1s = np.zeros(N)

    for j,i in enumerate(Ns):
        if j%100==0:
            print(f"On {j} {i} out of {N} for process {rank}")
        file_name = f_lambda(i)

        mode0_catalog = Table.read(ddir + "catalogs_constant0/" + file_name)
        mode1_catalog = Table.read(ddir + "catalogs_constant1/" + file_name)

        mode0_values = eR_fromcatalog(mode0_catalog, clean=False)
        mode1_values = eR_fromcatalog(mode1_catalog, clean=False)

        mode0_e1s[j] = mode0_values[0]
        mode1_e1s[j] = mode1_values[0]

        mode0_R1s[j] = mode0_values[2]
        mode1_R1s[j] = mode1_values[2]
    return mode0_e1s, mode1_e1s, mode0_R1s, mode1_R1s

def m_bootstrap(em, ep, Rm, Rp, Nsample=10000):
    N = len(ep)
    ms = np.zeros(Nsample)
    for i in range(Nsample):
        k = rng.choice(N, N, replace=True)
        new_gamma = np.sum(ep[k] - em[k]) / np.sum(Rp[k] + Rm[k])
        m = new_gamma / .02 - 1
        ms[i] = m
    return ms

def get_ms(e1_0, e1_1, R1_0, R1_1):
    m = (np.sum(e1_1 - e1_0) / np.sum(R1_1 + R1_0)) / .02 - 1
    ms = m_bootstrap(e1_0, e1_1, R1_0, R1_1)
    ord_ms = np.sort(ms)
    sigma_m = (ord_ms[9750] - ord_ms[250]) / 4
    mean_m = np.mean(m)
    print(f"Mean={mean_m:0.5f}, bootstrap mean:{np.mean(ms):0.5f}, σ_m={sigma_m:0.5f}")
    return ms


send_ndxs = None
if rank==0:
    full_catalogs = os.listdir('/pscratch/sd/p/pecom/anacal_blends/catalogs_constant0')
    const_ndxs0 = [int(k[8:-5]) for k in full_catalogs if not 'old' in k]
    full_catalogs = os.listdir('/pscratch/sd/p/pecom/anacal_blends/catalogs_constant1')
    const_ndxs1 = [int(k[8:-5]) for k in full_catalogs if not 'old' in k]
    const_ndxs = np.intersect1d(const_ndxs0, const_ndxs1)
    split_ndxs = np.array_split(const_ndxs, size)
else:
    split_ndxs = None

split_ndxs = comm.scatter(split_ndxs, root=0)

print(f"Looking at {len(split_ndxs)} objects at {rank}")

catv1 = lambda x : f'catalog_{x}.fits'
matched_const = load_matched_catalogs(catv1, split_ndxs)
# print(f"On {rank} we have {matched_const[0]}")

matched_const = comm.gather(matched_const, root=0)


if rank==0:
    # matched_const_np = np.array(matched_const)
    matched_proper = np.zeros((4, len(const_ndxs)))
    for i in range(4):
        matched_proper[i] = np.concatenate([mcp[i] for mcp in matched_const])
    np.save(f'{hdir}/anacal_scripts/data/matched_const.npy', matched_proper)
    print(matched_proper)
