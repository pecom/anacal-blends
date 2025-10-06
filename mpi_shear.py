import anacal
import numpy as np
import matplotlib.pylab as plt

import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
import fitsio
import os

# LSST task to define DC2-like skymap 
from lsst.skymap.discreteSkyMap import (
    DiscreteSkyMapConfig, DiscreteSkyMap,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo


# Simulation Task: simulate exposures, 
# generate input catalogs
from xlens.simulator.multiband import (
    MultibandSimShearTaskConfig,
    MultibandSimShearTask,
)

# Detection Task: Detect, shape measurement
from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipeConfig, 
    AnacalDetectPipe,
)

# Force color measurement Task: 
# flux measurement on the other bands
from xlens.process_pipe.anacal_force import (
    AnacalForcePipe,
    AnacalForcePipeConfig,
)

# Match Task: match to input catalog
from xlens.process_pipe.match import (
    matchPipe,
    matchPipeConfig,
)


from mpi4py import MPI
from astropy.table import Table

comm = MPI.COMM_WORLD
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

N = 10 # How many samples should each process generate?
shear_mode = 0 # What shear mode for this run
# 0: g=-0.02; 1: g=0.02; 2: g=0.00

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

### SkyMap

pixel_scale = 0.2
mag_zero = 30
npix_inner = 4000
npix_pad = 100

center_pix = int((npix_inner + npix_pad * 2) / 2)
config = DiscreteSkyMapConfig()
config.projection = "TAN"

# Define tract center explicitly (tract 24 in DC2-style layout)
config.raList = [0.0]         # degrees
config.decList = [0.0]        # degrees
config.radiusList = [6.4 / 9] # radius in degrees

config.rotation = 0.0         # tract rotation in degrees

# Patch and tract configuration
config.patchInnerDimensions = [4000, 4000]  # inner size of patch in pixels
config.patchBorder = 100                    # border size in pixels
config.pixelScale = pixel_scale             # arcsec/pixel
config.tractOverlap = 0.0                   # no overlap

# Create the skymap
skymap = DiscreteSkyMap(config)

if mpi_rank==0: print("Created skymap")


### Image Simulation

# configuration
config = MultibandSimShearTaskConfig()
config.survey_name = "lsst"
config.draw_image_noise = True

# 0: g=-0.02; 1: g=0.02; 2: g=0.00
config.z_bounds = [-0.01, 20.0]
config.mode = shear_mode

tract_id = 0
patch_id = 24
sim_task = MultibandSimShearTask(config=config)
patch_info = skymap[tract_id][patch_id]
bbox = makeSkyInfo(
    skymap,
    tractId=tract_id,
    patchId=patch_id,
).bbox

wcs = patch_info.getWcs()
if mpi_rank==0: print("WCS Setup")

### Detection + Measurement Configuration

config = AnacalDetectPipeConfig()
config.anacal.force_size = False
config.anacal.num_epochs = 8
config.anacal.do_noise_bias_correction = True
config.anacal.validate_psf = False
# Task and preparation
det_task = AnacalDetectPipe(config=config)

if mpi_rank==0: print("Detection task setup")


sendg1 = None
sendg2 = None
sende1 = None
sende2 = None
sendR1 = None
sendR2 = None
sendN = None
if mpi_rank==0:
    sendg1 = np.zeros([mpi_size, N], dtype=float)
    sendg2 = np.zeros([mpi_size, N], dtype=float)
    sende1 = np.zeros([mpi_size, N], dtype=float)
    sende2 = np.zeros([mpi_size, N], dtype=float)
    sendR1 = np.zeros([mpi_size, N], dtype=float)
    sendR2 = np.zeros([mpi_size, N], dtype=float)
    sendN = np.zeros([mpi_size, N], dtype=float)
g1s = np.empty(N, dtype=float)
g2s = np.empty(N, dtype=float)
e1s = np.empty(N, dtype=float)
e2s = np.empty(N, dtype=float)
R1s = np.empty(N, dtype=float)
R2s = np.empty(N, dtype=float)
Ns = np.empty(N, dtype=float)
comm.Scatter(sendg1, g1s, root=0)
comm.Scatter(sendg2, g2s, root=0)
comm.Scatter(sende1, e1s, root=0)
comm.Scatter(sende2, e2s, root=0)
comm.Scatter(sendR1, R1s, root=0)
comm.Scatter(sendR2, R2s, root=0)
comm.Scatter(sendN, Ns, root=0)

for i in range(N):
    print(f"Process {mpi_rank} on simulation{i}")
    sim_seed = mpi_rank*N + i

    outcome = sim_task.run(band="i", seed=sim_seed, boundaryBox=bbox, wcs=wcs)

    seed = 50000 + sim_seed
    data = det_task.anacal.prepare_data(
        exposure=outcome.outputExposure,
        seed=seed,
        noise_corr=None,
        detection=None,
        band=None,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
    )

    # Run catalog
    catalog = det_task.anacal.run(**data)

    ### Shear Estimation

    # Code magnitude cut and selection bias correction
    e1 = catalog["wsel"] * catalog["fpfs_e1"]
    de1_dg1 = catalog["dwsel_dg1"] * catalog["fpfs_e1"] + catalog["wsel"] * catalog["fpfs_de1_dg1"]

    e2 = catalog["wsel"] * catalog["fpfs_e2"]
    de2_dg2 = catalog["dwsel_dg2"] * catalog["fpfs_e2"] + catalog["wsel"] * catalog["fpfs_de2_dg2"]

    cat_len = len(catalog)

    g1 = np.sum(e1) / np.sum(de1_dg1)
    g2 = np.sum(e2) / np.sum(de2_dg2)

    g1s[i] = g1
    g2s[i] = g2
    e1s[i] = np.sum(e1)
    e2s[i] = np.sum(e2)
    R1s[i] = np.sum(de1_dg1)
    R2s[i] = np.sum(de2_dg2)
    Ns[i] = cat_len


recvg1 = None
recvg2 = None
recve1 = None
recve2 = None
recvR1 = None
recvR2 = None
recvN = None
if mpi_rank==0:
    recvg1 = np.zeros([mpi_size, N], dtype=float)
    recvg2 = np.zeros([mpi_size, N], dtype=float)
    recve1 = np.zeros([mpi_size, N], dtype=float)
    recve2 = np.zeros([mpi_size, N], dtype=float)
    recvR1 = np.zeros([mpi_size, N], dtype=float)
    recvR2 = np.zeros([mpi_size, N], dtype=float)
    recvN = np.zeros([mpi_size, N], dtype=float)

comm.Gather(g1s, recvg1, root=0)
comm.Gather(g2s, recvg2, root=0)
comm.Gather(e1s, recve1, root=0)
comm.Gather(e2s, recve2, root=0)
comm.Gather(R1s, recvR1, root=0)
comm.Gather(R2s, recvR2, root=0)
comm.Gather(Ns, recvN, root=0)

if mpi_rank==0:
    print("all done!")
    print("g1s: ", recvg1)
    print("g1 mean: ", np.mean(recvg1))
    print("--------------------------------------")
    print("g2s: ", recvg2)
    print("g2 mean: ", np.mean(recvg2))
    print("--------------------------------------")

    np.save(pdir + f'/anacal_blends/g1s_mode{shear_mode}', recvg1)
    np.save(pdir + f'/anacal_blends/g2s_mode{shear_mode}', recvg2)
    np.save(pdir + f'/anacal_blends/e1s_mode{shear_mode}', recve1)
    np.save(pdir + f'/anacal_blends/e2s_mode{shear_mode}', recve2)
    np.save(pdir + f'/anacal_blends/R1s_mode{shear_mode}', recvR1)
    np.save(pdir + f'/anacal_blends/R2s_mode{shear_mode}', recvR2)
    np.save(pdir + f'/anacal_blends/Ns_mode{shear_mode}', recvN)

