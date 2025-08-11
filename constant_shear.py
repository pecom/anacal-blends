import anacal
import numpy as np
import matplotlib.pylab as plt
import argparse

import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs
import fitsio
import os, sys

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

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode",)

args = parser.parse_args()


comm = MPI.COMM_WORLD
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

N = 40 # How many samples should each process generate?
shear_mode = int(args.mode)
print(shear_mode, type(shear_mode))
# 0: g=-0.02; 1: g=0.02; 2: g=0.00
output_dir = f"{pdir}/anacal_blends/catalogs_constant{shear_mode}/"

exit

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

cat_ref = fitsio.read('/pscratch/sd/p/pecom/anacal_blends/catsim/catsim-v4/OneDegSq.fits')

if mpi_rank==0: print("Detection task setup")

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

    config = matchPipeConfig()
    config.mag_zero = mag_zero
    config.mag_max_truth = 28.0
    match_task = matchPipe(config=config)
    match = match_task.run(
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
        catalog=catalog,
        dm_catalog=None,
        truth_catalog=outcome.outputTruthCatalog,
    ).catalog


    redshift = cat_ref[match["truth_index"]]["redshift"]
    mag_truth = cat_ref[match["truth_index"]]["i_ab"]
    match_cat = Table(match)
    match_cat['redshift'] = redshift
    match_cat['truth_mag'] = mag_truth

    output_name = f"{output_dir}/catalog_{sim_seed}.fits"
    match_cat.write(output_name, format='fits', overwrite=True)
