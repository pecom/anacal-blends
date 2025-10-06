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

from numpy.lib import recfunctions as rfn

from mpi4py import MPI
from astropy.table import Table

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode",)
parser.add_argument("-b", "--band",)

args = parser.parse_args()

band_dict = {'u': 1,
             'g': 2,
             'r': 3,
             'i': 0,
             'z': 4,
             'y': 5}


comm = MPI.COMM_WORLD
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()

hdir = os.getenv("HOME")
pdir = os.getenv("PSCRATCH")

N = 20 # How many samples should each process generate?
shear_mode = int(args.mode)
band = args.band
band_noise = band_dict[band]
print(shear_mode, type(shear_mode))
# 0: g=-0.02; 1: g=0.02; 2: g=0.00
# output_dir = f"{pdir}/anacal_blends/catalogs_constant{shear_mode}_unlensed"
output_dir = f"{pdir}/anacal_blends/catalogs_constant{shear_mode}"


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
# truth_dir = f"{pdir}/anacal_blends/truth_inputs_unlensed"
truth_dir = f"{pdir}/anacal_blends/truth_inputs"

if mpi_rank==0: print("Detection task setup")

config = AnacalForcePipeConfig()
config.anacal.force_size = True
config.anacal.force_center = True
config.anacal.num_epochs = 8
config.anacal.do_noise_bias_correction = True
config.anacal.validate_psf = False
config.anacal.noiseId = band_noise
force_task = AnacalForcePipe(config=config)

for i in range(N):
    print(f"Process {mpi_rank} on simulation{i}")
    sim_seed = mpi_rank*N + i 

    outcome = sim_task.run(band=band, seed=sim_seed, boundaryBox=bbox, wcs=wcs)

    truth_table = Table(outcome.outputTruthCatalog)
    truth_output_name = f"{truth_dir}/input_catalog_{sim_seed}_constant_mode{shear_mode}_band{band}.fits"
    truth_table.write(truth_output_name, format='fits', overwrite=True)

    # i-detection catalog
    i_det = Table.read(f'{output_dir}/catalog_{sim_seed}.fits')
    good_idet = i_det[i_det.colnames[:-3]]

    data = force_task.anacal.prepare_data(
        exposure=outcome.outputExposure,   # from simulation
        seed=sim_seed,           # same as before
        noise_corr=None,
        detection=good_idet, # the detection catalog from i-band
        band=band,
        skyMap=skymap,
        tract=tract_id,
        patch=patch_id,
    )

    flux_colnames = ["flux", "dflux_dg1", "dflux_dg2"]

    cat = rfn.repack_fields(force_task.anacal.run(**data)[flux_colnames])
    map_dict = {name: f"{band}_" + name for name in flux_colnames}
    renamed = rfn.rename_fields(cat, map_dict)

    force_cat = Table(renamed)
    output_name = f"{output_dir}/catalog_{sim_seed}_band{band}.fits"
    force_cat.write(output_name, format='fits', overwrite=True)
