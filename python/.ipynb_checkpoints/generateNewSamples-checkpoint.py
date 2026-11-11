##################################################################
# Setup
##################################################################

import os
import numpy as np
from interaction import Interaction

# Configuration: Specify which interaction posterior to sample from
# Uncomment one of the following lines to select the interaction posterior:
# samplefrom = "usdbuq"
# samplefrom = "usd66"
samplefrom = "usdbuq-500"  # Default selection

# Number of random samples to draw from the posterior
nsamples = 50

# Directories for input data and output samples
datadir = "../data/"
outdir = "../samples/"

# File paths for required data
transformfile = os.path.join(datadir, f"{samplefrom}-transform.npy")
meanfile = os.path.join(datadir, f"{samplefrom}.mean.npy")
covfile = os.path.join(datadir, f"{samplefrom}.cov.npy")

##################################################################
# Logging setup information
##################################################################

print(f"I will generate {nsamples} samples from {samplefrom}")
print("Linear combinations (LCs) will be drawn from a multivariate normal distribution")
print(f"Means read from: {meanfile}")
print(f"Covariance read from: {covfile}")
print(f"Linear transform matrix read from: {transformfile}")

##################################################################
# Determine hyperparameter handling for specific interactions
##################################################################

# The "usd66" interaction includes an unaccounted-for uncertainty parameter (UAUP)
# as a hyperparameter of the fit and a random variable.
if samplefrom == 'usd66':
    nmeta = 1
    print("Assuming 'usd66' has 1 hyperparameter")
else:
    nmeta = 0

# Ensure output directory exists
if not os.path.exists(outdir):
    os.mkdir(outdir)

##################################################################
# Define LC space transformation
##################################################################

# Transformation: y = A * x; x = A.T * y
# where x = matrix elements, y = linear combinations
LCtransform = np.load(transformfile)
dim, _ = LCtransform.shape
print(f"There are {dim} matrix elements")

# Read mean and covariance from fits for generating new samples
mean = np.load(meanfile)
cov = np.load(covfile)

# Exclude hyperparameter from mean and covariance if present
mean = mean[nmeta:]
cov = cov[nmeta:, nmeta:]

##################################################################
# Initialize interaction objects
##################################################################

# Base interaction object
intx = Interaction(os.path.join(datadir, 'w.int'))

# Reference interaction used to set matrix element (ME) order convention
usd = Interaction(os.path.join(datadir, 'w.int'))
intx.orderby(usd)

# Interaction used to set unconstrained MEs
rgsd = Interaction(os.path.join(datadir, 'sdba.int'))
rgsd.orderby(usd)

# Transform the interaction MEs for unconstrained MEs (RGSD)
y0 = LCtransform @ rgsd.get_matrix_elements()

##################################################################
# Generate random samples of linear combinations (LCs)
##################################################################

# Draw random samples from the multivariate normal distribution
setofLCs = np.random.multivariate_normal(mean=mean, cov=cov, size=nsamples)
nsamples, nd = setofLCs.shape
print(f"There are {nd} LCs defined.")
print(f"The remaining {dim - nd} will be set to RGSD LCs")

##################################################################
# Transform LCs to matrix elements and write to output files
##################################################################

# Process each sample
for isample in range(nsamples):
    # Combine sampled LCs with fixed LCs
    LCsample = np.concatenate((setofLCs[isample, :], y0[nd:]))
    
    # Transform LCs to matrix elements
    MEsample = LCtransform.T @ LCsample
    
    # Overwrite the interaction object with the new array of matrix elements
    intx.set_matrix_elements(MEsample)
    
    # Write the interaction object to a file
    output_file = os.path.join(outdir, f"{samplefrom}_{isample}.int")
    with open(output_file, "w+") as f:
        f.write(str(intx))

print(f"Interaction files (.int) written to {outdir}")

##################################################################
# End of script
##################################################################
