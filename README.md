# Shell model interaction library (SMIL)

Data and code repository to accompany article "Towards shell model interactions
with credible uncertainties" by Oliver Gorton and Kostas Kravvaris.

Journal: Phys. Rev. C 112, 014302 – Published 1 July, 2025 

Related DOI: https://doi.org/10.1103/fzxv-4q1r

Preprint: arXiv:2503.11889

This repository contains code and binaries that can be used to generate new
random interactions from the posterior fits of the USDBUQ, USD66, and USDBUQ-500
interactions.

We also include all samples (560 or 670) from the last iteration of the MCMC
fit.

## Folder structure


| Folder | Description |
|---|---|
| `data` | Contains files necessary to generate new random interactions from the posteriors. |
| `python` | Python script to generate new random interactions. Run with:<br>`/path/to/python3 generateNewSamples.py` |
| `usd66` | 670 samples from the USD66 interaction. |
| `usdbuq` | 560 samples from the USDBUQ interaction. |
| `usdbuq-500` | 560 samples from the USDBUQ-500 interaction. |
| `sdshelldata.xml` | XML file listing all experimental energies and their uncertainties used to fit the provided interactions. |

## Getting started

To generate new samples, `cd` to `python` and run generateNewSamples.py with 
your installation of Python. Change the variable `samplefrom` to your desired 
option: "usdbuq", "usd66", or "usdbuq-500".

## Acknowledgments

This work was performed under the auspices of the U.S. Department of Energy by
Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 with
support from LDRD number 23ERD023.

LLNL-CODE-2026615
