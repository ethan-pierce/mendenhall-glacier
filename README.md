[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13256637.svg)](https://doi.org/10.5281/zenodo.13256637)

## Modeling sediment fluxes from debris-rich basal ice layers
##### Pierce, Overeem, and Jouvet
###### *(recently submitted to JGR: Earth Surface)*

This repository contains data, models, and figures related to a submitted manuscript. The software is open-source (see License), 
and freely available for review, use as-is, or extension in other open-source applications. Issues or feature requests may be
raised here, but will not be addressed until after the authors have received reviews, so that reviewers have ready access to the 
relevant code base for the submitted manuscript.

### **Directory structure**

In this repository you will find:
- ```basis/``` core model code, including source code ```basis/src/``` and tests ```basis/test/```
- ```experiments/``` handler scripts to run ```experiments/sensitivity/``` sensitivity tests and ```experiments/static-effective-pressure``` pressure scenarios
- ```figures/``` for lots of different visualizations of model output
- ```inputs/``` for input files, including ```inputs/field-data``` field data and ```inputs/igm-results``` IGM model results
- ```manuscript/``` for draft manuscript files
- ```notebooks/``` for a few prototypes and other visualizations

### **Installation instructions**
We recommend using *Poetry*, *conda*, *mamba*, or a similar environment and package management tool to install this software.

To install poetry, follow the instructions at https://python-poetry.org/docs/#installing-with-the-official-installer.

To install conda or mamba, follow the instructions at https://github.com/conda-forge/miniforge.

Then:
1. Clone the repository with ```git clone https://github.com/ethan-pierce/mendenhall-glacier.git```
2. Navigate to the repo by ```cd mendenhall-glacier```
3. In the top-level directory, run ```poetry init```
4. Preface any Python commands with ```poetry run ...``` to run in the new environment

So, for example, you could run the sensitivity experiments with ```poetry run python3 experiments/sensitivity/sensitivity.py```
