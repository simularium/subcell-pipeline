# subcell-analysis

[![Build Status](https://github.com/Simularium/subcell-analysis/workflows/CI/badge.svg)](https://github.com/Simularium/subcell-analysis/actions)
[![Documentation](https://github.com/Simularium/subcell-analysis/workflows/Documentation/badge.svg)](https://Simularium.github.io/subcell-analysis)

Analysis functionality for subcellular models

---

## Installation

**Stable Release:** `pip install subcell-analysis`<br>
**Development Head:** `pip install git+https://github.com/Simularium/subcell-analysis.git`

### Prerequisite 

1. Install Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Set up `just`: https://just.systems/man/en/chapter_4.html

### Setup 
1. create a virtual env: `conda create -n subcell_analysis python=3.10`
2. `conda activate subcell_analysis`
3. `just install`


## Usage

The subcell_analysis pipeline contains multiple steps that can be run independently. 

### 1. Run Simulations
The notebook aws_batch_job_submission.py can be used for configuring and executing simulations. Before this notebook can be succesfully executed, one must setup their AWSCLI credentials for submitting jobs and job definitions via the CLI. This notebook has five parts: 1a.) upload configuration files to s3, which involves generating the configuration files based on the template (vary_compress_rate.cym.tpl). Next, we have to create and register our job definition. In AWS Batch, a job definition specifies how many jobs are to be run, which docker image to use, how many CPUs to use, and the command a container should run when it is started.  We then submit our job to AWS using the submit_batch_job function. We can monitor these simulations as needed using check_batch_job. Finally, we can load in our results using the create_dataframes_for_repeats function which requires a bucket name, number of repeats, configs, and a save folder. 

### 2. Metrics Calculation


### 3. Comparative Metrics Calculation


### 4. Comparison with Tomography Data            


### 5. PCA/PacMAP generate_figure_data.ipynb



## Glossary of terms
Definitions of some terms used in these analyses
* *polymer trace*:

    refers to the line traced by a polymer backbone in three dimensions. For cytosim, this corresponds to the positions of the control points. For ReaDDy, this corresponds to a derived metric that traces the polymer backbone

* *end-to-end axis*:

    refers to the line connecting the first and last points of a polymer trace

**Apache Software License 2.0**
