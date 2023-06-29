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

### Setup 
1. create a virtual env: `conda create -n subcell_analysis python=3.10`
2. `conda activate subcell_analysis`
3. `conda install readdy==2.0.9`
3. `pip install -e '.[lint,test,docs,dev]'`

**Note: `just install` will not install `readdy` correctly, which is needed for tests to pass.

## Documentation

For full package documentation please visit [Simularium.github.io/subcell-analysis](https://Simularium.github.io/subcell-analysis).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## Glossary of terms
Definitions of some terms used in these analyses
* *polymer trace*:

    refers to the line traced by a polymer backbone in three dimensions. For cytosim, this corresponds to the positions of the control points. For ReaDDy, this corresponds to a derived metric that traces the polymer backbone

* *end-to-end axis*:

    refers to the line connecting the first and last points of a polymer trace

**Apache Software License 2.0**
