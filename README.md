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
3. `python -m pip install -e .[dev]`

## Quickstart

```python
import numpy as np
from subcell_analysis.compression_analysis import get_end_to_end_axis_distances_and_projections

test_polymer_trace = np.array([
    [2,0,0],
    [4,1,0],
    [6,0,-1],
    [8,0,0],
])

# prints the following:
# distance of polymer trace points from the end-to-end axis:
# (array([0., 1., 1., 0.])
# scaled distances of projection points along the end-to-end-axis:
# array([0., 0.33, 0.67, 1.]),
# positions of projection points on the end-to-end axis:
# array([[2., 0., 0.],
#        [4., 0., 0.],
#        [6., 0., 0.],
#        [8., 0., 0.]]))
print(get_end_to_end_axis_distances_and_projections(test_polymer_trace))
```

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
