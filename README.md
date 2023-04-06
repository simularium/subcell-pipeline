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
1. create a virtual env: `conda create -n subcell_analysis`
2. `conda activate subcell_analysis`
3. `python -m pip install -e .[dev]`

## Quickstart

```python
from subcell_analysis import example

print(example.str_len("hello"))  # prints 5
```

## Documentation

For full package documentation please visit [Simularium.github.io/subcell-analysis](https://Simularium.github.io/subcell-analysis).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**Apache Software License 2.0**
