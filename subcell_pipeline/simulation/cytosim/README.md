# Cytosim simulations

Simulations and processing for cytoskeleton simulation engine [Cytosim](https://gitlab.com/f-nedelec/cytosim).

> - **Base simulator**: [https://gitlab.com/f-nedelec/cytosim](https://gitlab.com/f-nedelec/cytosim)
> - **Model development**: [https://github.com/simularium/Cytosim](https://github.com/simularium/Cytosim)

## Baseline single actin fiber with no compression

The `ACTIN_NO_COMPRESSION` simulation series simulates a single actin fiber with a free barbed end across five replicates.

- **Run Cytosim single fiber simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/cytosim/_run_cytosim_no_compression_batch_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/cytosim/_run_cytosim_no_compression_batch_simulations.html))
- **Process Cytosim single fiber simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/cytosim/_process_cytosim_no_compression_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/cytosim/_process_cytosim_no_compression_simulations.html))

## Single actin fiber compressed at different compression velocities

The `ACTIN_COMPRESSION_VELOCITY` simulation series simulates compression of a single 500 nm actin fiber at four different velocities (4.7, 15, 47, and 150 μm/s) with five replicates.

- **Run Cytosim compression simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/cytosim/_run_cytosim_compression_batch_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/cytosim/_run_cytosim_compression_batch_simulations.html))
- **Process Cytosim compression simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/cytosim/_process_cytosim_compression_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/cytosim/_process_cytosim_compression_simulations.html))
