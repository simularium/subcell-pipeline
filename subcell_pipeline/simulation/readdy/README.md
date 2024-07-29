# ReaDDy simulations

Simulations and processing for particle-based reaction-diffusion simulator [ReaDDy](https://readdy.github.io/).

> - **Base simulator**: [https://github.com/readdy/readdy](https://github.com/readdy/readdy)
> - **Model development**: [https://github.com/simularium/readdy-models](https://github.com/simularium/readdy-models)

## Baseline single actin fiber with no compression

The `NO_COMPRESSION` simulation series simulates a single actin fiber with a free barbed end across five replicates.

- **Run ReaDDy single fiber simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy/_run_readdy_no_compression_batch_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/readdy/_run_readdy_no_compression_batch_simulations.html))
- **Process ReaDDy single fiber simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy/_process_readdy_no_compression_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/readdy/_process_readdy_no_compression_simulations.html))

## Single actin fiber compressed at different compression velocities

The `COMPRESSION_VELOCITY` simulation series simulates compression of a single 500 nm actin fiber at four different velocities (4.7, 15, 47, and 150 μm/s) with five replicates.

- **Run ReaDDy compression simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy/_run_readdy_compression_batch_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/readdy/_run_readdy_compression_batch_simulations.html))
- **Process ReaDDy compression simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy/_process_readdy_compression_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/readdy/_process_readdy_compression_simulations.html))
