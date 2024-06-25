# ReaDDy simulations

Simulations and processing for particle-based reaction-diffusion simulator [ReaDDy](https://readdy.github.io/).

## Run ReaDDy simulations (compression and no compression)

> - **Base simulator**: [https://github.com/readdy/readdy](https://github.com/readdy/readdy)
> - **Model development**: [https://github.com/simularium/readdy-models](https://github.com/simularium/readdy-models)

- **Run ReaDDy compression simulations** ([source](https://github.com/simularium/readdy-models/tree/main/examples/actin) | [readme](https://github.com/simularium/readdy-models/blob/main/examples/README.md))


## Process baseline single actin fiber with no compression

The `ACTIN_NO_COMPRESSION` simulation series simulates a single actin fiber with a free barbed end across five replicates.

- **Process ReaDDy single fiber simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy/_process_readdy_no_compression_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/readdy/_process_readdy_no_compression_simulations.html))

## Process single actin fiber compressed at different compression velocities

The `ACTIN_COMPRESSION_VELOCITY` simulation series simulates compression of a single 500 nm actin fiber at four different velocities (4.7, 15, 47, and 150 μm/s) with five replicates.

- **Process Cytosim compression simulations** ([source](https://github.com/simularium/subcell-pipeline/blob/main/subcell_pipeline/simulation/readdy/_process_readdy_compression_simulations.py) | [notebook](https://simularium.github.io/subcell-pipeline/_notebooks/simulation/readdy/_process_readdy_compression_simulations.html))
