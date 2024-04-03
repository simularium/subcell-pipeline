# Post-processing of ReaDDy and Cytosim Outputs

This document provides instructions on how to perform post-processing of ReaDDy and Cytosim outputs.

## ReaDDy Output Post-processing

Run `create_dataframes_from_readdy_outputs.py` to create dataframes from ReaDDy outputs. This script reads the ReaDDy output files and creates dataframes that can be used for further analysis.

```bash
python create_dataframes_from_readdy_outputs.py
```
The dataframes will be saved in the `data/dataframes/readdy` directory.


## Cytosim Output Post-processing

Run `create_dataframes_from_cytosim_outputs.py` to create dataframes from Cytosim outputs. This script reads the Cytosim output files and creates dataframes that can be used for further analysis.
```bash
python create_dataframes_from_cytosim_outputs.py
```
The dataframes will be saved in the `data/dataframes/cytosim` directory.


## Combining outputs

Run `create_combined_dataframe.py` to combine ReaDDy and Cytosim outputs. This script reads the previously generated ReaDDy and Cytosim output files and combines them into a single dataframe.

```bash
python create_combined_dataframe.py
```

The combined dataframe will be saved in the `data/dataframes` directory.
