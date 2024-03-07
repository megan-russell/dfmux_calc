# dfmux_calc

### Introduction

The goal of this code is to estimate the readout noise of a full DfMUX system, including a SQUID, wiring harness, and some parasitics. The bolometer's effect on the readout circuit is included, as well as its Johnson noise. However, phonon and photon noise are not treated in `dfmux_calc`, as they are not considered "readout noise."

### Usage

Please see `LiteBIRD_SSAA_parameters.ipynb` for an example, as well as a brief explanation of how to install SPICE, which is an optional dependency.

### Revision History

 - Aug 2022: first version committed to https://github.com/megan-russell/dfmux_calc
 - Mar 2024: forked and simplified, hosted at https://github.com/tijmen/dfmux_calc

### Citations

 - Joshua Montgomery's PhD Thesis - https://escholarship.mcgill.ca/concern/theses/1c18dm29r
 - 2021 SPT-3G paper https://arxiv.org/abs/2103.16017
 - Joshua Montgomery's Masters Thesis - https://www.proquest.com/docview/2514727898?pq-origsite=gscholar&fromopenview=true