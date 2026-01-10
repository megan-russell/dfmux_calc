# dfmux_calc

Calculations related to the Simons Array / SPT-3G / LiteBIRD DfMux readout system for TES bolometers.

 - `dfmux_calc.py` is the main entry point. The goal of this code is to estimate the readout noise of a full DfMUX system, including a SQUID, wiring harness, and some parasitics. The bolometer's effect on the readout circuit is included, as well as its Johnson noise. However, phonon and photon noise are not treated in `dfmux_calc`, as they are not considered "readout noise."
 - `current_sharing.py` is needed by dfmux_calc if SPICE is enabled. 
 - `LiteBIRD_SSAA_parameters.ipynb` shows an example of how to use `dfmux_calc` for calculation of the LiteBIRD SQUID electrical performance requirements. It also has a brief explanation of how to install SPICE, which is an optional dependency.
 - `mntes.py` calculates the properties of a TES in the DfMux readout system including nonlinearity. 

### Using Docker to Run `dfmux_calc`

If you wish to use the SPICE-enabled calculation of current sharing (i.e., nulling efficiency), be aware that 
PySpice is an older library and may not function in modern environments. To address this, you can use Docker. Note that if PySpice runs well in your environment, the additional hassle of using a Docker container is not recommended.

If you wish to run `dfmux_calc` in a Docker container, follow these steps:

1. **Modify the Dockerfile**:  
   Edit the provided `Dockerfile` to match your preferred paths before building the image.

2. **Build the Docker image**:  
   In the root directory of the repository, run the following command to build the Docker image:
   ```bash
   docker build -t dfmux_calc .
   ```

3. **Run the container**:  
   Use the following command to execute the container and mount a local directory for output files (e.g., plots):
   ```bash
   docker run -v /path/to/local/plots:/app/plots dfmux_calc
   ```

   - Replace `/path/to/local/plots` with the full path to the directory on your machine where you want the output files to be saved.
   - Inside the container, the directory is mapped to `/app/plots`.

This setup ensures that `dfmux_calc` runs in an environment that is compatible with the PySpice and libngspice dependencies.

### Revision History

 - Nov 2024: Important precision enhancement in SPICE calculation, added Dockerfile 
 - Mar 2024: forked and simplified, hosted at https://github.com/tijmen/dfmux_calc
 - Aug 2022: first version committed to https://github.com/megan-russell/dfmux_calc


### Citations

 - Joshua Montgomery's PhD Thesis - https://escholarship.mcgill.ca/concern/theses/1c18dm29r
 - 2021 SPT-3G paper https://arxiv.org/abs/2103.16017
 - Joshua Montgomery's Masters Thesis - https://www.proquest.com/docview/2514727898?pq-origsite=gscholar&fromopenview=true
