"""
This code uses dfmux calc to simulate the readout noise 
of polarbear 2b detectors as they go in transition.

Other useful files:
in tostada simons_array_offline_software/scratch/nfarias/redout_noise_analysis/pb2b_prepare_data_to_compare_to_dfmux_calc.ipynb
    
"""


import numpy as np
import matplotlib.pyplot as plt
import paramiko
from io import StringIO
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d


# Plot Parameters
plt.rcParams["font.family"] = "FreeSerif"
plt.rcParams.update({'font.size': 18})
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rcParams['mathtext.fontset'] = 'stix'

import dfmux_calc as d
# from scipy.stats import norm
# import nep_calc as nep
import pickle as pk
import csv

from datetime import datetime

###### SETUP #####
TEST = False
noise_freq_range = np.array([6, 10]) # where we are measuring noise in pb2b
omega = 2*np.pi * np.mean(noise_freq_range)


# Setup Rfracs to look at
#r_fracs = np.array([1.0, 0.84, 0.82, 0.8,  0.78, 0.73, 0.68, 0.63, 0.5]) # tuning targets for these runs
r_fracs = np.array([1.0])

# Measurement files
# noise_dir = '/home/nfarias/nicole_data/unzipped_measure_noise/'
# noise_file = noise_dir + '20230912_003547_measure_noise/data/2785_BOLOS_INFO_AND_MAPPING.pkl'
squid_params_file = '/home/nfarias/nicole_data/unzipped_measure_noise/20230911_222403_SQUIDParams.pkl'

# Get today's date in YYYYMMDD format for saving the figures
today_date = datetime.now().strftime("%Y%m%d")


def main():
    
    print("Let's model the readout noise behavior as a function of Rfrac")
    
    # Connect to tostada to get measured stuff
    sftp = connect_to_tostada()
    print("Established connection to tostada")
    data = load_detector_parameters_from_csv(sftp) # note, the data is in string format, needs to be converted
    S_I_data = load_responsivity_data_from_csv(sftp)
        
    measured_nei_data = load_nei_data_from_csv(sftp)
    
    ###### SETUP THE DFMUX SYSTEM ######
    # Unpack measured quantities
    det_names = data[:, 0]
    f_bias_array = data[:, 1].astype(float)
    measured_ob_noise = data[:, 2].astype(float)
    rstrays = data[:, 3].astype(float)
    stripline_inductance = np.median(data[:, 4].astype(float))
    r_normals = data[:, 5].astype(float) * 0.9375
    zt = np.unique(data[:, 6]).astype(float)
    
    # get rdyn via model of zt vs rdyn (code in tostada to get zt can't load rdyn)
    rdyn = model_rdyn(zt)
    
    # initialize squid amplifier array
    SAA = initialize_squid(zt, rdyn, type="sa_13")
    # initialize wiring harness properties
    wh = initialize_wiring_harness()
    # initialize other parasitics
    para = d.parasitics(stripline_inductance, 0.7e-9, 0) #stripline inductance[H], parasitic capacitance to ground[F] and R48 [Ohms]
    

    print("Simulation parameters")
    print("stripline inductance: ", stripline_inductance)
    print("median stray: ", np.median(rstrays))
    print("median normal: ", np.median(r_normals))
    print("Zt: ", zt, " Rdyn: ", rdyn)

    
    # if this is a test, only use five detectors
    if TEST:
        rstrays = rstrays[0:5]
        r_normals = r_normals[0:5]
        f_bias_array = f_bias_array[0:5]
    n_dets = len(f_bias_array) # number of detectors
    
    # See noise behavior as a function of Rfrac

    
    for r, rfrac in enumerate(r_fracs):
        print("Rfrac: ", rfrac)
        
        all_total_noises = np.array([])
        
        # Setup each bolometer individually
        exp_values_readout = np.zeros(n_dets) # to store the expected readout noise of each det
        exp_values_total = np.zeros(n_dets) # to store the expected total noise of each det
        readout_neps = np.array([]) # initialize empty because I don't want to store the 
                                    # overbiased values if they exist
        total_neps = np.array([])  # initialize empty because I don't want to store the 
                                    # overbiased values if they exist
        measured_neis = np.zeros(n_dets)  # to store the measured (interpolated if not available) NEI of each detector                            
        measured_neps = np.array([])
                                    
        # go through each bolo
        for dc in range(0, n_dets):
            det = det_names[dc]
            rbolo = r_normals[dc] * rfrac
            rstray = rstrays[dc]
            f = [f_bias_array[dc]] # code breaks if this is not an array
            
            # setup the bath temperature depending on whether or not we are overbiased
            Tb = 0.5 if rfrac == 1.0 else 0.3
            
            if det in S_I_data[:, 0]:
                # get the effective time constant, loop gain and dc responsivity for this bolometer at this Rfrac
                tau, loop_gain, S_I_DC = interpolate_responsivity_data_one_det(det, r_normals[dc]*rfrac, S_I_data)
                    
                # force loop gain to be zero if rfrac is 1:
                if rfrac >= 1.0:
                    loop_gain = 0
                
                # Initialize bolometer
                bolo = d.bolo(rbolo,            # operating impedence [Ohms]
                    loop_gain,                # loopgain
                    rstray,                   # stray impedence [Ohms]
                    5e-12,                    # psat - this isn't used at all
                    2e-12,
                    0.5, Tb)                 #Tc and Tb [K] (ASSUMING EVERYTHING IS COLD)
                
                # Calculate NEI
                pb_noise = d.dfmux_noise(SAA, bolo, wh, para, nuller_cold=False,bias_res=True)
                pb_noise.init_freq(f, csf_factor=1.0) 
                total_noise = pb_noise.total.copy()
                readout_only = np.sqrt(pb_noise.warm_noise_nc**2 + pb_noise.demod**2 + pb_noise.saa_scale**2)
            
                # get expectation value of noise in A/sqrt(Hz)
                exp_values_readout[dc] = np.median(readout_only, axis=0)
                exp_values_total[dc] = np.median(total_noise, axis=0)
                # save all total noises for comparison
                all_total_noises = np.concatenate((all_total_noises, total_noise.flatten()))
                
                # GET MEASURED NEI FOR COMPARISON
                if rfrac < 1:
                    measured_neis[dc] = interpolate_measured_nei_data_one_det(det,  r_normals[dc]*rfrac, measured_nei_data)
                else:
                    measured_neis[dc] = measured_ob_noise[dc]/1e12
                
                # CALCULATE NEP if we are in transition
                if rfrac < 1:               
                    S_I_in_band = np.abs(S_I_DC / (1 + 1j*tau*omega) )
                    # simulated, readout only
                    readout_nep = exp_values_readout[dc]/S_I_in_band
                    readout_neps = np.append(readout_neps, readout_nep)
                    # simulated, readout + Johnson
                    total_nep = exp_values_total[dc]/S_I_in_band
                    total_neps = np.append(total_neps, total_nep)
                    # measured, total
                    measured_nep = measured_neis[dc]/S_I_in_band
                    measured_neps = np.append(measured_neps, measured_nep)


       



        #### SAVE DATA ########
        np.save(f'/home/nfarias/nicole_data/NEI_vs_Rfrac/date_{today_date}_readout_nei_{rfrac}.npy', exp_values_readout)
        np.save(f'/home/nfarias/nicole_data/NEI_vs_Rfrac/date_{today_date}_total_nei_{rfrac}.npy', exp_values_total)
        np.save(f'/home/nfarias/nicole_data/NEI_vs_Rfrac/date_{today_date}_total_nep_{rfrac}.npy', total_neps)
        np.save(f'/home/nfarias/nicole_data/NEI_vs_Rfrac/date_{today_date}_readout_nep_{rfrac}.npy', readout_neps)
        np.save(f'/home/nfarias/nicole_data/NEI_vs_Rfrac/date_{today_date}_measured_nep_{rfrac}.npy', measured_neps)
        np.save(f'/home/nfarias/nicole_data/NEI_vs_Rfrac/date_{today_date}_measured_nei_{rfrac}.npy', measured_neis)
    
        
        ##### MAKE PLOTS #########
        
        if rfrac < 1:                     
            # PLOT NEP
            plt.figure(2, figsize=(8,8))
            bins=np.linspace(0, 250, 32)
            plt.hist(readout_neps*1e18 , label=str(round(rfrac, 2)), alpha=0.5, bins=bins)
            plt.xlabel("Readout NEP [aW/sqrt(Hz)]")
            plt.ylabel("Number of detectors")
            plt.legend(title="$R_{frac}$")
            plt.grid(True)
            
            
            plt.figure(4, figsize=(8,8))
            bins=np.linspace(0, 250, 32)
            plt.hist(total_neps*1e18 , label=str(round(rfrac, 2)), alpha=0.5, bins=bins)
            plt.xlabel("Total NEP [aW/sqrt(Hz)]")
            plt.ylabel("Number of detectors")
            plt.legend(title="$R_{frac}$")
            plt.grid(True)
        
        else:
            # plot comparison of overbiased noise with simulated overbiased noise
            plt.figure(3, figsize=(8,8))
            bins=np.linspace(15, 32, 16)
            # plt.hist(exp_values_readout*1e12 , label="sim readout only", alpha=0.5, bins=bins)
            plt.hist(exp_values_total*1e12 , label="dfmux_calc", alpha=0.5, bins=bins)
            plt.hist(measured_ob_noise , label="measured", alpha=0.5, bins=bins)
            plt.xlabel("Noise equivalent current [pA/sqrt(Hz)]")
            plt.ylabel("Number of detectors")
            # plt.title("Using Megan's rstray and rnormal")
            plt.legend()
            plt.title("Total overbiased noise")
            plt.grid(True)
            
            print("Median measured overbiased NEI: ", round(np.median(measured_ob_noise), 2) )
            print("Median simulated overbiased NEI: ", round(np.median(exp_values_readout*1e12), 2) )
            
            
            # plot comparison of overbiased noise with simulated overbiased noise
            print(all_total_noises)
            plt.figure(3444, figsize=(8,8))
            bins=np.linspace(15, 32, 16)
            # plt.hist(exp_values_readout*1e12 , label="sim readout only", alpha=0.5, bins=bins)
            plt.hist(all_total_noises*1e12 , label="dfmux_calc", alpha=0.5, bins=bins)
            plt.hist(measured_ob_noise , label="measured", alpha=0.5, bins=bins)
            plt.xlabel("Noise equivalent current [pA/sqrt(Hz)]")
            plt.ylabel("Number of detectors")
            # plt.title("Using Megan's rstray and rnormal")
            plt.legend()
            plt.title("Total overbiased noise")
            plt.grid(True)
            plt.savefig(f"images/date_{today_date}_all_overbiased_noise.png")
            
            
            
            
        # PLOT THINGS
        plt.figure(1, figsize=(8,8))
        bins=np.linspace(10, 40, 20)
        plt.hist(exp_values_readout*1e12 , label=str(round(rfrac, 2)), alpha=0.5, bins=bins)
        plt.xlabel("Readout noise [pA/sqrt(Hz)]")
        plt.ylabel("Number of detectors")
        # plt.title("Using Megan's rstray and rnormal")
        plt.legend(title="$R_{frac}$")
        plt.grid(True)
        # PRINT THINGS
        print("Median readout NEI: ", round(np.median(exp_values_readout*1e12), 2) )
        
    plt.figure(1)
    plt.savefig(f"images/date_{today_date}_readout_noise_vs_rfrac.png")
    plt.figure(2)
    plt.savefig(f"images/date_{today_date}_readout_nep_vs_rfrac.png")
    plt.figure(3)
    plt.savefig(f"images/date_{today_date}_overbiased_noise.png")
    plt.figure(4)
    plt.savefig(f"images/date_{today_date}_total_nep_vs_rfrac.png")
    
    print("Done. \n\n")



def model_rdyn(zt, plot:bool=True):
    """
    Function to model the squid dynamic impedance
    in the case where we only have transimpedance, using the rdyn and zdyn data
    for all squids in pb2b.
    Inputs: 
    - array or float zt
    - plot (bool): whether or not to plot the fit of rdyn vs zt
    Outputs: 
    - array or float rdyn
    """
     # load average measured squid properties to get rdyn
    zts, rdyns = load_measured_squid_properties()  #this loads the average properties
    all_zts = np.array(zts)
    all_rdyns = np.array(rdyns)
    
    # see if we can get rdyn based on a model 
    mask_zt = (all_zts > 600) & (all_zts < 1100)
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(all_zts[mask_zt].reshape(-1, 1), all_rdyns[mask_zt])

    # Get the slope (coefficient) and intercept
    slope = model.coef_[0]
    intercept = model.intercept_
    zt_for_plot = np.linspace(600, 900, 10)
    
    if plot:
        plt.figure(9, figsize=(8,8))
        plt.scatter(all_zts[mask_zt], all_rdyns[mask_zt], color='black', label='pb2b data')
        plt.plot(zt_for_plot, slope*zt_for_plot + intercept, color='blue', label='fit')
        plt.xlabel("Zt [Ohms]")
        plt.ylabel("Zdyn [Ohms]")
        plt.legend()
        plt.savefig("images/rdyn_vs_zt")
    
    # calculate the expected dynamic impedance based on typical values
    rdyn = zt * slope + intercept  
    return rdyn

def interpolate_measured_nei_data_one_det(det, desired_R_TES, measured_nei_data):
    """
    Function to get the total NEI of a given detector
    at a given tes resistance.
    This is done by interpolating the measured nei data that we have at a few 
    different resistances
    """
    
    det_mask = measured_nei_data[:, 0] == det # find the responsivity data for this particular detector
    R_TESs = measured_nei_data[:, 1][det_mask].astype(float)
    measured_neis = measured_nei_data[:, 2][det_mask].astype(float)
    
    interp_function_nei = interp1d(R_TESs, measured_neis, \
    kind='linear', fill_value='extrapolate',  bounds_error=False)
    interpolated_nei = interp_function_nei(desired_R_TES)
    
    return interpolated_nei

def interpolate_responsivity_data_one_det(det, desired_R_TES, S_I_data, phase="i_phase"):
    """
    Function to get the responsivity and loop gain for a given detector
    at a given tes resistance.
    This is done by interpolating the responsivity data that we have at a few 
    different resistances
    """
    
    det_mask = S_I_data[:, 0] == det # find the responsivity data for this particular detector
    R_TESs = S_I_data[:, 1][det_mask].astype(float)
    loop_gains = S_I_data[:, 2][det_mask].astype(float)
    taus = S_I_data[:, 3][det_mask].astype(float)
    S_I_mags = S_I_data[:, 4][det_mask].astype(float) # total magnitude of the responsivity
    S_I_phases = S_I_data[:, 5][det_mask].astype(float) # phase of the responsivity with respect to I
    
    # calculate the responsivity in the desired phase
    if phase == 'i_phase':
        S_Is = S_I_mags * np.cos(S_I_phases)
    
    # interpolate to get the time constant at the desired TES resistance
    interp_function_tau = interp1d(R_TESs, taus, \
        kind='linear', fill_value='extrapolate',  bounds_error=False)
    
    interpolated_tau = interp_function_tau(desired_R_TES)
    
    # interpolate to get the loop gain at the desired TES resistance
    interp_function_loop_gain = interp1d(R_TESs, loop_gains, \
        kind='linear', fill_value='extrapolate',  bounds_error=False)
    
    interpolated_loop_gain = interp_function_loop_gain(desired_R_TES)
    
    # interpolate to get the responsivity at the desired TES resistance
    interp_function_S_I = interp1d(R_TESs, S_Is, \
        kind='linear', fill_value='extrapolate',  bounds_error=False)
    
    interpolated_S_I = interp_function_S_I(desired_R_TES)
    
    # values can't be smaller than zero --- for example, when we are overbiased, loop gain is zero and
    # we don't care about the responsivity (which I think is also zero?)
    if interpolated_loop_gain < 0:
        interpolated_loop_gain = 0
    if interpolated_S_I < 0:
        interpolated_S_I = 0
    if interpolated_tau < 0:
        interpolated_tau = 0
    
    return interpolated_tau, interpolated_loop_gain, interpolated_S_I

def load_responsivity_data_from_csv(sftp):
    """
    This code load important responsivity from a csv file in tostada.
    The csv was prepared on the notebook 
    /home/nfarias/simons_array_offline_software/scratch/nfarias/redout_noise_analysis/pb2b_prepare_data_to_compare_to_dfmux_calc.ipynb
    
    
    S_I_header = ['Detector name ', ' R_TES [Ohms] ',  'loop gain ', 'time constant [s]', \
    ' DC responsivity magnitude [A/W] ', ' responsivity phase [rad]']
    """
    
    
    # File location in tostada:
    csv_file_loc = '/home/nfarias/simons_array_offline_software/scratch/nfarias/redout_noise_analysis/'
    csv_file_name = csv_file_loc + 'responsivity_data_for_dfmux_calc_comparison_run_42000836.csv'
    
    with sftp.open(csv_file_name, mode="r") as remote_file:
        # Read the file content into memory
        file_content = remote_file.read()

    # Decode bytes to string (UTF-8 assumed)
    file_content_str = file_content.decode("utf-8")
    # Parse the file content using StringIO
    csv_data = StringIO(file_content_str)
    reader = csv.reader(csv_data)
       
    # Read and print each row
    S_I_data = []
    for r, row in enumerate(reader):
        if r ==0:
            continue # skip the header
        else:
            S_I_data.append(row)
            
    S_I_data = np.array(S_I_data)    
    
    return S_I_data

def load_nei_data_from_csv(sftp):
    """
    This code load nei data from a csv file in tostada.
    The csv was prepared on the notebook 
    /home/nfarias/simons_array_offline_software/scratch/nfarias/redout_noise_analysis/pb2b_prepare_data_to_compare_to_dfmux_calc.ipynb
    
    
    NEI_header = ['Detector name ', ' R_TES [Ohms] ', 'NEI (6-10 Hz) [A/sqrt(Hz)]']
    
    """    
    # File location in tostada:
    csv_file_loc = '/home/nfarias/simons_array_offline_software/scratch/nfarias/redout_noise_analysis/'
    csv_file_name = csv_file_loc + 'nei_data_for_dfmux_calc_comparison_run_42000836.csv'
    
    with sftp.open(csv_file_name, mode="r") as remote_file:
        # Read the file content into memory
        file_content = remote_file.read()

    # Decode bytes to string (UTF-8 assumed)
    file_content_str = file_content.decode("utf-8")
    # Parse the file content using StringIO
    csv_data = StringIO(file_content_str)
    reader = csv.reader(csv_data)
       
    # Read and print each row
    nei_data = []
    for r, row in enumerate(reader):
        if r ==0:
            continue # skip the header
        else:
            nei_data.append(row)
            
    nei_data = np.array(nei_data)    
    
    return nei_data

def load_detector_parameters_from_csv(sftp):
    """
    this code load important detector parameters that are not a function of Rfrac from a csv file in tostada.
    The csv was prepared on the notebook 
    /home/nfarias/simons_array_offline_software/scratch/nfarias/redout_noise_analysis/pb2b_prepare_data_to_compare_to_dfmux_calc.ipynb
    
    header information: ['Detector name ', ' bias frequency [Hz] ', ' overbiased NEI [pA/sqrt(s)] ', \
        ' Rs [Ohms]', "L_s [H]", " R_n [Ohms] ", " SQUID Zt "]
    """
    # File location in tostada:
    csv_file_loc = '/home/nfarias/simons_array_offline_software/scratch/nfarias/redout_noise_analysis/'
    csv_file_name = csv_file_loc + 'data_for_dfmux_calc_comparison_run_42000836.csv'

    with sftp.open(csv_file_name, mode="r") as remote_file:
        # Read the file content into memory
        file_content = remote_file.read()

    # Decode bytes to string (UTF-8 assumed)
    file_content_str = file_content.decode("utf-8")
    # Parse the file content using StringIO
    csv_data = StringIO(file_content_str)
    reader = csv.reader(csv_data)
    
    # Read and print each row
    data = []
    for r, row in enumerate(reader):
        if r ==0:
            continue # skip the header
        else:
            data.append(row)
            
    data = np.array(data)    
    return data

def connect_to_tostada():
    # Server details
    hostname = "tostada1.physics.berkeley.edu"
    port = 22  # Default SSH port
    username = "nfarias"
    password = "123elocim"
    # Connect to the remote server
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Accept unknown hosts
    ssh.connect(hostname, port, username, password)
    # Open an SFTP session
    sftp = ssh.open_sftp()
    
    return sftp

def initialize_r_normal():
    
    # From Megan
    r_normal = np.array([\
        0.9626058,  0.92854195, 0.97671754, 1.02534344, 0.90712921, 0.73541908,
        0.88514277, 0.84788629, 0.84710096, 0.673961,   1.08811606, 0.8800057,
        0.81783305, 1.03562617, 0.94716845,  1.07323231, 0.94703383, 0.94293215,
        0.84598007, 0.7227199 , 0.95915082,  0.90167417, 0.8608948 , 0.77598358,
        1.01380923, 0.89051242, 0.89894448,  0.74779782, 0.81738237, 0.82166417,
        1.14899087, 0.78721942, 1.00131014,  0.98617771, 1.10378498, 0.71130264,
        0.93687832, 0.87447909, 1.034875  ,  0.72230782] )
    
    return r_normal

def initialize_r_stray():
    """
    Function to initialize stray resistance array
    """
    # From Megan's:
    rstray = np.array([\
        0.14889617, 0.22575848, 0.17266659, 0.21391573, 0.21293207, 0.20291612,
        0.17974052, 0.13675212, 0.13878973, 0.19187134, 0.0703359 , 0.17498834,
        0.17734522, 0.16321762, 0.07591814, 0.11368773, 0.25449139, 0.15703011,
        0.22348886, 0.16820122, 0.19569417, 0.1801744 , 0.24450427, 0.24594952,
        0.18425028, 0.22912093, 0.23812164, 0.27830734, 0.21009893, 0.23573789,
        0.14902439, 0.27297393, 0.28568549, 0.19467917, 0.18442179, 0.25035455,
        0.22820852, 0.23692354, 0.15921249, 0.20781176])
    
    return rstray
       
def initialize_wiring_harness():
    wh = d.wire(30,                            #resistance [Ohms]
            40e-12,                        #capacitance [F] (this is the important one)
            0.75e-6,                       #inductance  [H]
            rshunt=False, cshunt=False)    #if theres any resistive or capcitive shunts across the output of the SAA
    return wh

def initialize_squid(zt, rdyn, type="sa_13"):
    # if type == "sa_13":
    #     #SA13 properties representative of the median SAA
    #     sa13 = d.squid( zt,                                        #Transimpedence [Ohms]
    #                     rdyn,                                      #Dynamic impedance [Ohms]
    #                 5e-12*np.sqrt(np.median(zt)/np.array(zt)),  #NEI [A/rtHz] (just taking the number JM used)
    #                 70e-9,                                      #Input inductance [H]
    #                 n_series=3*64,n_parallel=2,power=80e-9,     #array size and power dissipation
    #                 snubber=20,                                 #if there is a snubber on the input
    #                 t=4)
    #     return sa13
    if type == "sa_13":
        #SA13 properties representative of the median SAA
        sa13 = d.squid( zt,                                        #Transimpedence [Ohms]
                        rdyn,                                      #Dynamic impedance [Ohms]
                    5e-12*np.sqrt(np.median(zt)/np.array(zt)),  #NEI [A/rtHz]
                    70e-9,                                      #Input inductance [H]
                    n_series=3*64,n_parallel=2,power=80e-9,     #array size and power dissipation
                    snubber=20,                                 #if there is a snubber on the input
                    t=4)
        return sa13

def load_measured_squid_properties():
    """
    Function to load the measured squid properties in pb2b
    Returns measured transimpedances and dynamic impedances
    """
    pb_sqs = pk.load(open(squid_params_file, 'rb'))
    zt =  [pb_sqs[k]['transimpedance'] for k in pb_sqs.keys()]
    rdyn =  [pb_sqs[k]['output_impedance'] for k in pb_sqs.keys()]

    return zt, rdyn

def load_measured_overbiased_noise():
    """
    Function to load the overbiased noise measured in pb2b
    """
    di = pk.load(open(noise_file,'rb'))
    nei = []
    fbias = []
    for k in di.keys():
        if di[k]['calc_R']['V'] !=0:
            fbias.append(di[k]['frequency']/1e6)
            nei.append(di[k]['noise']['i_phase']['median_noise'])

    return nei, fbias
            
if __name__ == "__main__":
    main()