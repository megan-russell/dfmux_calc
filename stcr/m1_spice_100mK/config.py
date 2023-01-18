#input parameters
import dfmux_calc as d
import numpy as np



##############################
######## SQUID INFO ##########
##############################
#Do you want to increase the mutual inductance of the SAA versus baseline input?
mut = 1 #set to a number probably between 1-3 if you want to increase the mutual inductance
p_banks = 1    #number of parallel banks in the SAA to use, SAA provided will be rescaled to this
sq_step = 1   #number of SQUIDs to add to the array each time the system fails to meet noise requirements
              #smaller steps take longer to run, but give more precise minimum number of SQ required
#STCR E112 properties representative of the median SAA 
saa = d.squid(1500,                                           #Transimpedence [Ohms]
                   700,                                       #Dynamic impedence [Ohms]
                   2e-12,                                     #NEI [A/rtHz] (just assuming that this has about ~2x the NEI of the SA13)
                   20e-9,                                     #Input inductance [H]
                   n_series=112,n_parallel=1,power=25e-9,     #array size and power dissipation 
                   snubber=5,                                 #if there is a snubber on the input
                   t=0.1)                                     #what temperature the SAA is at [K]


##############################
######## PARASITICS ##########
##############################
#Wiring harness properties
wh = d.wire(30,                                #resistance [Ohms]
                40e-12,                        #capacitance [F] (this is the important one)
                0.75e-6,                       #inductance  [H]
                rshunt=False, cshunt=False)    #if theres any resistive or capcitive shunts across the output of the SAA
#Other parasitics
stripline = 60e-12 #assumed stripline inductance
cgnd = 0.7e-12  #assumed parasitic capacitance to ground
para = d.parasitics(stripline,cgnd,0) #stripline inductance[H], parasitic capacitance to ground[F] and R48 [Ohms]


##############################
######## BOLOMETERS ##########
##############################
#bolometer properties
bolo = d.bolo(1.0,                   #operating impedence [Ohms] this is ignored in litebird.py
                  10,                #loopgain 
                  0.2,               #stray impedence [Ohms] this is ignored in litebird.py
                  2.5 * 0.24187821,  #psat - this is ignored in litebird.py
                  0.171,0.1)         #Tc and Tb [K]
#which bolometer resistance, stray resistances to do calulations for
#and which combination to do a more detailed plot of what the readout solution looks like
itarget=5
jtarget=5

rbolo_min = 0.5
rbolo_max = 1.0
rstray_min = 0.0
rstray_max =0.2
r_steps = 10

bands = [1,11,18] #which bands to use in calculation
psat_factor = 1.0

##############################
######### FDM OTHER ##########
##############################
#bias frequencies of the detectors
#litebird.py only looks at the highest bias frequency
bias_f = np.linspace(1.5e6,5.5e6,68)
skip_spice = False
frac = 0.1  #the readout increases the total internal (photon, phonon) by frac
nuller_cold = True #if the nuller resistors are at 4K or 300K
csf_factor = False #if you want to assume the CSF is of by x factor






    
##############################
########### PLOTS ############
##############################   
max_power = 10  #max sq power dissipation to show on plots colorbar















