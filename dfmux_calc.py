#import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import constants as c
colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', 
          u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
import importlib.util
import current_sharing as cs



#helper class to store SAA information
class squid:
    def __init__(self, zt, rdyn, inoise, lin, lstray = 10e-9,
                 n_series=False, n_parallel = False, power = False,
                 linear_range=2e-6, snubber = False, snubber_c = False, t=0.3):
        self.zt = zt            #Transimpedance of SAA
        self.rdyn = rdyn        #Dynamic impedance of SAA
        self.inoise = inoise    #SAA noise refered to input coil
        self.lin = lin          #input inductance of the SAA
        self.lstray = lstray
        self.n_series = n_series #number of individual SQUIDs in series to form the SAA
        self.n_parallel = n_parallel #number of banks of SQUIDs in parallel to form the SAA
        self.power = power      #power dissipated by SAA when in operation
        self.linear_range=linear_range #linear range of SAA input in Amps
        self.snubber = snubber  # Resistance of any snubber used to regulate the SAA
        self.snubber_c = snubber_c# Capcitance of any snubber used to regulate the SAA
        self.t = t              #the temperature the SAA is operating at in Kelvin. This is ignored outside of 
                                #calculating the johnson noise of any snubber. 
            
        self.m_factor = 1       #starting with a mutual inductance factor of 1 - this is just a note of how far
                                #from base mutual inductance this has been modified

    #method to rescale the existing SAA to a new array
    def scale_SAA(self,new_series, new_parallel):
        if not self.n_series and not self.n_parallel and not self.power:
            print('squid has no specified array size and/or power!')
            return
        self.zt *= new_series / self.n_series
        self.rdyn *= new_series / new_parallel * self.n_parallel / self.n_series
        self.lin = (self.lin - self.lstray) * new_series * new_parallel / self.n_series / self.n_parallel + self.lstray
        self.inoise *= np.sqrt(self.n_series * self.n_parallel ) / np.sqrt(new_series * new_parallel)
        self.power *= new_series * new_parallel / self.n_series / self.n_parallel
        self.n_series = new_series
        self.n_parallel = new_parallel
        
    #method to scale the mutual inductance of the existing SAA up or down
    def change_mutual_ind(self, m_factor):
        self.zt *= m_factor
        self.lin = (self.lin - self.lstray) * m_factor**2 + self.lstray
        self.linear_range *= m_factor
        self.m_factor *= m_factor
        
    def print_info(self):
        print('Transimpedence: {} $\Omega$'.format(round(self.zt,0)))
        print('Dyn. impedence: {} $\Omega$'.format(round(self.rdyn,0)))
        print('Input inductance: {} nH'.format(round(self.lin/1e-9,0)))
        print('NEI: {} pA/rtHz'.format(round(self.inoise/1e-12,1)))
        print('Power dis.: {} nW'.format(round(self.power/1e-9,0)))
        print('Array size: {}x{}'.format(self.n_series, self.n_parallel))
        print('Linear range: {} $\mu$A'.format(round(self.linear_range/1e-6,2)))
        

#helper class to store bolometer information
class bolo:
    def __init__(self,r,loopgain,rstray,psat,tc,tb):
        self.r = r                    #Operating resistance of the bolometer in ohms
        self.loopgain = loopgain      #operating loopgain of the bolometer
        self.rstray = rstray          #stray resistance in series with the bolometer in ohms
        self.psat = psat              #saturation power of the bolometer in watts
        self.tc = tc                  #critical temperature of the bolometer in kelvin
        self.tb = tb                  #bath temperature the bolometer is operated at in kelvin
        
#helper class to store important parasitics
class parasitics:
    def __init__(self,stripline, c_gnd, r48=0):
        self.stripline = stripline    #self inductance in henries of the stripline conecting the SAA to the rest of the 
                                      #DfMUX circuit - or in its absence and stray inductance that looks like it
        self.c_gnd = c_gnd            #the parasitic capacitance in farads to ground from the focal plane, bolometers or LCs
        self.r48 = r48                #The value in ohms of R48 on the SQUID Controller Boards - this connected the negative
                                      #side of the nuller to ground and in 2022 its default value is 0 Ohms

#helper class to store wiring harness information and do some calculations    
class wire:
    def __init__(self, r, c, l, rshunt=False, cshunt=False):
        self.r = r              #wiring harness series resistance
        self.c = c              #capacitance
        self.l = l              #series self inductance
        self.rshunt = rshunt    #if a shunt resistor is places across the SAA output on the SQCB
        self.cshunt = cshunt    #if a shunt capacitor is placed across the SAA output on the SQCB
        
        
    #Calculate the series impedance of the wiring harness at a single given frequency
    #deceause of how transmission line models typically count wire resistance + inductance this is equivalent to
    #1 single wire of double length
    def series_imp(self,f):
        return 2j * np.pi * f * self.l + self.r
    
    
    #this calculates the complex ABCD or cascade matrix for the wiring harness using the transmision line model
    #at a given single frequency, taking into account resistive or capacitive shunts which are assumed to be
    #at the 300K input to a SQUID Controller board
    def get_abcd(self,f):
        wire_series = 2j * np.pi * f * self.l + self.r
        wire_shunt = 2j * np.pi * f * self.c
        
        gamma = np.sqrt(wire_series * wire_shunt)
        z0 = np.sqrt(wire_series / wire_shunt)
        
        b = z0 * np.sinh(gamma)
        if self.rshunt != False:
            a = np.cosh(gamma) + b/self.rshunt
        else:
            a = np.cosh(gamma)
        if self.cshunt != False:
            a = a + b * 2j*np.pi*f*self.cshunt
            
        
        d = np.cosh(gamma)
        if self.rshunt != False:
            c = 1/z0 * np.sinh(gamma) + d/self.rshunt
        else:
            c = 1/z0 * np.sinh(gamma)
        if self.cshunt != False:
            c = c + d * 2j*np.pi*f*self.cshunt
        return a, b, c, d
    
    
    #Calculates the effective resistance seen by the input of the 1st stage amplifier on the SQUID Controller board
    #this is used to calculate how the input current noise of that amplifer translates to voltage noise
    #at a given frequency and for a given squid with a dynamic impedance
    def reff(self,squid,f):
        [a,b,c,d] = self.get_abcd(f)
        return np.abs( (b + d * squid.rdyn)/( a + c * squid.rdyn))
    
    #Calculates the real effective resistance seen by the input of the 1st stage amplifier on the SQUID Controller board
    #without the SAA dynamic impedance, this is used to calculate the johnson noise of the wiring harness and any shunt elements
    def real_reff(f):
        [a,b,c,d] = self.get_abcd(f)
        return np.real( (b )/( a ))
        
            
    #calculates the voltage transfer function from the mK output of the SAA to the input of the 300K SQCB
    #for a list of frequencies and SAA with a dynamic impedance and stores the transfer function as a property of the wire
    def transfer_function(self, squid, frequencies):
        self.tf =[]        
        for f in frequencies:
            [a,b,c,d] = self.get_abcd(f) 
            self.tf.append( np.abs(1/(a + c * (squid.rdyn  ) - ( b + d * (squid.rdyn  ))/1e6 )) )
        return self.tf
    
    
    
    #Calculates the nuller(/carrier transfer function with inductive bias elements) from the wiring harness 
    #returns a ratio of current cold to current warm
    def nuller_tf(self,inductor,frequencies):
        itf = []
        for f in frequencies:
            
            [a,b,c,d] = self.get_abcd(f) 
            itf.append( np.abs(1/(c*2j*np.pi*f*inductor-d)))
        return itf



class dfmux_noise:
    
    def __init__(self, squid, bolo, wire, para, nuller_wire=None, nuller_cold=False):
        self.squid = squid         #a squid object 
        self.bolo = bolo           #a bolometer object
        self.wire = wire           #a wiring harness object
        self.para = para           #a parasitics object
        self.nuller_wire = nuller_wire #a wiring harness object for the nuller lines if this is different than the ones used for the
        self.nuller_cold = nuller_cold #if the current stiffening resistors for the nuller are located on the 300K stage this is False, if they are located on the 4K stage this is true
                                   #SQUID output lines. By default this is None and it is assumed they are the same.
        
        #Warm electrnics noise
        #carrier = 1.6e-12                                        #A/rtHz JM PhD Table 7.5
        carrier = 2.9e-12/(bolo.r + bolo.rstray)                  #A/rtHz JM PhD Table 7.5 with bias johnson removed, and scaled by bolometer resistance
        if self.nuller_cold:
            nuller = np.sqrt(0.38e-12**2 + 3.6e-12**2)           #A/rtHz JM PhD p176 + table 7.6
        else:
            nuller = 4.9e-12                                     #A/rtHz JM PhD Table 7.6
        self.warm_noise_nc = np.sqrt(carrier**2 + nuller**2)     #total noise from the carrier/nuller refered to SAA input
        

        #bolometer noise
        self.jnoise = np.sqrt(2) * 1/(1+self.bolo.loopgain)*np.sqrt(4*1.38e-23*self.bolo.tc / (self.bolo.r))  #JM masters section 5.6
        
        #if a snubber is being used - add the johnson noise of it in quadrature with bolometer johnson noise
        #this assumes that the snubber is at the same temperature stage as the SAA 
        if self.squid.snubber != False:
            self.jnoise = np.sqrt(self.jnoise**2 + 4*1.38e-23*self.squid.t / (self.squid.snubber)) 
        
    

    def init_freq(self, frequencies, #frequencies to calculate noise at
                  dan=True,          #if this is a dan on noise calculation or not
                  skip_spice = False,#if you want to skip the pyspice sim and fall back on an approximation
                  recal_csf = True,  #if you want to recalculate csf from scratch each time , only set to False for iterating SAA size, only Lin is changed
                  csf_factor = False, #if you want to assume you're underestimating the CSF by x% set to 1.x instead of False and it will be applied to the calculated csf
                  csf=None):         #if you want to calculate noise with a given csf (this must be the same size as frequencies)
        
            
        
        self.f = np.array(frequencies)
        self.csf = []                    #current sharing factor
        self.tf = []                     #wiring harness transfer function
        self.saa_in_impedance = []       #impedance of the SAA imput coil
        self.on_res_comb_impedance = []  #impedance of the comb assuming on resonance
        self.c_r48 = []                  #impedance of the parasitic path through ground and R48
        self.demod = []                  #total demodulator noise refered to the input of the SAA A/rtHz
        self.total = []                  #total noise refered to the input of the SAA pA/rtHz
        #self.wire_j = []
        if not dan:
            self.jnoise_f = []
            self.warm_noise_nc_f = []
            self.saa_scale_f = []
        
        
        
        #demod chain noise
        #see JM PhD table 7.7 1st stage amplifier
        if self.wire.rshunt==False:
            req=(1/(1/10 + 1/100 + 1/150) + 1/(1/4.22e3 + 1/ (self.wire.reff(self.squid,self.f))) ) #RSQCB as defined in table 7.7
            first_amp = np.sqrt(2) * np.sqrt((1.1e-9)**2  +
                                (2.2e-12* req)**2)
            
            
        #this is a modification to the previous line that adds the johnson noise of the shunt resistor
        #currently assuming that it is well represented by a current source in paraelle with the 1st stage amplifier
        #which adds in quadrature with that current noise and is then scaled by the impedance as defined in Joshua's thesis
        #the change in Reff due to the shunt will be automaticaly handled by the reff function
        else:
            req=(1/(1/10 + 1/100 + 1/150) + 1/(1/4.22e3 + 1/ (self.wire.reff(self.squid,self.f))) ) #RSQCB as defined in table 7.7
            first_amp = np.sqrt(2) * np.sqrt(   (1.1e-9)**2  +
                                ( np.sqrt(  (2.2e-12)**2 +  4* 1.38e-23 *300/self.wire.rshunt  ) * req)**2
                                    #+(4* 1.38e-23 *300 / (self.wire.r+self.wire.rshunt)*(self.wire.reff(self.squid,f))**2))
                                    #+(4* 1.38e-23 *300 * (self.wire.r+self.wire.rshunt))
                                   #(2.2e-12 * req)**2 +  
                                   #        1.38e-23 *300/self.wire.rshunt * (self.wire.reff(self.squid,f))**2 
                                                )

        #table 7.7- the rest of the terms
        demod_dc = np.sqrt(first_amp**2 + 
                               2*(0.23e-9**2 +     #ADC noise
                                  0.14e-9**2 +     #2nd stage amplifier
                               #0.36e-9**2 +        #Signal path johnson - think this is just wiring harness johnson
                                  (8.36e-9*self.wire.reff(self.squid,self.f)/(self.wire.reff(self.squid,self.f)+4.22e3))**2   #SQUID I bias johnson
                                  + 4* 1.38e-23 *300 * self.wire.r ))  #johnson noise of the wiring harness resistance
                                                                      #pessimisitcally assuming that the entire harness is at 300K
            
            
            
        if dan:
            #in this case estimate the current sharing factor based off of JM SPT-3G noise paper
            try:
                if csf == None:
                    spec = importlib.util.find_spec('PySpice')
                    if spec is None or skip_spice == True:
                        if not skip_spice:
                            print("PySpice is not installed... continuing with analytic approximation")


                        ##Calculate the impedances necesary for current sharing

                        #the input impedance of the SAA
                        self.saa_in_impedance =  2 * np.pi * self.f * self.squid.lin  

                        #the comb impedance- assuming this is for an on resonance frequency
                        if not self.squid.snubber:
                            self.on_res_comb_impedance =  2 * np.pi * self.f * self.para.stripline + self.bolo.r + self.bolo.rstray  
                        else:
                            self.on_res_comb_impedance =   1/(
                                1/(2 * np.pi * self.f * self.para.stripline + self.bolo.r + self.bolo.rstray  ) + 1/self.squid.snubber ) 

                        #the impedance of the path through ground and R48
                        self.c_r48 =   1/(2 * np.pi * self.f * self.para.c_gnd) + self.para.r48 

                        #combining those terms to estimate the current sharing factor- from joshua's spt3g paper
                        if self.nuller_wire == None:
                            self.csf =  1/(self.c_r48 / ( (self.on_res_comb_impedance + np.abs(self.wire.series_imp(self.f))) * 
                                             self.saa_in_impedance / 
                               (self.on_res_comb_impedance + self.saa_in_impedance + np.abs(self.wire.series_imp(self.f))) + 
                               np.abs(self.wire.series_imp(self.f)) + self.c_r48 ) * 
                                    self.on_res_comb_impedance / (self.on_res_comb_impedance + self.saa_in_impedance))

                        #if nuller_wire is not none this has a different wiring than connected to the SAA output
                        else:
                            self.csf =  1/(self.c_r48 / ( (self.on_res_comb_impedance + np.abs(self.nuller_wire.series_imp(self.f))) * 
                                             self.saa_in_impedance / 
                               (self.on_res_comb_impedance + self.saa_in_impedance + np.abs(self.nuller_wire.series_imp(self.f))) + 
                               np.abs(self.nuller_wire.series_imp(self.f)) + self.c_r48 ) * 
                                    self.on_res_comb_impedance / (self.on_res_comb_impedance + self.saa_in_impedance))
                    else:
                        #instead use a full PySpice calculation
                        self.csf = cs.get_csf(self)
                    
                
            #in this case take the measured CSF given as input
            except:
                self.csf = np.array(csf)
            
        #this is an option to calculate SQCB input refered noise with DAN off forcing CSF to not impact the noise
        else:
            self.csf = np.ones(lens(self.f))
                
        #applying the CSF factor if asked for
        if csf_factor != False:
            self.csf *= csf_factor
                
        #calculating the transfer function caused by the SAA Z_dyn and the wiring harness capacitance/any shunts across SAA
        self.tf = np.array(self.wire.transfer_function(self.squid,self.f))
            
            
        #scaling the noise from the warm electronics by the current sharing factor, the transfer function
        # and the transimpedance of the saa to refer it to the SAA input coil - units now A/rtHz
        if dan:
            self.demod= demod_dc * self.csf / self.tf / self.squid.zt 
        else:
            self.demod =  demod_dc * np.ones(self.f) 
                
        #self.wire_j.append(np.sqrt(4* 1.38e-23 *300 * self.wire.real_reff(self.squid,f)*np.sqrt(2)*self.csf[-1]/self.squid.zt))

            
        #scaling the noise of the SAA by the current sharing factor and the demodulation factor
        if dan:
            self.saa_scale = self.squid.inoise * self.csf * np.sqrt(2)
        #or if dan off noise by the transimpedance and tf to refer to SQCB
        else:
            self.saa_scale_f = self.squid.inoise * self.squid.zt * self.tf * np.sqrt(2)
            
        #total noise from all sources in order the noise from the carrier/nuller chain, the johnson noise of the bolo
        #the scaled demodulator chain noise, and the scaled SAA noise in PICOAMPS/rtHz
        if dan:
            self.total = np.sqrt(self.warm_noise_nc**2 + self.jnoise**2 + self.demod**2 + self.saa_scale**2)*1e12
            
        #if dan is off refering remaining terms to SQCB input - outputs noise in nV/rtHz
        else:
            self.warm_noise_nc_f = self.warm_noise_nc* self.squid.zt* self.tf
            self.jnoise_f = self.jnoise* self.squid.zt* self.tf
            self.total = np.sqrt((self.warm_noise_nc_f)**2  + (self.jnoise_f)**2 + 
                                          self.demod**2 + self.saa_scale_f**2)*1e9
                
        


#helper function which takes a SQUID object as input and returns another SQUID object as output
#it takes warm transimpecence dyn impedance etc measurements dont with a resistive shunt in place
#and calculates the expected parameters for the SQUID on its own, assuming the measurements were done
#with a 1.6 uA current bias step as is pydfmux's default behavior
def refer_squid(warm_squid, shunt):
    delta_v = warm_squid.rdyn*1.6e-6
    cold_zdyn = delta_v / (1.6e-6 - delta_v /shunt)
    
    cold_zt = warm_squid.zt * (cold_zdyn + shunt ) / shunt
    
    return squid(cold_zt, cold_zdyn, warm_squid.inoise, warm_squid.lin)
  

#function to produce conversion factor from pA/rtHz NEI to NEP
def nei_to_nep(dfmux_noise,optical_power):
    vbias = np.sqrt(dfmux_noise.bolo.r * (dfmux_noise.bolo.psat - optical_power) )
    responsivity = np.sqrt(2) * dfmux_noise.bolo.loopgain/(
        1+dfmux_noise.bolo.loopgain*(dfmux_noise.bolo.r - dfmux_noise.bolo.rstray)/(
            dfmux_noise.bolo.r + dfmux_noise.bolo.rstray))/ vbias
    return 1/responsivity



#function to make plots of the noise 

def plot_noise(dfmux_noise,f,c,label=None):
    plt.plot(f/1e6,dfmux_noise.total,c=c,label=label)
    plt.legend()
    plt.plot(f/1e6,np.abs(dfmux_noise.demod)*1e12 ,'--',label='Expected DEMOD noise',c=c)
    plt.plot(f/1e6,np.abs(dfmux_noise.saa_scale )*1e12 ,'-.',label='Expected SAA noise',c=c)
    plt.plot(f/1e6,[np.abs(dfmux_noise.jnoise )*1e12 for i in range(len(f))] ,':', label = 'Johnson',lw=2,c=c)
    plt.plot(f/1e6,[np.abs(dfmux_noise.warm_noise_nc )*1e12 for i in range(len(f))] ,':', label = 'warm n/c',c=c)
    if c == colors[0]:
        plt.legend()
    plt.xlabel('Bias frequency [MHz]')
    plt.ylabel('Noise [pA/rtHz]')
    
    
def sweep_squids(dfmux_noise,start_n=10,end_n=200,step=5):
    f1 = plt.figure('total')
    f2 = plt.figure('csf')
    f3 = plt.figure('tf')
    
    norm = mpl.colors.Normalize(vmin=start_n, vmax=end_n)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])
    
    
    for n in range(start_n, end_n, step):
        dfmux_noise.squid.scale_SAA(n,dfmux_noise.squid.n_parallel)
        dfmux_noise.init_freq(dfmux_noise.f)
        
        plt.figure('total')
        plt.plot(dfmux_noise.f/1e6,dfmux_noise.total,c=cmap.to_rgba(n))
        
        
        plt.figure('csf')
        plt.plot(dfmux_noise.f/1e6,dfmux_noise.csf,c=cmap.to_rgba(n))
        
        
        plt.figure('tf')
        plt.plot(dfmux_noise.f/1e6,dfmux_noise.tf,c=cmap.to_rgba(n))
        
        
    plt.figure('total')
    plt.xlabel('Bias Frequency [MHz]')
    plt.ylabel('Total noise [pA/$\sqrt{\mathrm{Hz}}$]')
    cbar = f1.colorbar(cmap)
    cbar.set_label('Number of SQUIDs in series')
    
    
    plt.figure('csf')
    plt.xlabel('Bias Frequency [MHz]')
    plt.ylabel('Current sharing factor')
    cbar = f2.colorbar(cmap)
    cbar.set_label('Number of SQUIDs in series')
    
    
    
    plt.figure('tf')
    plt.xlabel('Bias Frequency [MHz]')
    plt.ylabel('SAA transfer function')
    cbar = f3.colorbar(cmap)
    cbar.set_label('Number of SQUIDs in series')
        
        
        
#function to make plots of noise as function of bolometer operating impedance and stray resistance
def plt_nei_v_r(saa, bolo, wh, para,f,vmin=None,vmax=None):
    
    rbolo, rstray = np.meshgrid(np.linspace(0.2, 1.0 , 100), np.linspace(0, 0.2, 100))

    noise_min = np.zeros(rbolo.shape)
    noise_max = np.zeros(rbolo.shape)

    i=0
    j=0
    for b in rbolo[0]:
        for s in rstray[:,0]:
            bolo.r=b
            bolo.rstray = s
            cnoise1 = dfmux_noise(saa,bolo,wh,para,nuller_cold=True)
            cnoise1.init_freq(f)
            noise_min[i][j] = np.min(cnoise1.total)
            noise_max[i][j] = np.max(cnoise1.total)
            j+=1
        
        i+=1
        j=0
    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(rbolo, rstray, noise_min, vmin=vmin, vmax=vmax, cmap='jet')
    CS = ax.contour(rbolo, rstray, noise_max, 6, colors='k') 
    ax.clabel(CS, fontsize=9, inline=True)
    cbar = fig.colorbar(c, ax=ax)
    
    plt.xlabel('$R_{bolo}$')
    plt.ylabel('$R_{stray}$')
    cbar.set_label('Low bias frequency noise [pA/$\sqrt{\mathrm{Hz}}$]')
    
    
