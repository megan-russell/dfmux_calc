import numpy as np
import matplotlib.pyplot as plt
import dfmux_calc as d
import nep_calc as n
import matplotlib
matplotlib.use('Agg')

import sys

path = sys.argv[1]
sys.path.append(path)
from config import *


if mut != False:
        saa.change_mutual_ind(mut)
        
        
#printing info about how close to done this is
tracker_total = 100*len(bands)
tracker_status = 0
print('Calculating needed SQUIDs at {} bands, for {} bolometer resistances, and {} stray resistances for a total of {} scenarios'.format(len(bands), 10, 10, tracker_total))
print('... {}% complete'.format(round(tracker_status/tracker_total*100)), end='', flush=True)


note =  []

for band in bands:
    lb = n.experiment('litebird',0) #loading definitions about bands
    popt = lb.popt[band] #optical power in the band of interest
    psat = 2.5 * popt *psat_factor    #saturation power in the band


    #required readout NEP
    nep = np.sqrt(lb.ntot[band]**2 - lb.rreq[band]**2)  * np.sqrt((1+frac)**2 - 1)

    #define operating bolometer resistance and stray to sweep through
    rbolo, rstray = np.meshgrid(np.linspace(rbolo_min, rbolo_max , r_steps), np.linspace(rstray_min, rstray_max, r_steps))

    #what voltage bias the detector should be operated at
    vbias = np.sqrt( (rbolo + rstray) * (psat - popt) )

    #aproximate responsivity of the detector
    resp = np.sqrt(2) / vbias * bolo.loopgain / (1 + bolo.loopgain * (rbolo - rstray)/(rbolo + rstray ) ) 

    #by what factor the loogain is being reduced by the stray resistance
    loop_atten = (rbolo - rstray)/(rbolo + rstray )

    #what the required NEI for the band is
    nei_req = nep * resp

    #initializing arrays to store the required power consumption of the SAA, the number of SQUIDs in the array 
    #and an array to note when there is no found solution to meet the requirements
    req_power = nei_req.copy()
    req_nsq = nei_req.copy()
    fail = nei_req.copy()
    csf = nei_req.copy()

    #baseline SAA and other parasitics

    
    
    

    

    dfm = d.dfmux_noise(saa,bolo,wh,para,nuller_cold=nuller_cold)

    

    #calc needed NEI at each combo
    #step through each
    for i in range(10):
        #print(i)
        for j in range(10):
            #print(i,j)
            dfm.bolo.r = rbolo[i][j]
            dfm.bolo.rstray = rstray[i][j]
            target = nei_req[i][j]*1e12
            #print(target)
            n_sq = 1
            #increase number of SAA until NEI met
            while True:
                #print(i,j,n_sq)
                dfm.squid.scale_SAA(n_sq, p_banks)
                dfm.init_freq([np.max(bias_f)],skip_spice=skip_spice,csf_factor = csf_factor)
                if max(dfm.total) <= target:
                    req_power[i][j] = dfm.squid.power
                    req_nsq[i][j] = n_sq
                    fail[i][j] = 0
                    csf[i][j] = dfm.csf[0]
                    
                    if [i,j] == [itarget,jtarget]:
                        dfm.init_freq(bias_f,skip_spice=skip_spice,csf_factor=csf_factor)
                        fig, ax = plt.subplots()
                        d.plot_noise(dfm,bias_f,u'#1f77b4')
                        plt.plot([np.min(bias_f)/1e6,np.max(bias_f)/1e6],[target,target],'--',label='NEI Requirement',c=u'#ff7f0e')
                        ax.set_title('Readout NEI for {} GHz band $R_{{bolo}}$={}$\Omega$ $R{{stray}}$={}$\Omega$, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                        lb.opt_freqs[band] ,round(dfm.bolo.r,2), round(dfm.bolo.rstray,2), bolo.loopgain, round(nep*1e18,1), frac*100))
                        plt.legend()
                        plt.savefig(path + '/band_'+str(band) + '_readout_nei.png')
                        
                        fig, ax = plt.subplots()
                        plt.plot(bias_f/1e6, 1/dfm.tf, label='1/TF')
                        plt.plot(bias_f/1e6, dfm.csf, label='CS')
                        plt.xlabel('Bias frequency [MHz]')
                        ax.set_title('TF + CS for {} GHz band $R_{{bolo}}$={}$\Omega$ $R{{stray}}$={}$\Omega$, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                        lb.opt_freqs[band] ,round(dfm.bolo.r,2), round(dfm.bolo.rstray,2), bolo.loopgain, round(nep*1e18,1), frac*100))
                        plt.legend()
                        plt.savefig(path + '/band_'+str(band) + '_tf_cs.png')
                        
                        note.append(dfm.squid.power)
                    
                    #print(n_sq)
                    break
                else:
                    n_sq += sq_step
                    if n_sq <=100:
                        continue
                    else:
                        #print('fail!:',max(dfm.total))
                        req_power[i][j] = dfm.squid.power
                        req_nsq[i][j] = n_sq
                        fail[i][j] = 1
                        
                        if [i,j] == [itarget,jtarget]:
                            dfm.init_freq(bias_f,skip_spice=skip_spice,csf_factor=csf_factor)
                            fig, ax = plt.subplots()
                            d.plot_noise(dfm,bias_f,u'#1f77b4')
                            plt.plot([np.min(bias_f)/1e6,np.max(bias_f)/1e6],[target,target],'--',label='NEI Requirement',c=u'#ff7f0e')
                            ax.set_title('Readout NEI for {} GHz band $R_{{bolo}}$={}$\Omega$ $R{{stray}}$={}$\Omega$, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                            lb.opt_freqs[band] ,round(dfm.bolo.r,2), round(dfm.bolo.rstray,2), bolo.loopgain, round(nep*1e18,1), frac*100))
                            plt.legend()
                            plt.savefig(path + '/band_'+str(band) + '_readout_nei.png')
                            
                            fig, ax = plt.subplots()
                            plt.plot(bias_f/1e6, 1/dfm.tf, label='1/TF')
                            plt.plot(bias_f/1e6, dfm.csf, label='CS')
                            plt.xlabel('Bias frequency [MHz]')
                            ax.set_title('TF + CS for {} GHz band $R_{{bolo}}$={}$\Omega$ $R{{stray}}$={}$\Omega$, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                            lb.opt_freqs[band] ,round(dfm.bolo.r,2), round(dfm.bolo.rstray,2), bolo.loopgain, round(nep*1e18,1), frac*100))
                            plt.legend()
                            plt.savefig(path + '/band_'+str(band) + '_tf_cs.png')
                        
                        break
            tracker_status += 1
            print('\r... {}% complete'.format(round(tracker_status/tracker_total*100)), end='',flush=True)

        #calculate power per mux
        #print(j)


    #plot    
    fig, ax = plt.subplots()

    c = ax.pcolormesh(rbolo, rstray, req_nsq, cmap='jet',vmin=1,vmax = 50,shading='auto')
    c2 = ax.pcolormesh(rbolo, rstray, np.where(fail == 1, 1, np.nan), cmap='binary', vmin=0, vmax=1,zorder=10,shading='auto')
    CS = ax.contour(rbolo, rstray, loop_atten, 6, colors='w') 
    ax.clabel(CS, fontsize=9, inline=True)
    ax.set_title('# SQ requirement for {} GHz band: $P_{{sat}}$={}pW, $P_{{opt}}=${}pW, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                        lb.opt_freqs[band] ,round(psat*1e12,2), round(popt*1e12,2), bolo.loopgain, round(nep*1e18,1), frac*100))
    cbar = fig.colorbar(c, ax=ax)

    plt.xlabel('$R_{bolo}$ [$\Omega$]')
    plt.ylabel('$R_{stray}$ [$\Omega$]')
    cbar.set_label('Required SQUIDs in array')
    plt.savefig(path + '/band_'+str(band) + '_SQ_req.png')



    fig, ax = plt.subplots()

    c = ax.pcolormesh(rbolo, rstray, nei_req*1e12, cmap='jet',vmin=5,vmax = 15,shading='auto')
    #d = ax.pcolormesh(rbolo, rstray, np.where(fail == 1, 1, np.nan), cmap='binary', vmin=0, vmax=1,zorder=10)
    CS = ax.contour(rbolo, rstray, loop_atten, 6, colors='w') 
    ax.clabel(CS, fontsize=9, inline=True)
    ax.set_title('NEI requirement for {} GHz band: $P_{{sat}}$={}pW, $P_{{opt}}=${}pW, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                        lb.opt_freqs[band] ,round(psat*1e12,2), round(popt*1e12,2), bolo.loopgain, round(nep*1e18,1), frac*100))
    cbar = fig.colorbar(c, ax=ax)

    plt.xlabel('$R_{bolo}$ [$\Omega$]')
    plt.ylabel('$R_{stray}$ [$\Omega$]')
    cbar.set_label('Required NEI [pA/$\sqrt{\mathrm{Hz}}$]')
    plt.savefig(path + '/band_'+str(band) + '_NEI_req.png')


    fig, ax = plt.subplots()

    c = ax.pcolormesh(rbolo, rstray, csf, cmap='jet',vmin=1,shading='auto')
    c2 = ax.pcolormesh(rbolo, rstray, np.where(fail == 1, 1, np.nan), cmap='binary', vmin=0, vmax=1,zorder=10,shading='auto')
    CS = ax.contour(rbolo, rstray, loop_atten, 6, colors='w') 
    ax.clabel(CS, fontsize=9, inline=True)
    ax.set_title('CSF at 4.5MHz for {} GHz band: $P_{{sat}}$={}pW, $P_{{opt}}=${}pW, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                        lb.opt_freqs[band] ,round(psat*1e12,2), round(popt*1e12,2), bolo.loopgain, round(nep*1e18,1), frac*100))
    cbar = fig.colorbar(c, ax=ax)

    plt.xlabel('$R_{bolo}$ [$\Omega$]')
    plt.ylabel('$R_{stray}$ [$\Omega$]')
    cbar.set_label('CSF @ 4.5MHz')
    plt.savefig(path + '/band_'+str(band) + '_csf.png')
    
    
    
    fig, ax = plt.subplots()
    
    if max_power > max(req_power.flatten()*1e9):
        vmax =  max(req_power.flatten()*1e9)
    else:
        vmax = max_power
    c = ax.pcolormesh(rbolo, rstray, req_power*1e9, cmap='jet',shading='auto', vmax = vmax)
    c2 = ax.pcolormesh(rbolo, rstray, np.where(fail == 1, 1, np.nan), cmap='binary', vmin=0, vmax=1,zorder=10,shading='auto')
    CS = ax.contour(rbolo, rstray, loop_atten, 6, colors='w') 
    ax.clabel(CS, fontsize=9, inline=True)
    ax.set_title('SQUID power dissipation for {} GHz band: $P_{{sat}}$={}pW, $P_{{opt}}=${}pW, \n $\mathcal{{L}}=${}, NEP$_{{read}}$={}aW$/\sqrt{{\mathrm{{Hz}}}}$, {}% NEP increase'.format(
                        lb.opt_freqs[band] ,round(psat*1e12,2), round(popt*1e12,2), bolo.loopgain, round(nep*1e18,1), frac*100))
    cbar = fig.colorbar(c, ax=ax)

    plt.xlabel('$R_{bolo}$ [$\Omega$]')
    plt.ylabel('$R_{stray}$ [$\Omega$]')
    cbar.set_label('Power dissipated by SQUID Array [nW]')
    plt.savefig(path + '/band_'+str(band) + '_sq_power.png')
    
    
    
    

print(bands,note)