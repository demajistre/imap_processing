import numpy as np
import ultra_user.data_access.MyUltraFile as MyUltraFile
import ultra_user.culling.UltraCull0 as UltraCull0
import spiceypy
import matplotlib.pyplot as plt
import imap_processing.spice.time as spiceTime


def l1c_energy_ranges(repoint=47, base_ebin=3, n_1cbins=8,set_maxbin:bool=True,maxbin_lim:float=100) -> np.ndarray:
    l1c = MyUltraFile.L1C(repoint).data
    l1c_ebins = np.transpose([l1c['energy_bin_geometric_mean'] - l1c['energy_delta_minus'],
                              l1c['energy_bin_geometric_mean'] + l1c['energy_delta_plus']])
    ebin_start = np.arange(base_ebin, len(l1c_ebins), n_1cbins, dtype=int)
    ebin_end = np.append(ebin_start[1:] - 1, len(l1c_ebins) - 1)
    nen = len(ebin_start)
    #ebin_start = ebin_start[:-1]
    eranges = np.ndarray((2, nen))
    for ic in range(len(ebin_start)):
        eranges[:, ic] = [l1c_ebins[ebin_start[ic]][0], l1c_ebins[ebin_end[ic]][1]]
    energy_ranges = np.transpose(eranges)
    if set_maxbin:
        iemax = np.nonzero(eranges[0, :] > maxbin_lim)[0][0]
        eranges1 = eranges[:, :iemax + 1]
        eranges1[1, iemax] = np.max(eranges[1, :])
        energy_ranges = np.transpose(eranges1)
    return energy_ranges


def get_pointings(start_pointing, end_pointing, sensor='90') -> list:
    result = list()
    for pointing in range(start_pointing, end_pointing):
        files = MyUltraFile.L1Bde(pointing, silent=True, sensor=sensor).fileCandidates()
        if len(files) > 0:
            result.append(pointing)
    return result


def runculls(pointings: list, energy_ranges: np.ndarray, sensor='90', earthAng45=np.radians(15), spin_range=20,n_iter=5,
             sep_threshold_per_spin=None,nAddChans=3):
    cullData = dict()
    ecull = dict()
    scull = dict()
    vcull = dict()
    cnt_sum = dict()
    cullFrac = dict()
    for repoint in pointings:
        print(repoint)
        cullData[repoint] = UltraCull0.UltraCull0(repoint, energy_ranges, sensor=sensor, spin_range=spin_range,
                                                  earthAng45=earthAng45,sep_threshold_per_spin=sep_threshold_per_spin)
        cnt_sum[repoint] = cullData[repoint].get_count_summary()
        vcull[repoint] = cullData[repoint].voltage_cull()
        ecull[repoint] = cullData[repoint].high_energy_cull(nAddChans=nAddChans)
        scull[repoint] = cullData[repoint].statistical_cull(n_iter=n_iter)
        cullFrac[repoint] = cullData[repoint].currentCullFraction()

    return {'cullData': cullData, 'ecull': ecull, 'scull': scull, 'vcull': vcull, 'cnt_sum': cnt_sum,
            'cullFrac': cullFrac}


def cullplot(cull: dict, echans: list = None, loud=False, start_utc="2026-01-01T00", chan_lims=False):
    if echans is None:
        echans = [0, 1, 2, 3]
    if chan_lims is False:
        chan_lims = [50, 30, 20, 10]
    nch = len(echans)
    t0 = spiceypy.sce2t(-43, spiceypy.str2et(start_utc)) * 2.e-5 + 1
    cullData = cull['cullData']
    cnt_sum = cull['cnt_sum']
    scull = cull['scull']
    repointings = list(cullData.keys())
    fig, axs = plt.subplots(nch)
    for repoint in repointings:
        if loud:
            print(repoint)
        tDay = (cullData[repoint].center_spin()['center_time'] - t0) / 86400
        for ech in range(nch):
            cnts = cnt_sum[repoint][:, ech]
            ii = np.nonzero(cullData[repoint].currentMask['bin_mask'][ech, :])[0]
            axs[ech].plot(tDay, cnts, 'r')
            if len(ii) > 0:
                color = 'g'
                if scull[repoint]['converge'][ech] == False:
                    color = 'b'
                axs[ech].plot(tDay[ii], cnts[ii], color)
    for ech in range(nch):
        axs[ech].set_ylim(0, chan_lims[ech])
        axs[ech].set_ylabel(f"counts ({ech})")
    axs[nch - 1].set_xlabel(f"days since {start_utc}")
    fig.suptitle(f"Ultra {cull['cullData'][repointings[0]].sensor}")
    plt.show()


def full_cntsum(cull: dict,precull_voltage=True) -> (np.ndarray, np.ndarray):
    repointings = list(cull['cullData'].keys())
    ii = range(len(cull['cullData'][repointings[0]].spinbins))
    if precull_voltage:
        ii = np.nonzero(cull['vcull'][repointings[0]]['binMask'])[0]
    full_sum = cull['cnt_sum'][repointings[0]][ii,:]
    start_spin = cull['cullData'][repointings[0]].spinbins[ii]
    for pointing in repointings[1:]:
        ii = np.arange(len(cull['cullData'][pointing].spinbins))
        if precull_voltage:
            ii = np.nonzero(cull['vcull'][pointing]['binMask'])[0]
        full_sum = np.vstack([full_sum, cull['cnt_sum'][pointing][ii,:]])
        start_spin = np.vstack([start_spin, cull['cullData'][pointing].spinbins[ii]])
    start_spin = start_spin[:, 0]
    return full_sum, start_spin
