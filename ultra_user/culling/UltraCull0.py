from typing import Dict, Any, List

import numpy
import numpy as np
import ultra_user.data_access.MyUltraFile as MyUltraFile
import numpy.typing as npt
import imap_processing.spice.time as spiceTime
import ultra_user.planets.ENA_planets as ENA_planets


class UltraCull0():
    def __init__(self, repoint: int, energy_ranges: npt.NDArray, spin_range=20, rootDir='data/imap',
                 sensor='90', earthAng45=np.radians(20), sep_threshold_per_spin=None):
        if sep_threshold_per_spin is None:
            sep_threshold_per_spin = np.array([4., 2., 1.20, 0.45, 0.1, .1])
            # revised 3/31/26
            #sep_threshold_per_spin = np.array([4., 2., 1.25, 0.9, 0.2,.2])
        self.sep_threshold_per_spin = sep_threshold_per_spin
        self.currentMask = None
        self.repoint = repoint
        self.energy_ranges = energy_ranges
        self.sensor = sensor
        self.earthAng45 = earthAng45
        self.de = MyUltraFile.L1Bde(repoint, rootDir=rootDir,sensor=sensor).data
        self.xspin = MyUltraFile.L1Bxspin(repoint, rootDir=rootDir,sensor=sensor).data
        self.status = MyUltraFile.L1Bstatus(repoint, rootDir=rootDir,sensor=sensor).data
        self.spin_range = spin_range
        self.n_spinbin = int(len(self.xspin['spin_number']) / self.spin_range)
        self.spinbins = np.ndarray((self.n_spinbin, 2))
        self.binTimes = np.ndarray((self.n_spinbin, 2))
        self.sep_thresh = sep_threshold_per_spin*spin_range
        self.clear_mask()
        kc = 0
        iLast = len(self.xspin['spin_number']) - 1
        for ic in range(0, len(self.xspin['spin_number']), self.spin_range):
            if kc < self.n_spinbin:
                self.spinbins[kc, 0] = self.xspin['spin_number'][ic]
                self.binTimes[kc, 0] = self.xspin['spin_start_time'][ic]
                lastInd = np.min([ic + self.spin_range-1, iLast])
                self.spinbins[kc, 1] = self.xspin['spin_number'][lastInd]
                self.binTimes[kc, 1] = self.xspin['spin_start_time'][lastInd] + self.xspin['spin_period'][lastInd]
            kc = kc + 1

    def get_count_summary(self) -> npt.NDArray[int]:
        cnts = np.ndarray((self.n_spinbin, len(self.energy_ranges[:, 0])))
        for ic in range(len(self.energy_ranges[:, 0])):
            deMet = self.goodEventMet(ic)
            for jc in range(self.n_spinbin):
                #            jj0 = np.nonzero(
                #                np.logical_and(spins >= self.spinbins[jc, 0], spins < self.spinbins[jc, 1]))
                jj0 = np.nonzero(
                    np.logical_and(deMet >= self.binTimes[jc, 0], deMet < self.binTimes[jc, 1]))
                cnts[jc, ic] = len(jj0[0])
        return cnts

    def goodEventMet(self, ieBin:int) -> npt.NDArray:
        ebin_range = [1, 19]
        ii = np.nonzero(np.logical_and(
            np.logical_and(
                np.logical_and(np.logical_and(
                    np.logical_and(self.de['energy_spacecraft'] > self.energy_ranges[ieBin, 0],
                                   self.de['energy_spacecraft'] < self.energy_ranges[ieBin, 1]),
                    self.de['quality_outliers'] == 0), self.de['quality_scattering'] == 0),
                self.de['ebin'] >= ebin_range[0]), self.de['ebin'] <= ebin_range[1]))[0]
        if self.sensor == '45' and len(ii)>0:
            t0 = np.mean(self.de['event_times'][ii])
            try:
                earth = ENA_planets.ENA_planets(t0)
                local_uv = earth.local_uvec(self.de['velocity_dps_sc'][ii, :])
                coslim = np.cos(self.earthAng45)
                jj = np.nonzero(np.abs(local_uv[0, :] < coslim))[0]
                ii = ii[jj]
            except:
                print(f"No DPS frame data for {self.repoint}")
                ii = []
        return self.de['de_event_met'][ii]

    def get_dvolt_summary(self) -> (npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]):
        dvMean = np.full(self.n_spinbin, np.nan)
        dvMin = np.full(self.n_spinbin, np.nan)
        dvMax = np.full(self.n_spinbin, np.nan)
        istat = 0
        if max(self.status['shcoarse']) < self.binTimes[0, 0]:
            print("something is hosed")
            return dvMean, dvMin, dvMax
        while self.status['shcoarse'][istat] < self.binTimes[0, 0]:
            istat = istat + 1

        for ic in range(self.n_spinbin):
            if ic % 10 == 0:
                print(f"{ic} in {self.n_spinbin}")
            vset = list()
            while self.status['shcoarse'][istat] < self.binTimes[ic, 1] and istat < len(self.status['shcoarse']):
                vset.append(self.status['rightdeflection_v'][istat])
                vset.append(self.status['leftdeflection_v'][istat])
                istat = istat + 1
            if len(vset) > 0:
                dvMean[ic] = np.mean(np.array(vset))
                dvMin[ic] = np.min(np.array(vset))
                dvMax[ic] = np.max(np.array(vset))
        return dvMean, dvMin, dvMax

    def lowVoltagespins(self, threshold: float, return_binned=True) -> Dict:
        ii = np.nonzero(np.minimum(self.status['rightdeflection_v'], self.status['leftdeflection_v']) < threshold)[0]
        binmask = np.full(self.n_spinbin, True)
        if ii.size == 0:
            return {'lvtimes': list(), 'spin_index': list(),
                    'bin_index': list(), 'binMask': binmask}
        lvTimes = self.status['shcoarse'][ii]

        tSpins = self.xspin['spin_start_time']
        tBins = self.binTimes[:, 0]
        spin_index = list()
        bin_index = list()

        for tStat in lvTimes:
            iStart = np.searchsorted(tSpins, tStat) - 1
            jStart = np.searchsorted(tBins, tStat) - 1
            if iStart >= 0:
                spin_index.append(iStart)
            if jStart >= 0:
                bin_index.append(jStart)
        bin_index = list(dict.fromkeys(bin_index))
        spin_index = list(dict.fromkeys(spin_index))
        binmask[bin_index] = False
        return {'lvTimes': lvTimes, 'spin_index': spin_index,
                'bin_index': bin_index, 'binMask': binmask}

    def add_mask(self, mask: npt.NDArray, opName: str) -> None:
        self.currentMask['bin_mask'] = np.logical_and(self.currentMask['bin_mask'], mask)
        self.currentMask['status'].append(opName)
        return

    def clear_mask(self) -> None:
        self.currentMask = {'status': ['initialized'],
                            'bin_mask': self.empty_mask()}

    def empty_mask(self) -> npt.NDArray:
        return np.full((len(self.energy_ranges[:, 0]), self.n_spinbin),
                       True, dtype=bool)

    def voltage_cull(self, v_threshold: float = 3000, apply: bool = True) -> dict:
        lvSpins = self.lowVoltagespins(v_threshold, return_binned=True)
        v_mask = self.empty_mask()
        if apply is True:
            for ic in range(len(self.energy_ranges[:, 0])):
                v_mask[ic, :] = lvSpins['binMask']
            self.add_mask(v_mask, opName=f"Voltage: {v_threshold}")
        return lvSpins

    def center_spin(self) -> dict:
        return {'spin_bin': (self.spinbins[:, 0] + self.spinbins[:, 1]) / 2,
                'center_time': (self.binTimes[:, 0] + self.binTimes[:, 1]) / 2}

    def statistical_cull(self, n_iter: int = 5, std_thresh: float = 0.05,link_echans=True, apply: bool = True) -> dict:
        result = {'n_iter': n_iter, 'std_thresh': std_thresh, 'apply': apply}
        nen = len(self.energy_ranges[:, 0])
        cnt_summary = self.get_count_summary()
        mask = self.currentMask['bin_mask'].copy()
        nit = np.zeros(nen, dtype=int)
        conv = np.full(nen, dtype=bool, fill_value=False)
        std_diff = np.zeros(nen, dtype=float)
        fullmask = mask[0, :].copy()
        for ich in range(nen):
            cnt0 = cnt_summary[:, ich]
            icnt0 = np.arange(len(cnt0), dtype=int)
            for it in range(n_iter):
                cnt = cnt0[np.nonzero(mask[ich, :])]
                if len(cnt) < 3:
                    conv[ich] = True
                    mask[ich, :] = False
                    std_diff[ich] = -1
                    break
                icnt = icnt0[np.nonzero(mask[ich, :])]
                sdiff, submask = self.stat_iteration(cnt,icnt)
                std_diff[ich] = sdiff
                fullmask[icnt0[icnt[np.nonzero(submask)[0]]]] = False
                mask[ich, icnt0[icnt[np.nonzero(submask)[0]]]] = False
                # mask[ich,:] = np.logical_and(mask[ich,:],submask)
                nit[ich] = it + 1
                if std_diff[ich] < std_thresh:
                    nit[ich] = it + 1
                    conv[ich] = True
                    break
        if link_echans:
            for ich in range(nen):
                mask[ich,:] = fullmask
                # recalculate convergence just in case
                if not conv[ich]:
                    ii = np.nonzero(fullmask)[0]
                    cnts = cnt_summary[ii, ich]
                    icnts = np.arange(len(cnts), dtype=int) #we're only interested in the sdiff, this is fill
                    sdiff, _ = self.stat_iteration(cnts, icnts)
                    if sdiff < std_thresh:
                        conv[ich] = True
        result["link_echans"] = link_echans
        result["converge"] = conv
        result["iterations"] = nit
        result["mask"] = mask
        result["std_diff"] = std_diff
        if apply:
            self.add_mask(mask, f"statistical: converged={conv}, thresh={std_thresh}")
        return result

    def stat_iteration(self,cnt: np.ndarray,icnt: np.ndarray):
        mean = np.mean(cnt)
        std = np.std(cnt)
        std_diff = std / np.sqrt(mean) - 1
        submask = np.abs((cnt - mean) / std) > 3
        return std_diff,submask

    def combine_spin_bins(self,channel:int, nAddChans:int):
        cnt0 = np.float32(self.get_count_summary()[:,channel])
        if nAddChans == 0:
            return cnt0
        nbin = len(cnt0)
        cnt = np.zeros(nbin,dtype=float)
        for ic in range(nbin):
            im = np.max([0,ic-nAddChans])
            ip = np.min([ic+nAddChans,nbin-1])+1
            cnt[ic] = np.mean(cnt0[im:ip])
        return cnt
    def high_energy_cull(self, cull_channel=5, nAddChans=5, apply=True) -> dict:
        result = {'cull_channel': cull_channel, 'threshold': self.sep_thresh, 'apply': apply}
        nen = len(self.energy_ranges[:, 0])
        cnt_summary = self.get_count_summary()
        hichan = self.combine_spin_bins(cull_channel, nAddChans)
        #emask = np.logical_and(cnt_summary[:, cull_channel] < threshold, cnt_summary[:, cull_channel] >= 0)
        emask = self.currentMask['bin_mask'].copy()
        for ic in range(nen):
            emask[ic,:] = np.logical_and(hichan < self.sep_thresh[ic],
                                         hichan >= 0)
        if apply is True:
            self.add_mask(emask, opName=f"energy channel {cull_channel} counts < {self.sep_thresh}")
        result["mask"] = emask
        return result

    def spectral_cull(self, sigThreshold = 1,channels=None, apply=True) -> dict:
        if channels is None:
            channels = [0,1,2,4]
        nch=len(channels)
        result = {'channels':channels,'nch': nch, 'sigThreshold': sigThreshold, 'apply': apply}
        csum = self.get_count_summary()[:,channels]
        mask = self.currentMask['bin_mask'].copy()
        for ic in range(nch-1):
            diff = (csum[:, ic + 1] - csum[:, ic] - sigThreshold*(np.sqrt(csum[:,ic+1]+csum[:,ic])))
            ii = np.nonzero(diff > 0)[0]
            mask[:,ii] = False
        result["mask"] = mask
        if apply is True:
            self.add_mask(mask, opName="Upstream ion cull")
        return result

    def upstream_cull(self, sigThreshold = 2.5,channels=None, apply=True) -> dict:
        if channels is None:
            channels = [0,1,2]
        nch=len(channels)
        result = {'channels':channels,'nch': nch, 'sigThreshold': sigThreshold, 'apply': apply}
        csum = self.get_count_summary()[:,channels]
        scaled_cnt = np.zeros_like(csum)
        sumScaled_cnts = np.zeros_like(csum[:, 0])
        weights = np.zeros_like(csum[:, 0])
        mask = self.currentMask['bin_mask'].copy()

        for ic in range(nch):
            ii = np.nonzero(mask[channels[ic], :])[0]
            scaled_cnt[ii, ic] = csum[ii, ic]
            sumScaled_cnts[ii] += scaled_cnt[ii, ic]
            weights[ii] += 1
        kk = np.nonzero(weights > 0)[0]
        totalScaled = sumScaled_cnts[kk]
        totalMean = np.mean(totalScaled)
        thresh = totalMean + sigThreshold * np.sqrt(totalMean)
        jj = np.nonzero(totalScaled > thresh)[0]
        for ic in range(len(mask[:,0])):
            mask[ic,kk[jj]] = False
        result["mask"] = mask
        if apply is True:
            self.add_mask(mask, opName="Upstream ion cull")
        result['totalScaled'] = totalScaled
        result['scaled_counts'] = scaled_cnt
        result['thresh'] = thresh
        return result


#
#        for ic in range(nch):
#            ii = np.nonzero(mask[channels[ic], :])[0]
#            cntmean[ic] = np.mean(csum[ii, ic])
#            cntstd[ic] = np.std(csum[ii, ic])
#            scaled_cnt[ii, ic] = (csum[ii, ic] - cntmean[ic]) / cntstd[ic]
#            #sumScaled_cnts[ii] += scaled_cnt[ii, ic] * cntstd[ic]
#            #weights[ii] += cntstd[ic]
#            sumScaled_cnts[ii] += scaled_cnt[ii, ic]
#            weights[ii] += 1
#            sumScaled_cnts0[ii] += scaled_cnt[ii, ic] * np.sqrt(cntmean[ic])
#        kk = np.nonzero(weights > 0)[0]
#        totalScaled = sumScaled_cnts[kk] / weights[kk]
#        totalMean = np.mean(totalScaled)
#        totalStd = np.std(totalScaled)
#        thresh = totalMean + sigThreshold*totalStd
#        jj = np.nonzero(totalScaled > thresh)[0]
#        for ic in range(len(mask[:,0])):
#            mask[ic,jj] = False
#        result["mask"] = mask
#        if apply is True:
#            self.add_mask(mask, opName="Upstream ion cull")
#        result['totalScaled'] = totalScaled
#        result['scaled_counts'] = scaled_cnt
#        result['thresh'] = thresh
#        return result


    def currentCullFraction(self) -> npt.NDArray[float]:
        nen = len(self.energy_ranges[:, 0])
        result = numpy.ndarray(nen, dtype=float)
        for ic in range(nen):
            result[ic] = len(np.nonzero(self.currentMask['bin_mask'][ic])[0])/self.n_spinbin
        return result


# for debugging only
def main():
    repoint = 30
    energy_ranges = np.ndarray((6, 2))
    energy_ranges[0, :] = [4.6, 10.27]
    energy_ranges[1, :] = [10.27, 23.4444]
    energy_ranges[2, :] = [23.4444, 52.2113]
    energy_ranges[3, :] = [52.2113, 116.276]
    energy_ranges[4, :] = [116.276, 258.95]
    energy_ranges[5, :] = [258.95, 316.335]

    me = UltraCull0(repoint, energy_ranges)


if __name__ == "__main__":
    main()
