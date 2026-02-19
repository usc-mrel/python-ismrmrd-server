import numpy as np
from scipy.signal import find_peaks

def beat_rejection(pktimes, padloc, pkamps=None):
    '''Rejects the "bad beats", that are 3*std away from the mean
    heart rate, and optionally peaks that have amplitude at least 3*std away 
    from the mean amplitude.
    '''
    hr_perpeak = 60./np.diff(pktimes)
    hr_variation = np.std(hr_perpeak)
    hr_mean = np.mean(hr_perpeak)
    hr_diffs = hr_perpeak - hr_mean
    if padloc == "pre":
        hr_accept_list = np.hstack((1, abs(hr_diffs)<(3*hr_variation)))
    elif padloc == "post":
        hr_accept_list = np.hstack((abs(hr_diffs)<(3*hr_variation), 1))
    
    if pkamps is not None:
        mean_ptpk = np.mean(pkamps(hr_accept_list))
        ptpk_variation = np.std(pkamps(hr_accept_list))
        hr_accept_list = hr_accept_list & (np.abs(pkamps-mean_ptpk) < 3*ptpk_variation)
    
    return hr_accept_list


def prepeak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs):
    # Somewhat working immediately preceding peak matching algo.

    t_first_ecg_pk = time_ecg[ecg_peak_locs[0]]
    idx_first_pt_after_ecg = np.nonzero(time_pt[pt_cardiac_peak_locs] > t_first_ecg_pk)[0][0]
    pt_peaks_selected = pt_cardiac_peak_locs[idx_first_pt_after_ecg:]
    ecg_peaks_selected = ecg_peak_locs[:len(pt_peaks_selected)]

    # # Reject bad beats
    # ptPeaksSelected = ptPeaksSelected(hrAcceptList(idx_first_pt_after_ecg:end))
    # ecgPeaksSelected = ecgPeaksSelected(hrAcceptList(idx_first_pt_after_ecg:end))

    peak_diff = time_pt[pt_peaks_selected] - time_ecg[ecg_peaks_selected]
    return peak_diff, pt_peaks_selected

def interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs):

    # Iterate over ECG peaks, and check if there is a PT peak between current and next ECG peak.
    pt_trig_wf = np.zeros(time_pt.shape, dtype=int)
    pt_trig_wf[pt_cardiac_peak_locs] = 1
    peak_diff = []
    extra_pk_idx = []
    miss_pk_idx = []

    n_ecg_pk = ecg_peak_locs.shape[0]
    for pk_i in range(n_ecg_pk-1):
        curr_pk_t = time_ecg[ecg_peak_locs[pk_i]]
        next_pk_t = time_ecg[ecg_peak_locs[pk_i+1]]

        masked_pt_trig = pt_trig_wf & ((time_pt > curr_pk_t) & (time_pt < next_pk_t))

        n_trig_per_trig = np.sum(masked_pt_trig)
        if n_trig_per_trig == 0:
            # Missed beat, nothing to do, move on.
            miss_pk_idx.append(pk_i)
            continue
        elif n_trig_per_trig > 1:
            # Extraneous trigger, not cool. Mark, but still accept the first one.
            extra_pk_idx.append(pk_i)
        
        pt_peak_idx = np.nonzero(masked_pt_trig)[0][0]
        peak_diff.append(time_pt[pt_peak_idx] - curr_pk_t)

    return np.asarray(peak_diff), np.asarray(miss_pk_idx), np.asarray(extra_pk_idx)

def extract_triggers(time_pt, cardiac_waveform, skip_time=0.6, prominence=0.4, max_hr=120):
    ''' Extract triggers from the cardiac waveform.
        Parameters:
            time_pt: np.array
                Time points for the cardiac waveform.
            cardiac_waveform: np.array
                Cardiac waveform.
            skip_time: float
                Time to skip at the beginning of the waveform.
        Returns:
            pt_cardiac_trigs: np.array
                Trigger waveform.
    '''
    dt_pt = (time_pt[1] - time_pt[0])
    Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
    pt_cardiac_peak_locs,_ = find_peaks(cardiac_waveform[time_pt > skip_time], prominence=prominence, distance=Dmin)
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)
    pt_peaks_selected = pt_cardiac_peak_locs
    n_acq = time_pt.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1

    return pt_cardiac_trigs

def pt_ecg_jitter(time_pt, pt_cardiac, pt_cardiac_derivative, time_ecg, ecg_waveform, pt_cardiac_trigs=None, pt_derivative_trigs=None, ecg_trigs=None, skip_time=0.6, max_hr=120, show_outputs=True): 
    """ 
    This function calculates the jitter between the pilot tone and the ECG triggers.
    The ECG triggers are assumed to be correct. The PT triggers are assumed to be correct.
    The peak locations of the PT and ECG are found and matched. The jitter is calculated from the differences between the matched peaks.

    Parameters
    ----------
    time_pt : numpy array
        Time axis of the PT waveform in seconds.
    pt_cardiac : numpy array
        PT waveform.
    pt_cardiac_derivative : numpy array
        Derivative of the PT waveform.
    time_ecg : numpy array
        Time axis of the ECG waveform in seconds.
    ecg_waveform : numpy array
        ECG waveform.
    pt_cardiac_trigs : numpy array, optional
        PT triggers. If None, they are calculated.
    pt_derivative_trigs : numpy array, optional
        Derivative PT triggers. If None, they are calculated.
    ecg_trigs : numpy array, optional
        ECG triggers. If None, they are calculated.
    skip_time : float, optional
        Time to skip at the beginning of the waveforms in seconds. The default is 0.6.
    max_hr : int, optional
        Maximum heart rate in bpm. The default is 120.
    show_outputs : bool, optional
        Whether to show the outputs. The default is True.

    Returns
    -------
    peak_diff : numpy array
        Differences between the matched peaks.
    derivative_peak_diff : numpy array
        Differences between the matched derivative peaks.

    """

    # ECG Triggers
    if ecg_trigs is None:
        ecg_peak_locs,_ = find_peaks(ecg_waveform[time_ecg > skip_time], prominence=0.7)
    else:
        ecg_peak_locs = np.nonzero(ecg_trigs[time_ecg > skip_time])[0]
    ecg_peak_locs += np.sum(time_ecg <= skip_time)

    # PT Triggers
    dt_pt = (time_pt[1] - time_pt[0])
    
    if pt_cardiac_trigs is None:
        Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
        pt_cardiac_peak_locs,_ = find_peaks(pt_cardiac[time_pt > skip_time], prominence=0.4, distance=Dmin)
    else:
        pt_cardiac_peak_locs = np.nonzero(pt_cardiac_trigs[time_pt > skip_time])[0]
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)

    # PT Derivative Triggers
    if pt_derivative_trigs is None:
        pt_cardiac_derivative_peak_locs,_ = find_peaks(pt_cardiac_derivative[time_pt > skip_time], prominence=0.6, distance=Dmin)
    else:
        pt_cardiac_derivative_peak_locs = np.nonzero(pt_derivative_trigs[time_pt > skip_time])[0]
    pt_cardiac_derivative_peak_locs += np.sum(time_pt <= skip_time)

    # "Arryhtmia detection" by heart rate variation
    hr_accept_list = beat_rejection(pt_cardiac_peak_locs*dt_pt, "post")
    hr_accept_list_derivative = beat_rejection(pt_cardiac_derivative_peak_locs*dt_pt, "pre")
    # TODO: Is pre post even correct? Why does it change? Need to investigate.

    # peak_diff, pt_peaks_selected = prepeak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    # derivative_peak_diff, pt_derivative_peaks_selected = prepeak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)

    peak_diff, miss_pks, extra_pks = interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    pt_peaks_selected = pt_cardiac_peak_locs

    derivative_peak_diff, derivative_miss_pks, derivative_extra_pks = interval_peak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)
    pt_derivative_peaks_selected = pt_cardiac_derivative_peak_locs

    # Create trigger waveforms from peak locations.
    n_acq = pt_cardiac.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_derivative_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1
    pt_derivative_trigs[pt_derivative_peaks_selected] = 1

    if show_outputs:
        # Print some useful info

        print(f'Rejection ratio for pt peaks is {100*(len(hr_accept_list) - np.sum(hr_accept_list))/len(hr_accept_list):.2f} percent.\n')
        print(f'Rejection ratio for derivative pt peaks is {100*(len(hr_accept_list_derivative) - np.sum(hr_accept_list_derivative))/len(hr_accept_list_derivative):.2f} percent.\n')

        print(f'Peak difference {np.mean(peak_diff*1e3):.1f} \u00B1 {np.std(peak_diff*1e3):.1f}')
        print(f'Derivative peak difference {np.mean(derivative_peak_diff*1e3):.1f} \u00B1 {np.std(derivative_peak_diff*1e3):.1f}')

        print(f'Number of ECG triggers: {ecg_peak_locs.shape[0]}.')
        print(f'Number of PT triggers: {pt_cardiac_peak_locs.shape[0]}.')
        print(f'Number of missed PT triggers: {miss_pks.shape[0]}.')
        print(f'Number of extraneous PT triggers: {extra_pks.shape[0]}.')
        print(f'Number of derivative PT triggers: {pt_cardiac_derivative_peak_locs.shape[0]}.')
        print(f'Number of missed derivative PT triggers: {derivative_miss_pks.shape[0]}.')
        print(f'Number of extraneous derivative PT triggers: {derivative_extra_pks.shape[0]}.')

        import matplotlib.pyplot as plt
        # Plots
        plt.figure()
        plt.plot(time_ecg, ecg_waveform)
        plt.plot(time_ecg[ecg_trigs==1], ecg_waveform[ecg_trigs==1], '*')
        plt.plot(time_pt, pt_cardiac_trigs, 'x', label='PT Triggers')

        f, axs = plt.subplots(2,2, sharex='col')
        axs[0,0].plot(time_pt, pt_cardiac, '-gD', markevery=pt_cardiac_peak_locs, label='Pilot Tone')
        axs[0,0].plot(time_ecg, ecg_waveform, '-bs', markevery=ecg_peak_locs, label='ECG')
        axs[0,0].set_xlabel('Time [s]')
        axs[0,0].legend()
        axs[0,0].set_title('ECG and Pilot Tone. Markers show triggers.')

        axs[0,1].hist((peak_diff - np.mean(peak_diff))*1e3)
        axs[0,1].set_xlabel('Time diff [ms]')
        axs[0,1].set_ylabel('Number of peaks')

        axs[1,0].plot(time_pt, pt_cardiac_derivative, '-gD', markevery=pt_cardiac_derivative_peak_locs, label='Pilot Tone')
        axs[1,0].plot(time_ecg, ecg_waveform, '-bs', markevery=ecg_peak_locs, label='ECG')
        axs[1,0].set_xlabel('Time [s]')
        axs[1,0].legend()
        axs[1,0].set_title('ECG and Inverse Derivative Pilot Tone. Markers show triggers.')

        axs[1,1].hist((derivative_peak_diff - np.mean(derivative_peak_diff))*1e3)
        axs[1,1].set_xlabel('Time diff [ms]')
        axs[1,1].set_ylabel('Number of peaks')

        plt.show()
    
    return peak_diff, derivative_peak_diff

def calculate_jitter(time_pt, pt_cardiac, time_ecg, ecg_waveform, pt_cardiac_trigs=None, ecg_trigs=None, skip_time=0.6, peak_prominence=0.4, max_hr=120): 
    """ 
    This function calculates the jitter between the pilot tone and the ECG triggers.
    The ECG triggers are assumed to be correct. The PT triggers are assumed to be correct.
    The peak locations of the PT and ECG are found and matched. The jitter is calculated from the differences between the matched peaks.

    Parameters
    ----------
    time_pt : numpy array
        Time axis of the PT waveform in seconds.
    pt_cardiac : numpy array
        PT waveform.
    time_ecg : numpy array
        Time axis of the ECG waveform in seconds.
    ecg_waveform : numpy array
        ECG waveform.
    pt_cardiac_trigs : numpy array, optional
        PT triggers. If None, they are calculated.
    ecg_trigs : numpy array, optional
        ECG triggers. If None, they are calculated.
    skip_time : float, optional
        Time to skip at the beginning of the waveforms in seconds. The default is 0.6.
    peak_prominence : float, optional
        Prominence of the pilot tone peaks. The default is 0.4.
    max_hr : int, optional
        Maximum heart rate in bpm. The default is 120.

    Returns
    -------
    peak_diff : numpy array
        Differences between the matched peaks.
    miss_pks : numpy array
        False negative PT triggers.
    extra_pks : numpy array
        False positive PT triggers.
    """

    # ECG Triggers
    if ecg_trigs is None:
        ecg_peak_locs,_ = find_peaks(ecg_waveform[time_ecg > skip_time], prominence=0.7)
    else:
        ecg_peak_locs = np.nonzero(ecg_trigs[time_ecg > skip_time])[0]
    ecg_peak_locs += np.sum(time_ecg <= skip_time)

    # PT Triggers
    dt_pt = (time_pt[1] - time_pt[0])
    
    if pt_cardiac_trigs is None:
        Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
        pt_cardiac_peak_locs,_ = find_peaks(pt_cardiac[time_pt > skip_time], prominence=peak_prominence, distance=Dmin)
    else:
        pt_cardiac_peak_locs = np.nonzero(pt_cardiac_trigs[time_pt > skip_time])[0]
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)

    peak_diff, miss_pks, extra_pks = interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    pt_peaks_selected = pt_cardiac_peak_locs

    # Create trigger waveforms from peak locations.
    n_acq = pt_cardiac.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1
    
    return peak_diff, miss_pks, extra_pks

def report_jitter(time_pt, pt_cardiac, time_ecg, ecg_waveform, ecg_trigs):
    pt_cardiac[:20] = pt_cardiac[20]
    pt_cardiac[-20:] = pt_cardiac[-20]
    pt_cardiac -= np.percentile(pt_cardiac, 10)
    pt_cardiac /= np.percentile(pt_cardiac, 95)

    # pt_cardiac_filtered = savgol_filter(pt_cardiac, sg_filter_len, 3, axis=0)
    pt_cardiac_filtered = pt_cardiac.copy()

    pt_cardiac_derivative = np.hstack((0, np.diff(pt_cardiac_filtered)/(time_pt[1] - time_pt[0])))
    pt_cardiac_derivative[:20] = pt_cardiac_derivative[20]
    pt_cardiac_derivative[-20:] = pt_cardiac_derivative[-20]
    pt_cardiac_derivative -= np.percentile(pt_cardiac_derivative, 10)
    pt_cardiac_derivative /= np.percentile(pt_cardiac_derivative, 98)

    skip_time = 1 # seconds, to skip initial part of the signal
    max_hr = 160 # bpm, maximum expected heart rate to set minimum peak distance

    pt_cardiac_trigs = extract_triggers(time_pt, pt_cardiac, skip_time=skip_time, prominence=0.4, max_hr=max_hr)
    pt_derivative_trigs = extract_triggers(time_pt, pt_cardiac_derivative, skip_time=skip_time, prominence=0.5, max_hr=max_hr)

    # ECG Triggers
    if ecg_trigs is None:
        ecg_peak_locs,_ = find_peaks(ecg_waveform[time_ecg > skip_time], prominence=0.7)
    else:
        ecg_peak_locs = np.nonzero(ecg_trigs[time_ecg > skip_time])[0]
    ecg_peak_locs += np.sum(time_ecg <= skip_time)

    # PT Triggers
    dt_pt = (time_pt[1] - time_pt[0])
    
    if pt_cardiac_trigs is None:
        Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
        pt_cardiac_peak_locs,_ = find_peaks(pt_cardiac[time_pt > skip_time], prominence=0.4, distance=Dmin)
    else:
        pt_cardiac_peak_locs = np.nonzero(pt_cardiac_trigs[time_pt > skip_time])[0]
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)

    # PT Derivative Triggers
    if pt_derivative_trigs is None:
        pt_cardiac_derivative_peak_locs,_ = find_peaks(pt_cardiac_derivative[time_pt > skip_time], prominence=0.6, distance=Dmin)
    else:
        pt_cardiac_derivative_peak_locs = np.nonzero(pt_derivative_trigs[time_pt > skip_time])[0]
    pt_cardiac_derivative_peak_locs += np.sum(time_pt <= skip_time)

    # peak_diff, pt_peaks_selected = prepeak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    # derivative_peak_diff, pt_derivative_peaks_selected = prepeak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)

    peak_diff, miss_pks, extra_pks = interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    pt_peaks_selected = pt_cardiac_peak_locs

    derivative_peak_diff, derivative_miss_pks, derivative_extra_pks = interval_peak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)
    pt_derivative_peaks_selected = pt_cardiac_derivative_peak_locs

    # Create trigger waveforms from peak locations.
    n_acq = pt_cardiac.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_derivative_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1
    pt_derivative_trigs[pt_derivative_peaks_selected] = 1

    report_str = f"""ECG vs Pilot Tone Cardiac Trigger Comparison:
-------------------------------------------------------------
Mean peak difference: {np.mean(peak_diff*1e3):.1f} +- {np.std(peak_diff*1e3):.1f} ms
Mean derivative peak difference: {np.mean(derivative_peak_diff*1e3):.1f} +- {np.std(derivative_peak_diff*1e3):.1f} ms
Number of ECG triggers: {ecg_peak_locs.shape[0]}
Number of PT triggers: {pt_cardiac_peak_locs.shape[0]}
Number of missed PT triggers: {miss_pks.shape[0]}
Number of extraneous PT triggers: {extra_pks.shape[0]}
Number of derivative PT triggers: {pt_cardiac_derivative_peak_locs.shape[0]}
Number of missed derivative PT triggers: {derivative_miss_pks.shape[0]}
Number of extraneous derivative PT triggers: {derivative_extra_pks.shape[0]}
"""
    return report_str
