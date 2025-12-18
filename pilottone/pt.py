import ismrmrd
from numpy.fft import ifft, fft
from scipy.signal.windows import tukey
from scipy.linalg import lstsq
from scipy.signal import find_peaks, peak_widths
from scipy.sparse.linalg import svds
from .signal import apply_filter_freq, designbp_tukeyfilt_freq, designlp_tukeyfilt_freq, qint
from sobi import sobi
import numpy as np
import numpy.typing as npt
import pyfftw
import math
import matplotlib.pyplot as plt
import re

def calc_fovshift_phase(kx: npt.NDArray, ky: npt.NDArray, acq: ismrmrd.Acquisition) -> npt.NDArray[np.complex64]:
    '''Calculate the phase demodulation due to the FOV shift in the GCS.

    Parameters:
    ----------
    kx (np.ndarray): 
        1D array of k-space points in the logical x coordinates.
    ky (np.ndarray): 
        2D array of k-space points in the logical y coordinates.
    acq (ismrmrd.Acquisition): 
        Acquisition object containing the phase and read directions, and position.

    Returns:
    ----------
    np.ndarray: 
        1D array of phase demodulation values in the GCS.
    '''

    gbar = 42.576e6

    gx = np.diff(np.concatenate((np.zeros((1, kx.shape[1]), dtype=kx.dtype), kx)), axis=0)/gbar # [T/m]
    gy = np.diff(np.concatenate((np.zeros((1, kx.shape[1]), dtype=kx.dtype), ky)), axis=0)/gbar # [T/m]
    g_nom = np.stack((gx, gy), axis=2)
    g_gcs = np.concatenate((g_nom, np.zeros((g_nom.shape[0], g_nom.shape[1], 1))), axis=2)

    r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                            [1,    0,   0],  # [RO] = [1 0 0] * [c]
                            [0,    0,   1]]) # [SL]   [0 0 1] * [s]

    r_GCS2PCS = np.array([np.array(acq.phase_dir), 
                        np.array(acq.read_dir), 
                        np.array(acq.slice_dir)])
    PCS_offset = np.array([1, 1, 1])*np.array(acq.position)*1e-3
    GCS_offset = r_GCS2PCS.dot(PCS_offset)
    # RCS_offset = r_GCS2RCS.dot(GCS_offset)
    g_rcs = np.dot(r_GCS2RCS, np.transpose(g_gcs, (1,2,0))).transpose((2,1,0))
    phase_mod_rads = np.exp(-1j*np.cumsum(2*np.pi*gbar*np.sum(g_rcs*GCS_offset, axis=2), axis=0)) # [rad]

    return phase_mod_rads.astype(np.complex64)

def est_dtft(t, data, deltaf: npt.NDArray, window: npt.NDArray | None = None):
    ''' est_dtft MMSE sine amplitude estimate by DTFT sum given freq deltaf.
    Also subtracts the estimated windowed sine from the data.
     Inputs:
       t:      Time axis along readout [s].
       data:   (Nx x Nline x Nch) Time data to estimate peak amplitudes. 
       deltaf: Frequency where the peak occurs [Hz].
       window: Window function to multiply sine model before subtraction.
     Outputs:
       clean:  (Nx x Nline x Nch) Model subtracted data
       x_fit:  (Nline x Nch) Estimated complex amplitudes.'''

    Nsamp = data.shape[0]
    dt = t[1]-t[0] # [s]
    w0 = 2*deltaf*dt*np.pi # Normalized frequency [-pi, +pi]
    
    x_fit = (np.sum(data*np.exp(+1j*w0[None,:,None]*np.arange(0, Nsamp)[:,None,None]),axis=0, keepdims=True)/Nsamp)

    if window is None:
        s = (np.exp(-1j*2*np.pi*(deltaf[None,:])*t[:,None]))[:,:,None]
    else:
        s = (np.exp(-1j*2*np.pi*(deltaf[None,:])*t[:,None])*window[:,None])[:,:,None]
        
    clean = data - s*x_fit

    return (clean, x_fit)


def find_freq_qifft(data, df, f_center, f_radius, os, ave_dim):
    Nsamp = data.shape[0]
    dfint = df / os
    data = pyfftw.byte_align(data.transpose(2,1,0))

    # start_time = time.time()
    fft = pyfftw.builders.ifft(data, n=Nsamp*os, axis=2, threads=64, planner_effort='FFTW_ESTIMATE')
    data_f = np.fft.ifftshift(fft(), axes=2)
    data_f = data_f.transpose((2, 1 ,0))
    # end_time = time.time()
    # print(end_time-start_time)
    # data_f = np.fft.ifftshift(pyfftw.interfaces.numpy_fft.ifft(data, Nsamp*os, axis=0), axes=0)
    # data_f = np.fft.ifftshift(np.fft.ifft(data, Nsamp * os, axis=0), axes=0)
    data_f = np.abs(data_f)

    if ave_dim is not None:
        data_fpk = np.mean(data_f, axis=ave_dim)

    f_axis = np.arange(-Nsamp * os / 2, Nsamp * os / 2) * dfint

    f_search_interval = (f_axis < (f_center + f_radius / 2)) & (f_axis > (f_center - f_radius / 2))

    if np.sum(f_search_interval) == 0:
        raise ValueError('Search frequency is outside of the imaging bandwidth. Check PT frequency.')

    data_fpk_srch = data_fpk[f_search_interval]
    f_axis_fpk_srch = f_axis[f_search_interval]

    Iinit = np.argmax(data_fpk_srch, axis=0)

    if np.any((Iinit == 0) | (Iinit == len(data_fpk_srch) - 1)):
        print(f'Peak is found at the edge index of {Iinit}.\nThis may mean the peak is outside of the given frequency range or there is no peak at all. Returning 0.')
        return 0

    finit = f_axis_fpk_srch[Iinit]
    if data_fpk_srch.ndim > 1: 
        Ipr = np.ravel_multi_index((Iinit-1, np.arange(data_fpk_srch.shape[1])), data_fpk_srch.shape)
        Icr = np.ravel_multi_index((Iinit, np.arange(data_fpk_srch.shape[1])), data_fpk_srch.shape)
        Inx = np.ravel_multi_index((Iinit+1, np.arange(data_fpk_srch.shape[1])), data_fpk_srch.shape)
    else:
        Ipr = Iinit-1
        Icr = Iinit
        Inx = Iinit+1

    p, _, _ = qint(data_fpk_srch.ravel()[Ipr], data_fpk_srch.ravel()[Icr], data_fpk_srch.ravel()[Inx])

    f_found = finit + p * dfint
    fcorrmin = f_center - f_found

    return fcorrmin

def sniffer_sub(b: npt.NDArray, A: npt.NDArray):
    Npe = A.shape[0]
    filt = np.hstack((0, tukey(Npe-1, 0.6)))[:,None]
    
    A_f = np.real(ifft(fft(A)*filt/Npe))
    b_f = np.real(ifft(fft(b)*filt/Npe))
    # x = A_f\b_f LSQ
    x,_,_,_ = lstsq(A_f, b_f)
    clean = b - A.dot(x)

    return clean - np.mean(clean)

def plot_multich_comparison(tt: npt.NDArray[np.float64], sigs: tuple[npt.NDArray[np.float64], ...], 
                            titles: npt.ArrayLike, labels: tuple[str, ...]):
    # Plotting fcn
    n_ch = sigs[0].shape[1]
    n_sigs = len(sigs)
    if n_ch < 3:
        nc = 1
    else:
        nc = 2
    nr = math.ceil(n_ch/nc)
    ff, axs = plt.subplots(nr, nc, sharex=True)
    for ii in range(n_ch):
        xi = np.unravel_index(ii, (nr, nc))
        ax_ = axs[xi[0], xi[1]]

        for si in range(n_sigs):
            ax_.plot(tt, sigs[si][:,ii], label=labels[si])

        ax_.set_title(titles[ii])
        ax_.set_xlabel('Time [s]')

        if ii == 0:
            ax_.legend()

    # If number of chs is odd, last axes is empty, so remove
    if n_ch%2 == 1:
        axs[-1, -1].remove()


def pickcoilsbycorr(insig, start_ch, corr_th):
    Nch = insig.shape[1]
    C = np.corrcoef(insig, rowvar=False)

    # Automatic start_ch selector
    if start_ch == -1:
        C_ = np.copy(C)
        C_[np.abs(C_) < corr_th] = 0
        s = np.sum(np.abs(C_), axis=0)-1.0
        start_ch = np.argmax(s)

    accept_list = [start_ch]
    sign_list = [1]
    corrs = [1]

    for ii in range(Nch):
        if ii > start_ch:
            if abs(C[start_ch, ii]) > corr_th:
                accept_list.append(ii)
                sign_list.append(np.sign(C[start_ch, ii]))
                corrs.append(abs(C[start_ch, ii]))
        elif ii < start_ch:
            if abs(C[ii, start_ch]) > corr_th:
                accept_list.append(ii)
                sign_list.append(np.sign(C[ii, start_ch]))
                corrs.append(abs(C[ii, start_ch]))

    return accept_list, sign_list, corrs

def check_waveform_polarity(waveform: npt.NDArray[np.float64], prominence: float=0.5) -> int:
    '''Check the polarity of the waveform and return the sign.
    The logic is, peaks looking up should be narrower than the bottom side for better triggering.
    
    Parameters:
    ----------
    waveform (np.array): Waveform to check.
    prominence (float): Prominence threshold for peak detection.

    Returns:
    ----------
    wf_sign (int): Sign of the waveform. 1 for positive, -1 for negative.
    '''
    waveform_ = waveform.copy()
    waveform_ -= np.percentile(waveform_, 5)
    waveform_ = waveform_/np.percentile(waveform_, 99)
    p1, d1 = find_peaks(waveform_, prominence=prominence)
    w1,_,_,_ = peak_widths(waveform_, p1)

    waveform_ = -waveform_
    waveform_ -= np.percentile(waveform_, 5)
    waveform_ = waveform_/np.percentile(waveform_, 99)

    p2, d2 = find_peaks(-waveform, prominence=prominence)
    w2,_,_,_ = peak_widths(-waveform, p2)

    wf_sign = 1
    if np.sum(w1) > np.sum(w2):
        print('Cardiac waveform looks flipped. Flipping it..')
        wf_sign = -1

    return wf_sign

def extract_pilottone_navs(pt_sig, f_samp: float, params: dict):
    '''Extract the respiratory and cardiac pilot tone signals from the given PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    f_samp (float): Sampling frequency of the PT signal.
    params (dict): Dictionary containing the parameters for the extraction.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''
    n_pt_samp = pt_sig.shape[0]
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 

    from scipy.signal import savgol_filter
    
    pt_denoised = savgol_filter(pt_sig, params['golay_filter_len'], 3, axis=0)
    pt_denoised = pt_denoised - np.mean(pt_denoised, axis=0)

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_sig, pt_denoised, [' ']*n_ch, ['Original', 'SG filtered'])


    # ================================================================
    # Filter out higher than resp frequency ~1 Hz
    # ================================================================ 
    # df = f_samp/n_pt_samp/2
    # f_filt = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2 # Handles both even and odd length signals.

    if params['respiratory']['freq_start'] is None:
        filt_bp_resp = designlp_tukeyfilt_freq(params['respiratory']['freq_stop'], f_samp, n_pt_samp)
    else:
        filt_bp_resp = designbp_tukeyfilt_freq(params['respiratory']['freq_start'], params['respiratory']['freq_stop'], f_samp, n_pt_samp)

    pt_respiratory_freqs = apply_filter_freq(pt_denoised, filt_bp_resp, 'symmetric')

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_denoised, pt_respiratory_freqs, [' ']*n_ch, ['Original', 'respiratory filtered'])

    
    # ================================================================
    # Reject channels that have low correlation
    # ================================================================
    (accept_list, sign_list, corrs) = pickcoilsbycorr(pt_respiratory_freqs, params['respiratory']['corr_init_ch'], params['respiratory']['corr_threshold'])
    accept_list = np.sort(accept_list)
    print(f'Number of channels selected for respiratory PT: {len(accept_list)}')

    if params['respiratory']['separation_method'] == 'pca':
        # ================================================================
        # Apply PCA along coils to extract common signal (hopefuly resp)
        # ================================================================ 
        U, S, _ = svds(pt_respiratory_freqs[:,accept_list], k=1)

        # ================================================================
        # Separate a single respiratory source
        # ================================================================
        pt_respiratory = U*S
        pt_respiratory = pt_respiratory[:,0]

    elif params['respiratory']['separation_method'] == 'sobi':
        pt_respiratory, _, _ = sobi(pt_respiratory_freqs[:,accept_list].T)
        pt_respiratory = pt_respiratory[0,:]

    filt_bp_cardiac = designbp_tukeyfilt_freq(params['cardiac']['freq_start'], params['cardiac']['freq_stop'], f_samp, n_pt_samp)

    pt_cardiac_freqs = apply_filter_freq(pt_denoised, filt_bp_cardiac, 'symmetric')

    # Separate a single cardiac source
    # Correlation based channel selection
    # This is a semi automated fix for the case when a variety of SNR is
    # provided, corr_th needs to be adjusted. So, we start from high corr, and
    # loop until we have at least 2 channels with cardiac. My observation is,
    # if we can't find at least 2 channels, signal is too noisy to use anyways,
    # so we fail to extract cardiac PT.
    corr_threshold_cardiac = params['cardiac']['corr_threshold']
    while corr_threshold_cardiac >= 0.5:
        [accept_list_cardiac, signList, corrChannels] = pickcoilsbycorr(pt_cardiac_freqs, params['cardiac']['corr_init_ch'], corr_threshold_cardiac)
        if len(accept_list_cardiac) < 2:
            corr_threshold_cardiac -= 0.05
        else:
            break


    if len(accept_list_cardiac) == 1:
        print('Could not find more channels with cardiac PT. Extraction is possibly failed.')

    print(f'Number of channels selected for cardiac PT: {len(accept_list_cardiac)}')
    if params['cardiac']['separation_method'] == 'pca':
        U, S, _ = svds(pt_cardiac_freqs[:,accept_list_cardiac], k=1)
        pt_cardiac = U*S
        pt_cardiac = pt_cardiac[:,0]
    elif params['cardiac']['separation_method'] == 'sobi':
        pt_cardiac, _, _ = sobi(pt_cardiac_freqs[:,accept_list_cardiac].T)
        pt_cardiac = pt_cardiac[0,:]

    # Normalize navs before returning.
    # Here, I am using prctile instead of the max to avoid weird spikes.
    if not params['debug']['no_normalize']:
        pt_respiratory -= np.percentile(pt_respiratory, 5)
        pt_respiratory /= np.percentile(pt_respiratory, 99)

        # Check if the waveform is flipped and flip if necessary.
        # Logic is, peaks looking up should be narrower than the bottom side for better triggering.
        ptc_sign = check_waveform_polarity(pt_cardiac[40:], prominence=0.5)
        pt_cardiac = ptc_sign*pt_cardiac
        
        # Shift the base and normalize again to make it mostly 0 to 1
        pt_cardiac -= np.percentile(pt_cardiac, 5)
        pt_cardiac = pt_cardiac/np.percentile(pt_cardiac, 99)

    return pt_respiratory, pt_cardiac

def calibrate_pt(pt_sig, f_samp: float, params: dict):
    '''Extract the respiratory and cardiac pilot tone signals from the given PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    f_samp (float): Sampling frequency of the PT signal.
    params (dict): Dictionary containing the parameters for the extraction.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''
    n_pt_samp = pt_sig.shape[0]
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 

    from scipy.signal import savgol_filter
    
    pt_denoised = savgol_filter(pt_sig, params['golay_filter_len'], 3, axis=0)
    pt_denoised = pt_denoised - np.mean(pt_denoised, axis=0)

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_sig, pt_denoised, [' ']*n_ch, ['Original', 'SG filtered'])


    # ================================================================
    # Filter out higher than resp frequency ~1 Hz
    # ================================================================ 
    # df = f_samp/n_pt_samp/2
    # f_filt = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2 # Handles both even and odd length signals.

    if params['respiratory']['freq_start'] is None:
        filt_bp_resp = designlp_tukeyfilt_freq(params['respiratory']['freq_stop'], f_samp, n_pt_samp)
    else:
        filt_bp_resp = designbp_tukeyfilt_freq(params['respiratory']['freq_start'], params['respiratory']['freq_stop'], f_samp, n_pt_samp)

    pt_respiratory_freqs = apply_filter_freq(pt_denoised, filt_bp_resp, 'symmetric')

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_denoised, pt_respiratory_freqs, [' ']*n_ch, ['Original', 'respiratory filtered'])

    
    # ================================================================
    # Reject channels that have low correlation
    # ================================================================
    (accept_list_resp, sign_list, corrs) = pickcoilsbycorr(pt_respiratory_freqs, params['respiratory']['corr_init_ch'], params['respiratory']['corr_threshold'])
    accept_list_resp = np.sort(accept_list_resp)
    print(f'Number of channels selected for respiratory PT: {len(accept_list_resp)}')

    if params['respiratory']['separation_method'] == 'pca':
        # ================================================================
        # Apply PCA along coils to extract common signal (hopefuly resp)
        # ================================================================ 
        Uresp, S, Vresp = svds(pt_respiratory_freqs[:,accept_list_resp], k=1)

        # ================================================================
        # Separate a single respiratory source
        # ================================================================
        pt_respiratory = Uresp
        pt_respiratory = pt_respiratory[:,0]

    elif params['respiratory']['separation_method'] == 'sobi':
        pt_respiratory, _, Vresp = sobi(pt_respiratory_freqs[:,accept_list_resp].T)
        pt_respiratory = pt_respiratory[0,:]

    filt_bp_cardiac = designbp_tukeyfilt_freq(params['cardiac']['freq_start'], params['cardiac']['freq_stop'], f_samp, n_pt_samp)

    pt_cardiac_freqs = apply_filter_freq(pt_denoised, filt_bp_cardiac, 'symmetric')

    # Separate a single cardiac source
    # Correlation based channel selection
    # This is a semi automated fix for the case when a variety of SNR is
    # provided, corr_th needs to be adjusted. So, we start from high corr, and
    # loop until we have at least 2 channels with cardiac. My observation is,
    # if we can't find at least 2 channels, signal is too noisy to use anyways,
    # so we fail to extract cardiac PT.
    corr_threshold_cardiac = params['cardiac']['corr_threshold']
    while corr_threshold_cardiac >= 0.5:
        [accept_list_cardiac, signList, corrChannels] = pickcoilsbycorr(pt_cardiac_freqs, params['cardiac']['corr_init_ch'], corr_threshold_cardiac)
        if len(accept_list_cardiac) < 2:
            corr_threshold_cardiac -= 0.05
        else:
            break


    if len(accept_list_cardiac) == 1:
        print('Could not find more channels with cardiac PT. Extraction is possibly failed.')

    print(f'Number of channels selected for cardiac PT: {len(accept_list_cardiac)}')
    if params['cardiac']['separation_method'] == 'pca':
        Ucard, S, Vcard = svds(pt_cardiac_freqs[:,accept_list_cardiac], k=1)
        pt_cardiac = Ucard
        pt_cardiac = pt_cardiac[:,0]
    elif params['cardiac']['separation_method'] == 'sobi':
        pt_cardiac, _, Vcard = sobi(pt_cardiac_freqs[:,accept_list_cardiac].T)
        pt_cardiac = pt_cardiac[0,:]

    # Normalize navs before returning.
    # Here, I am using prctile instead of the max to avoid weird spikes.
    if not params['debug']['no_normalize']:
        pt_respiratory -= np.percentile(pt_respiratory, 5)
        pt_respiratory /= np.percentile(pt_respiratory, 99)

        # Check if the waveform is flipped and flip if necessary.
        # Logic is, peaks looking up should be narrower than the bottom side for better triggering.
        ptc_sign = check_waveform_polarity(pt_cardiac[40:], prominence=0.5)
        pt_cardiac = ptc_sign*pt_cardiac
        
        # Shift the base and normalize again to make it mostly 0 to 1
        pt_cardiac -= np.percentile(pt_cardiac, 5)
        pt_cardiac = pt_cardiac/np.percentile(pt_cardiac, 99)

    return Vresp, accept_list_resp, pt_respiratory, Vcard, accept_list_cardiac, pt_cardiac

def apply_pt_calib(pt_sig, Vresp, accept_list_resp, Vcard, accept_list_cardiac, f_samp, params):
    '''Apply the calibration matrices to the PT signal.
    Parameters:
    ----------
    pt_sig (np.array): Pilot tone signal.
    Uresp (np.array): Respiratory calibration matrix.
    accept_list_resp (list): List of channels used for respiratory calibration.
    Ucard (np.array): Cardiac calibration matrix.
    accept_list_cardiac (list): List of channels used for cardiac calibration.

    Returns:
    ----------
    pt_respiratory (np.array): Extracted respiratory pilot tone signal.
    pt_cardiac (np.array): Extracted cardiac pilot tone signal.
    '''

    n_pt_samp = pt_sig.shape[0]
    n_ch = pt_sig.shape[1]
    dt_pt = 1/f_samp
    time_pt = np.arange(n_pt_samp)*dt_pt
    
    # ================================================================
    # Denoising step
    # ================================================================ 

    from scipy.signal import savgol_filter
    
    pt_denoised = savgol_filter(pt_sig, params['golay_filter_len'], 3, axis=0)
    pt_denoised = pt_denoised - np.mean(pt_denoised, axis=0)

    if params['debug']['show_plots'] is True:
        plot_multich_comparison(time_pt, pt_sig, pt_denoised, [' ']*n_ch, ['Original', 'SG filtered'])


    # ================================================================
    # Filter out higher than resp frequency ~1 Hz
    # ================================================================ 
    # df = f_samp/n_pt_samp/2
    # f_filt = np.arange(0, f_samp, df) - (f_samp - (n_pt_samp % 2)*df)/2 # Handles both even and odd length signals.

    if params['respiratory']['freq_start'] is None:
        filt_bp_resp = designlp_tukeyfilt_freq(params['respiratory']['freq_stop'], f_samp, n_pt_samp)
    else:
        filt_bp_resp = designbp_tukeyfilt_freq(params['respiratory']['freq_start'], params['respiratory']['freq_stop'], f_samp, n_pt_samp)

    pt_respiratory_freqs = apply_filter_freq(pt_denoised, filt_bp_resp, 'symmetric')

    pt_respiratory = pt_respiratory_freqs[:, accept_list_resp]@Vresp[:,0]

    filt_bp_cardiac = designbp_tukeyfilt_freq(params['cardiac']['freq_start'], params['cardiac']['freq_stop'], f_samp, n_pt_samp)
    pt_cardiac_freqs = apply_filter_freq(pt_denoised, filt_bp_cardiac, 'symmetric')

    pt_cardiac = pt_cardiac_freqs[:, accept_list_cardiac]@Vcard[:,0]


    return pt_respiratory, pt_cardiac

def get_volt_from_protoname(proto_name: str) -> float:
    """
    Extract the PT volt if written in the protocol name as _XXXV_ or _XXXmV_.
    
    Parameters:
    proto_name (str): Protocol name containing the voltage information.
    
    Returns:
    pt_volt (float): Extracted voltage in volts (V). Returns NaN if no voltage information is found.
    """
    proto_fields = proto_name.lower().split('_')
    pt_volt = np.nan
    
    for fld in proto_fields:
        if 'mv' in fld:
            vval = re.findall(r'\d+\.?\d*', fld)
            if not vval:
                continue
            pt_volt = float(vval[0]) * 1e-3
            break

        if 'v' in fld:
            vval = re.findall(r'\d+\.?\d*', fld)
            if not vval:
                continue
            pt_volt = float(vval[0])
            break

    if np.isnan(pt_volt):
        print('Could not extract PT voltage from the protocol name.')

    return pt_volt

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
