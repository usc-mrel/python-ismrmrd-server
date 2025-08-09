
import collections
import logging
import os
import pathlib
import threading
import time
from datetime import datetime
from typing import Tuple

import ismrmrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rtoml

import connection

matplotlib.use('Agg')  # Use non-interactive backend for saving figures

import sigpy as sp
from scipy.io import savemat
from scipy.signal import savgol_filter
from sigpy import fourier

import GIRF
from pilottone import calc_fovshift_phase, pt
from reconutils import (
    data_acquisition_thread,
    load_trajectory,
    process_csm,
    process_frame_complex,
    process_group,
)

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(conn: connection.Connection, config, metadata):
    logging.disable(logging.DEBUG)
    
    logging.info("Config: \n%s", config)
    # logging.info("Metadata: \n%s", metadata)

    # We now read these parameters from toml file, so that we won't have to keep restarting the server when we change them.
    logging.info('''Loading and applying file configs/rtspiral_vspt_config.toml''')
    with open(pathlib.Path(__file__).parent / 'configs/rtspiral_vspt_config.toml') as jf:
        cfg = rtoml.load(jf)

    n_arm_per_frame = cfg['reconstruction']['arms_per_frame']
    window_shift    = cfg['reconstruction']['window_shift']
    APPLY_GIRF      = cfg['reconstruction']['apply_girf']
    gpu_device      = cfg['reconstruction']['gpu_device']
    coil_combine    = cfg['reconstruction']['coil_combine']
    save_complex    = cfg['reconstruction']['save_complex']
    ignore_arms_per_frame = cfg['reconstruction']['ignore_arms_per_frame']
    metafile_paths = cfg['metafile_paths']
    f_pt = cfg['pilottone']['pt_freq']
    save_folder = cfg['pilottone']['save_folder']

    logging.info(f'''
                 ================================================================
                 Arms per frame: {n_arm_per_frame}
                 Apply GIRF?: {APPLY_GIRF}
                 GPU Device: {gpu_device}
                 Coil Combine: {coil_combine}
                 Save Complex: {save_complex}
                 =================================================================''')
    
    # start = time.perf_counter()
    traj = load_trajectory(metadata, metafile_paths)
    if traj is None:
        logging.error("Failed to load trajectory.")
        return

    if ignore_arms_per_frame:
        n_arm_per_frame = int(traj['param']['interleaves'][0,0][0,0])
        window_shift = n_arm_per_frame
        logging.info(f"Overriding arms per frame to {n_arm_per_frame} and window shift to {window_shift}")

    n_unique_angles = int(traj['param']['interleaves'][0,0][0,0])
    nTRs = traj['param']['repetitions'][0,0][0,0]

    kx = traj['kx'][:,:n_unique_angles]
    ky = traj['ky'][:,:n_unique_angles]



    # We get dwell time too late from MRD, as it comes with acquisition.
    # So we ask it from the metadata.
    try:
        dt = traj['param']['dt'][0,0][0,0]
    except KeyError:
        dt = 1e-6 # [s]
        logging.warning("Dwell time (dt) not found in trajectory parameters, using default value of 1 us.")

    # Useful parameters for pilot tone
    f0 = metadata.experimentalConditions.H1resonanceFrequency_Hz
    fdiff =  f0-f_pt #-45.2e3;   # fpt-f0 [Hz]
    t_adc = np.arange(0, kx.shape[0])*dt
    df = 1/(dt*kx.shape[0])
    coil_name = []

    for clbl in metadata.acquisitionSystemInformation.coilLabel:
        coil_name.append(clbl.coilName)

    coil_name = np.asarray(coil_name)

    # Prepare gradients and variables if GIRF is requested. 
    # Unfortunately, we don't know rotations until the first data, so we can't prepare them yet.
    if APPLY_GIRF:

        patient_position = metadata.measurementInformation.patientPosition.value
        r_PCS2DCS = GIRF.calculate_matrix_pcs_to_dcs(patient_position)

        gx = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(kx, axis=0)))/dt/42.58e6
        gy = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(ky, axis=0)))/dt/42.58e6
        g_nom = np.stack((gx, -gy), axis=2)

        sR = {'T': 0.55}
        tRR = 3*1e-6/dt


    ktraj = np.stack((kx, -ky), axis=2)

    # find max ktraj value
    kmax = np.max(np.abs(kx + 1j * ky))

    # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
    ktraj = np.swapaxes(ktraj, 0, 1)

    msize = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0])

    ktraj = 0.5 * (ktraj / kmax) * msize

    nchannel = metadata.acquisitionSystemInformation.receiverChannels
    pre_discard = traj['param']['pre_discard'][0,0][0,0]
    w = traj['w']
    w = np.reshape(w, (1,w.shape[1]))
    # end = time.perf_counter()
    # logging.debug("Elapsed time during recon prep: %f secs.", end-start)
    # print(f"Elapsed time during recon prep: {end-start} secs.")

    # Discard phase correction lines and accumulate lines until we get fully sampled data
    frames = []
    arm_counter = 0
    rep_counter = 0
    device = sp.Device(gpu_device)

    coord_gpu = sp.to_device(ktraj, device=device)
    w_gpu = sp.to_device(w, device=device)
    # Create deque and threading objects
    data_deque = collections.deque()
    stop_event = threading.Event()
    
    # Start data acquisition thread
    acquisition_thread = threading.Thread(
        target=data_acquisition_thread,
        args=(conn, data_deque, stop_event)
    )
    acquisition_thread.start()

    sens = None
    wf_list = []
    pt_sig = []  # Pilot tone signal
    arm: ismrmrd.Acquisition | ismrmrd.Waveform | None

    while True:
        # Try to get data from deque
        arm = None
        
        if data_deque:
            arm = data_deque.popleft()

        if arm is None:
            # No data available, check if acquisition thread is still alive
            if not acquisition_thread.is_alive():
                break
            # time.sleep(0.001)  # Small sleep to avoid busy waiting
            continue
            
        # if data_type == 'end':
        #     break
        elif type(arm) is ismrmrd.Waveform:
            # Accumulate waveforms to send at the end
            wf_list.append(arm)
            continue
        elif type(arm) is not ismrmrd.Acquisition:
            continue
            
        # At this point, we know arm is an Acquisition object
        assert arm is not None
        
        start_iter = time.perf_counter()


        # First arm came, if GIRF is requested, correct trajectories and reupload.
        if (arm.scan_counter == 1) and APPLY_GIRF:
            r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                                    [1,    0,   0],  # [RO] = [1 0 0] * [c]
                                    [0,    0,   1]]) # [SL]   [0 0 1] * [s]
            r_GCS2PCS = np.array([arm.phase_dir, -np.array(arm.read_dir), arm.slice_dir])
            r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
            sR['R'] = r_GCS2DCS.dot(r_GCS2RCS)
            k_pred, _ = GIRF.apply_GIRF(g_nom, dt, sR, tRR=tRR)
            # k_pred = np.flip(k_pred[:,:,0:2], axis=2) # Drop the z
            k_pred = k_pred[:,:,0:2] # Drop the z

            kmax = np.max(np.abs(k_pred[:,:,0] + 1j * k_pred[:,:,1]))
            k_pred = np.swapaxes(k_pred, 0, 1)
            k_pred = 0.5 * (k_pred / kmax) * msize
            coord_gpu = sp.to_device(k_pred, device=device) # Replace  the original k-space
        # This is a good place to calculate FOV shift phase.
        if (arm.scan_counter == 1):
            phase_mod_rads = calc_fovshift_phase(kx, ky, arm)


        if ((arm.scan_counter == 1) and (arm.data.shape[1]-pre_discard/2) == coord_gpu.shape[1]/2):
            # Check if the OS is removed. Should only happen with offline recon.
            coord_gpu = coord_gpu[:,::2,:]
            w_gpu = w_gpu[:,::2]
            pre_discard = int(pre_discard//2)

        if (arm.scan_counter == 1) and (cfg['reconstruction']['remove_oversampling']):
            logging.info("Removing oversampling from the data.")
            coord_gpu = coord_gpu[:,::2,:]
            w_gpu = w_gpu[:,::2]

        data_demod = arm.data[:,pre_discard:]*phase_mod_rads[None,:, arm_counter]
        ksp_ptsubbed, pt_sig_fit = pt.est_dtft(t_adc, data_demod.T[:,None,:], np.array([fdiff]))
        pt_sig.append(pt_sig_fit)
        startarm = time.perf_counter()
        if cfg['pilottone']['remove_pt']:
            adata = sp.to_device(ksp_ptsubbed.squeeze().T*phase_mod_rads[None,:, arm_counter].conj(), device=device)
        else:
            adata = sp.to_device(arm.data[:,pre_discard:], device=device)

        if cfg['reconstruction']['remove_oversampling']:
            n_samp = adata.shape[1]
            keepOS = np.concatenate([np.arange(n_samp // 4), np.arange(n_samp * 3 // 4, n_samp)])
            adata = fourier.fft(fourier.ifft(adata, center=False)[:, keepOS], center=False)
            
        with device:
            frames.append(fourier.nufft_adjoint(
                    adata*w_gpu,
                    coord_gpu[arm_counter,:,:],
                    (nchannel, msize, msize)))
            
        endarm = time.perf_counter()
        logging.debug("Elapsed time for arm %d NUFFT: %f ms.", arm_counter, (endarm-startarm)*1e3)

        arm_counter += 1
        if arm_counter == n_unique_angles:
            arm_counter = 0

        if ((arm.scan_counter) % window_shift) == 0 and ((arm.scan_counter) >= n_arm_per_frame):
            start = time.perf_counter()
            if coil_combine == "adaptive" and rep_counter == 0:
                sens = sp.to_device(process_csm(frames), device=device)

            if save_complex:
                image = process_frame_complex(arm, frames, sens, device, rep_counter, cfg, metadata)
            else:
                image = process_group(arm, frames, sens, device, rep_counter, cfg, metadata)
            end = time.perf_counter()

            logging.debug("Elapsed time for frame processing: %f secs.", end-start)
            del frames[:window_shift]
            logging.debug("Sending image to client:\n%s", image)
            conn.send_image(image)

            rep_counter += 1

        end_iter = time.perf_counter()
        logging.debug("Elapsed time for per iteration: %f secs.", end_iter-start_iter)


    # Send waveforms back to save them with images
    # for wf in wf_list:
    #     conn.send_waveform(wf)
    # Wait for acquisition thread to finish
    acquisition_thread.join()
    
    conn.send_close()
    logging.info('Reconstruction is finished.')
    # Filter and prepare pilot tone signal

    process_pilot_tone_signal(metadata, cfg, save_folder, coil_name, pt_sig)

def process_pilot_tone_signal(metadata, cfg, save_folder, coil_name, pt_sig):
    if len(pt_sig) > 0:
        logging.info("Processing pilot tone signal...")
        dt_pt = metadata.sequenceParameters.TR[0]*1e-3  # Convert TR from ms to seconds
        pt_sig = np.abs(np.array(pt_sig)).squeeze()
        pt_sig = np.squeeze(pt_sig - np.mean(pt_sig, axis=0, keepdims=True))
        pt_sig_filt = savgol_filter(pt_sig, 81, 3, axis=0)
        time_pt = np.arange(0, pt_sig.shape[0]) * dt_pt
        save_path = os.path.join(save_folder, f"MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_{metadata.measurementInformation.protocolName}_{datetime.now().strftime('%H%M%S')}")
        logging.info(f"Saving pilot tone signal to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        if cfg['pilottone']['save_raw']:
            logging.info("Saving raw pilot tone signal...")
            fig, axs = plot_rawpt(pt_sig_filt, coil_name, time_pt, sort=True)

            if cfg['pilottone']['save_svg']:
                fig.savefig(os.path.join(save_path, "pt_raw.svg"))
            if cfg['pilottone']['save_png']:
                fig.savefig(os.path.join(save_path, "pt_raw.png"), dpi=300)
            if cfg['pilottone']['save_mat']:
                savemat(os.path.join(save_path, "pt_raw.mat"), {'pt_signal': pt_sig_filt, 'dt': dt_pt})
            plt.close()
        
        if cfg['pilottone']['save_navs']:
            logging.info("Processing pilot tone signal for respiratory/cardiac signals...")
            f_samp = 1/dt_pt # [Hz]
            
            # Check if initial cardiac channel exists
            cardiac_init_ch = -1
            if cfg['pilottone']['cardiac']['initial_channel'] in coil_name:
                cardiac_init_ch = np.nonzero(coil_name == cfg['pilottone']['cardiac']['initial_channel'])[0][0]
            else:
                logging.warning(f"Initial cardiac channel {cfg['pilottone']['cardiac']['initial_channel']} not found in coil names. Using -1 as default. \nAvailable coils:\n{coil_name}")


            pt_extract_params = {'golay_filter_len': cfg['pilottone']['golay_filter_len'],
                    'respiratory': {
                            'freq_start': cfg['pilottone']['respiratory']['freq_start'],
                            'freq_stop': cfg['pilottone']['respiratory']['freq_stop'],
                            'corr_threshold': cfg['pilottone']['respiratory']['corr_threshold'],
                            'corr_init_ch': cfg['pilottone']['respiratory']['initial_channel'],
                            'separation_method': cfg['pilottone']['respiratory']['separation_method'], # 'sobi', 'pca'
                    },
                    'cardiac': {
                                'freq_start': cfg['pilottone']['cardiac']['freq_start'],
                                'freq_stop': cfg['pilottone']['cardiac']['freq_stop'],
                                'corr_threshold': cfg['pilottone']['cardiac']['corr_threshold'],
                                'corr_init_ch': cardiac_init_ch,                           
                                'separation_method': cfg['pilottone']['cardiac']['separation_method'], # 'sobi', 'pca'
                                'num_lags': 375, # SOBI number of lags
                    },
                    'debug': {
                        'selected_coils': cfg['pilottone']['debug']['selected_coils'],
                        'coil_legend': coil_name,
                        'show_plots': cfg['pilottone']['debug']['show_plots'],
                        'no_normalize': cfg['pilottone']['debug']['no_normalize'],
                    }
                }

            pt_respiratory, pt_cardiac = pt.extract_pilottone_navs(pt_sig, f_samp, pt_extract_params)

            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            axs[0].plot(time_pt, pt_respiratory)
            axs[0].set_title('Respiratory Signal')
            axs[1].plot(time_pt, pt_cardiac)
            axs[1].set_title('Cardiac Signal')
            axs[1].set_xlabel('Time [s]')
            plt.tight_layout()

            if cfg['pilottone']['save_svg']:
                fig.savefig(os.path.join(save_path, "pt_navs.svg"))
            if cfg['pilottone']['save_png']:
                fig.savefig(os.path.join(save_path, "pt_navs.png"), dpi=300)
            if cfg['pilottone']['save_mat']:
                savemat(os.path.join(save_path, "pt_navs.mat"), {'respiratory': pt_respiratory, 'cardiac': pt_cardiac, 'dt': dt_pt})
            plt.close()
        logging.info("Pilot tone processing is complete.")
    else:
        logging.warning("No pilot tone signal found. Skipping pilot tone processing.")


def plot_rawpt(pt_raw: np.ndarray, coil_name: np.ndarray, time_pt: np.ndarray, sort: bool=True) -> Tuple[plt.Figure, np.ndarray]:
    if sort:
        Isort = np.argsort(coil_name)
    else:
        Isort = np.arange(pt_raw.shape[1])
    
    spacing = np.abs(pt_raw).max()*pt_raw.shape[1]/2
    ptb = np.linspace(spacing, -spacing, pt_raw.shape[1])

    f, axs = plt.subplots(1,1)
    axs = np.atleast_1d(axs)
    f.set_size_inches(10, 10)
    lines = axs[0].plot(time_pt, ptb+pt_raw[:,Isort])
    for i, coil in enumerate((coil_name)[Isort]):
        axs[0].text(time_pt[-1]+10, ptb[i]+np.mean(pt_raw[:,i]), coil[-3:], fontsize=10, ha='right', va='center', color=lines[i].get_color())
    axs[0].set_xlabel('Time [s]')

    axs[0].set_xlim(0, time_pt[-1]+10)
    axs[0].set_yticks([])

    plt.suptitle('Raw Pilot Tones', fontsize=16)
    plt.tight_layout()
    return f, axs