import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import ismrmrd
import numpy as np
import sigpy as sp
from sigpy import fourier

import connection
import reconutils

# Folder for debug output files
debugFolder = "/tmp/share/debug"


def process(conn: connection.Connection, config, metadata):
    logging.disable(logging.DEBUG)
    
    logging.info("Config: \n%s", config)
    # logging.info("Metadata: \n%s", metadata)

    cfg = reconutils.load_config('rtspiral_vs_config.toml')
    if cfg is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    n_arm_per_frame = cfg['reconstruction']['arms_per_frame']
    window_shift    = cfg['reconstruction']['window_shift']
    APPLY_GIRF      = cfg['reconstruction']['apply_girf']
    gpu_device      = cfg['reconstruction']['gpu_device']
    coil_combine    = cfg['reconstruction']['coil_combine']
    save_complex    = cfg['reconstruction']['save_complex']
    ignore_arms_per_frame = cfg['reconstruction']['ignore_arms_per_frame']
    metafile_paths = cfg['metafile_paths']

    logging.info(f'''
                 ================================================================
                 Arms per frame: {n_arm_per_frame}
                 Window shift: {window_shift}
                 FOV oversampling: {cfg['reconstruction']['fov_oversampling']}
                 Apply GIRF?: {APPLY_GIRF}
                 GPU Device: {gpu_device}
                 Coil Combine: {coil_combine}
                 Save Complex: {save_complex}
                 =================================================================''')

    traj = reconutils.load_trajectory(metadata, metafile_paths)
    if traj is None:
        logging.error("Failed to load trajectory.")
        return

    if ignore_arms_per_frame:
        n_arm_per_frame = int(traj['param']['interleaves'][0,0][0,0])
        window_shift = n_arm_per_frame
        logging.info(f"Overriding arms per frame to {n_arm_per_frame} and window shift to {window_shift}")

    n_unique_angles = int(traj['param']['interleaves'][0,0][0,0])
    # nTRs = traj['param']['repetitions'][0,0][0,0]  # Not used in this implementation

    kx = traj['kx'][:,:n_unique_angles]
    ky = traj['ky'][:,:n_unique_angles]

    # Prepare gradients and variables if GIRF is requested. 
    # Unfortunately, we don't know rotations until the first data, so we can't prepare them yet.
    if APPLY_GIRF:
        # We get dwell time too late from MRD, as it comes with acquisition.
        # So we ask it from the metadata.
        try:
            dt = traj['param']['dt'][0,0][0,0]
        except KeyError:
            dt = 1e-6 # [s]

        gx = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(kx, axis=0)))/dt/42.58e6
        gy = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(ky, axis=0)))/dt/42.58e6
        g_nom = np.stack((gx, -gy), axis=2)

    ktraj = np.stack((kx, -ky), axis=2)

    # find max ktraj value
    kmax = np.max(np.abs(kx + 1j * ky))

    # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
    ktraj = np.swapaxes(ktraj, 0, 1)

    msize = int(cfg['reconstruction']['fov_oversampling'] * 10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0])

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
    slc_counter = 0
    device = sp.Device(gpu_device)

    coord_gpu = sp.to_device(ktraj, device=device)
    w_gpu = sp.to_device(w, device=device)

    sens = None
    wf_list = []

    # Create thread-safe queue and stop event
    data_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent excessive memory usage
    stop_event = threading.Event()
    
    # Use ThreadPoolExecutor for better resource management
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="DataAcquisition") as executor:
        # Submit the data acquisition worker
        if cfg['save_raw']:
            output_file_path = os.path.join(conn.savedataFolder)
            if (metadata.measurementInformation.protocolName != ""):
                output_file_path = os.path.join(conn.savedataFolder, f"meas_MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_{metadata.measurementInformation.protocolName}_{datetime.now().strftime('%H%M%S')}.h5")
            else:
                output_file_path = os.path.join(conn.savedataFolder, f"meas_MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_UnknownProtocol_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.h5")
            future = executor.submit(reconutils.data_acquisition_with_save_worker, conn, data_queue, stop_event, output_file_path, metadata)
        else:
            future = executor.submit(reconutils.data_acquisition_worker, conn, data_queue, stop_event)
        
        scan_counter_tracker = 0
        
        try:
            # Main processing loop
            while True:
                arm = None
                
                try:
                    # Get data from queue with timeout to avoid indefinite blocking
                    arm = data_queue.get(timeout=1.0)
                    
                    if arm is None:
                        # Sentinel value indicates end of data
                        break
                        
                except queue.Empty:
                    # Check if the acquisition task is still running
                    if future.done():
                        # Task completed, check for any remaining data
                        try:
                            arm = data_queue.get_nowait()
                            if arm is None:
                                break
                        except queue.Empty:
                            break
                    else:
                        # Task still running, continue waiting
                        continue

                if isinstance(arm, ismrmrd.Waveform):
                    # Accumulate waveforms to send at the end
                    wf_list.append(arm)
                    continue
                elif not isinstance(arm, ismrmrd.Acquisition):
                    continue
                    
                # At this point, we know arm is an Acquisition object
                start_iter = time.perf_counter()

                if arm.scan_counter % 1000 == 0:
                    logging.info("Processing acquisition %d", arm.scan_counter)

                if arm.scan_counter == (scan_counter_tracker+1):
                    scan_counter_tracker = arm.scan_counter
                else:
                    logging.warning("Scan counter mismatch: expected %d, got %d", scan_counter_tracker+1, arm.scan_counter)
                    scan_counter_tracker = arm.scan_counter
        
                # First arm came, if GIRF is requested, correct trajectories and reupload.
                if (arm.scan_counter == 1) and APPLY_GIRF:
                    k_pred = reconutils.girf_calibration(g_nom, metadata.measurementInformation.patientPosition.value, arm, dt, msize, girf_file=cfg['girf_file'])
                    coord_gpu = sp.to_device(k_pred, device=device) # Replace  the original k-space

                if (arm.scan_counter == 1) and (arm.data.shape[1]-pre_discard/2) == coord_gpu.shape[1]/2:
                    # Check if the OS is removed. Should only happen with offline recon.
                    coord_gpu = coord_gpu[:,::2,:]
                    w_gpu = w_gpu[:,::2]
                    pre_discard = int(pre_discard//2)
                    

                startarm = time.perf_counter()
                adata = sp.to_device(arm.data[:,pre_discard:], device=device)

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

                if arm.idx.slice != slc_counter:
                    rep_counter = 0
                    slc_counter = arm.idx.slice
                    logging.info(f"Processing slice {slc_counter} ...")

                if ((arm.scan_counter) % window_shift) == 0 and ((arm.scan_counter) >= n_arm_per_frame):
                    start = time.perf_counter()
                    if coil_combine == "adaptive" and rep_counter == 0:
                        logging.info("Calculating coil sensitivity maps...")
                        sens = sp.to_device(reconutils.process_csm(frames), device=device)

                    if save_complex:
                        image = reconutils.process_frame_complex(arm, frames, sens, device, rep_counter, cfg, metadata)
                    else:
                        image = reconutils.process_group(arm, frames, sens, device, rep_counter, cfg, metadata)
                    end = time.perf_counter()

                    logging.debug("Elapsed time for frame processing: %f secs.", end-start)
                    del frames[:window_shift]
                    logging.debug("Sending image to client:\n%s", image)
                    conn.send_image(image)

                    rep_counter += 1

                end_iter = time.perf_counter()
                logging.debug("Elapsed time for per iteration: %f secs.", end_iter-start_iter)

        except KeyboardInterrupt:
            logging.info("Received interrupt signal, stopping acquisition...")
            stop_event.set()
        
        except Exception as e:
            logging.error(f"Error in main processing loop: {e}")
            stop_event.set()
        
        finally:
            # Ensure clean shutdown
            stop_event.set()
            
            # Wait for the acquisition task to complete with timeout
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logging.warning(f"Error while waiting for acquisition task to complete: {e}")

    # Send waveforms back to save them with images
    # for wf in wf_list:
    #     conn.send_waveform(wf)

    conn.send_close()
    logging.debug('Reconstruction is finished.')
