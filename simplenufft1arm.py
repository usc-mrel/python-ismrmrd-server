
import ismrmrd
import logging
import numpy as np
import numpy.typing as npt
import ctypes
import mrdhelper
import time
import rtoml

from scipy.io import loadmat
from scipy.ndimage import rotate
import sigpy as sp
from sigpy import fourier
import GIRF
import coils
# Folder for debug output files
debugFolder = "/tmp/share/debug"


def process(connection, config, metadata):
    logging.disable(logging.CRITICAL)
    
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

    # We now read these parameters from toml file, so that we won't have to keep restarting the server when we change them.
    print('Loading and applying file configs/rtspiral_vs_config.toml')
    with open('configs/rtspiral_vs_config.toml') as jf:
        cfg = rtoml.load(jf)
        n_arm_per_frame = cfg['reconstruction']['arms_per_frame']
        window_shift    = cfg['reconstruction']['window_shift']
        APPLY_GIRF      = cfg['reconstruction']['apply_girf']
        gpu_device      = cfg['reconstruction']['gpu_device']
        coil_combine    = cfg['reconstruction']['coil_combine']

    print(f'Arms per frame: {n_arm_per_frame}, Apply GIRF?: {APPLY_GIRF}')
    
    # start = time.perf_counter()
    # get the k-space trajectory based on the metadata hash.
    traj_name = metadata.userParameters.userParameterString[1].value

    # load the .mat file containing the trajectory
    traj = loadmat("seq_meta/" + traj_name)

    n_unique_angles = traj['param']['repetitions'][0,0][0,0]

    kx = traj['kx'][:,:]
    ky = traj['ky'][:,:]


    # Prepare gradients and variables if GIRF is requested. 
    # Unfortunately, we don't know rotations until the first data, so we can't prepare them yet.
    if APPLY_GIRF:
        # We get dwell time too late from MRD, as it comes with acquisition.
        # So we ask it from the metadata.
        try:
            dt = traj['param']['dt'][0,0][0,0]
        except:
            dt = 1e-6 # [s]

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

    sens = []
    wf_list = []

    for arm in connection:
        if arm is None:
            break
        start_iter = time.perf_counter()
        
        if type(arm) is ismrmrd.Acquisition:
            # First arm came, if GIRF is requested, correct trajectories and reupload.
            if (arm.idx.kspace_encode_step_1 == 0) and APPLY_GIRF:
                r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                                        [1,    0,   0],  # [RO] = [1 0 0] * [c]
                                        [0,    0,   1]]) # [SL]   [0 0 1] * [s]
                r_GCS2PCS = np.array([arm.phase_dir, -np.array(arm.read_dir), arm.slice_dir])
                r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
                sR['R'] = r_GCS2DCS.dot(r_GCS2RCS)
                k_pred, _ = GIRF.apply_GIRF(g_nom, dt, sR, tRR)
                # k_pred = np.flip(k_pred[:,:,0:2], axis=2) # Drop the z
                k_pred = k_pred[:,:,0:2] # Drop the z

                kmax = np.max(np.abs(k_pred[:,:,0] + 1j * k_pred[:,:,1]))
                k_pred = np.swapaxes(k_pred, 0, 1)
                k_pred = 0.5 * (k_pred / kmax) * msize
                coord_gpu = sp.to_device(k_pred, device=device) # Replace  the original k-space

            if (arm.idx.kspace_encode_step_1 == 0) and (arm.data.shape[1]-pre_discard/2) == coord_gpu.shape[1]/2:
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
                
            # frames.append(sp.nufft_adjoint(arm.data[:,pre_discard:] * w, ktraj[arm_counter,:,:], oshape=(nchannel, msize, msize)))

            endarm = time.perf_counter()
            # print(f"Elapsed time for arm {arm_counter} NUFFT: {(endarm-startarm)*1e3} ms.")

            arm_counter += 1
            if arm_counter == n_unique_angles:
                arm_counter = 0

            if ((arm.idx.kspace_encode_step_1+1) % window_shift) == 0 and ((arm.idx.kspace_encode_step_1+1) >= n_arm_per_frame):
                start = time.perf_counter()
                if coil_combine == "adaptive" and rep_counter == 0:
                    sens = sp.to_device(process_csm(frames), device=device)

                image = process_group(arm, frames, sens, device, rep_counter, cfg, metadata)
                end = time.perf_counter()

                # print(f"Elapsed time for frame processing: {end-start} secs.")
                del frames[:window_shift]
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)

                rep_counter += 1

            end_iter = time.perf_counter()
            # print(f"Elapsed time for per iteration: {end_iter-start_iter} secs.")
        elif type(arm) is ismrmrd.Waveform:
            wf_list.append(arm)

    # Send waveforms back to save them with images
    for wf in wf_list:
        connection.send_waveform(wf)

    connection.send_close()
    print('Reconstruction is finished.')


def process_csm(frames):
    data = np.zeros(frames[0].shape, dtype=np.complex128)
    for g in frames:
        data += sp.to_device(g)
    (csm_est, rho) = coils.calculate_csm_inati_iter(data, smoothing=5)

    return csm_est


def process_group(group, frames: list, sens: npt.ArrayLike, device, rep, config, metadata):
    xp = device.xp
    with device:
        data = xp.zeros(frames[0].shape, dtype=np.complex128)
        for g in frames:
            data += g

        if sens.__len__() == 0:
            # Sum of squares coil combination
            data = np.abs(np.flip(data, axis=(1,)))
            data = np.square(data)
            data = np.sum(data, axis=0)
            data = np.sqrt(data)
        else:
            # Coil combine
            data = np.flip(np.abs(np.sum(np.conj(sens) * data, axis=0)), axis=(0,))
            
        # Determine max value (12 or 16 bit)
        BitsStored = 12
        if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
            BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
        maxVal = 2**BitsStored - 1

        # Normalize and convert to int16
        data *= maxVal/data.max()
        data = np.around(data)
        data = data.astype(np.int16)

    data = sp.to_device(data)

    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data.transpose(), acquisition=group, transpose=False)

    image.image_index = rep
    image.repetition = rep

    # Set field of view
    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON', 'simplenufft1arm'],
                         'WindowCenter':           str((maxVal+1)/2),
                         'WindowWidth':            str((maxVal+1)),
                         'NumArmsPerFrame':        str(config['reconstruction']['arms_per_frame']),
                         'GriddingWindowShift':    str(config['reconstruction']['window_shift'])})

    # Add image orientation directions to MetaAttributes if not already present
    if meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]), "{:.18f}".format(image.getHead().read_dir[1]), "{:.18f}".format(image.getHead().read_dir[2])]

    if meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]), "{:.18f}".format(image.getHead().phase_dir[1]), "{:.18f}".format(image.getHead().phase_dir[2])]

    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image


