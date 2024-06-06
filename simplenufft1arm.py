
import ismrmrd
# import os
# import itertools
import logging
import numpy as np
# import numpy.fft as fft
import ctypes
import mrdhelper
# from datetime import datetime
import time
import json

from scipy.io import loadmat
# from sigpy.linop import NUFFT
from scipy.ndimage import rotate
import sigpy as sp
from sigpy import fourier
# import cupy as cp
import GIRF
# Folder for debug output files
debugFolder = "/tmp/share/debug"

def groups(iterable, predicate):
    group = []
    for item in iterable:
        group.append(item)

        if predicate(item):
            yield group
            group = []


def conditionalGroups(iterable, predicateAccept, predicateFinish):
    group = []
    try:
        for item in iterable:
            if item is None:
                break

            if predicateAccept(item):
                group.append(item)

            if predicateFinish(item):
                yield group
                group = []
    finally:
        iterable.send_close()


def process(connection, config, metadata, N=None, w=None):
    logging.disable(logging.CRITICAL)
    
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

    start = time.perf_counter()
    # We now read these parameters from json file, so that we won't have to keep restarting the server when we change them.
    with open('spiralrt_config.json') as jf:
        cfg = json.load(jf)
        n_arm_per_frame = cfg['arms_per_frame']
        window_shift = cfg['window_shift']
        APPLY_GIRF = cfg['apply_girf']
        gpu_device = cfg['gpu_device']
    end = time.perf_counter()
    print(f"Elapsed time during json config read: {end-start} secs.")
    # n_arm_per_frame = 89
    # APPLY_GIRF = True
    print(f'Arms per frame: {n_arm_per_frame}, Apply GIRF?: {APPLY_GIRF}')

    if N is None:
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
    else:
        interleaves = N.ishape[0]

    # Discard phase correction lines and accumulate lines until we get fully sampled data
    frames = []
    arm_counter = 0
    rep_counter = 0
    device = sp.Device(gpu_device)

    coord_gpu = sp.to_device(ktraj, device=device)
    w_gpu = sp.to_device(w, device=device)

    # for group in conditionalGroups(connection, lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA), lambda acq: ((acq.idx.kspace_encode_step_1+1) % interleaves == 0)):
    for arm in connection:
        start_iter = time.perf_counter()

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

            image = process_group(arm, frames, device, rep_counter, config, metadata)
            end = time.perf_counter()

            # print(f"Elapsed time for frame processing: {end-start} secs.")
            del frames[:window_shift]
            logging.debug("Sending image to client:\n%s", image)
            start = time.perf_counter()
            connection.send_image(image)
            end = time.perf_counter()

            print(f"Elapsed time for frame sending: {end-start} secs.")
            rep_counter += 1

        end_iter = time.perf_counter()
        # print(f"Elapsed time for per iteration: {end_iter-start_iter} secs.")



def process_group(group, frames, device, rep, config, metadata):
    xp = device.xp
    with device:
        data = xp.zeros(frames[0].shape, dtype=np.complex128)
        for g in frames:
            data += g
        # Sum of squares coil combination
        data = np.abs(np.flip(data, axis=(1,)))
        data = np.square(data)
        data = np.sum(data, axis=0)
        data = np.sqrt(data)

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

    # Set field of view
    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           str((maxVal+1)/2),
                         'WindowWidth':            str((maxVal+1))})

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


