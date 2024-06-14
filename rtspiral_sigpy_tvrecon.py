
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
from sigpy import mri
# import cupy as cp
import GIRF
import tvrecon
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

    data = []
    coord = []
    dcf = []
    grp = None
    # for group in conditionalGroups(connection, lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA), lambda acq: ((acq.idx.kspace_encode_step_1+1) % interleaves == 0)):
    for arm in connection:
        # start_iter = time.perf_counter()
        if arm is None:
            break
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
            ktraj = k_pred
            
        if (arm.idx.kspace_encode_step_1 == 0):
            grp = arm

        # startarm = time.perf_counter()
        # adata = sp.to_device(arm.data[:,pre_discard:], device=device)
        data.append(arm.data[:,pre_discard:])
        coord.append(ktraj[arm_counter,:,:])
        dcf.append(w[0,:])
        # with device:
        #     frames.append(fourier.nufft_adjoint(
        #             adata*w_gpu,
        #             coord_gpu[arm_counter,:,:],
        #             (nchannel, msize, msize)))

        # endarm = time.perf_counter()
        # print(f"Elapsed time for arm {arm_counter} NUFFT: {(endarm-startarm)*1e3} ms.")

        arm_counter += 1
        if arm_counter == n_unique_angles:
            arm_counter = 0

        # if ((arm.idx.kspace_encode_step_1+1) % n_arm_per_frame) == 0:
        #     start = time.perf_counter()

        #     image = process_group(arm, frames, device, rep_counter, config, metadata)
        #     end = time.perf_counter()

        #     # print(f"Elapsed time for frame processing: {end-start} secs.")
        #     frames = []
        #     logging.debug("Sending image to client:\n%s", image)
        #     start = time.perf_counter()
        #     # connection.send_image(image)
        #     end = time.perf_counter()

        #     print(f"Elapsed time for frame sending: {end-start} secs.")
        #     rep_counter += 1
        # end_iter = time.perf_counter()
        # print(f"Elapsed time for per iteration: {end_iter-start_iter} secs.")

    data = np.array(data)
    data = np.transpose(data, axes=(1, 2, 0))
    ksp_gpu = sp.to_device(data, device=device)
    coord = np.array(coord, dtype=np.float32)
    coord = np.transpose(coord, axes=(1, 0, 2))
    coord_gpu = sp.to_device(coord, device=device)
    dcf_gpu = sp.to_device(np.array(dcf, dtype=np.float32).T, device=device)
    with device:
        sens_map = mri.app.JsenseRecon(ksp_gpu,
                                coord=coord_gpu, weights=dcf_gpu,device=device, img_shape=(msize, msize)).run()
    sens_map = sp.to_device(sens_map, device=device)
    # lamda = 0.01
    # img_sense = mri.app.SenseRecon(ksp_gpu, sens_map, coord=coord_gpu, lamda=lamda, device=device).run()

    reg_lambda = 0.01
    n_recon_frames= 80
    n_arms = 7
    nc = ksp_gpu.shape[0]
    nk = ksp_gpu.shape[1]
    xp = device.xp
    rNy = msize
    rNx = msize
    max_iter = 30
    methods = 'pdhg'

    with device:

        kdata = xp.transpose(ksp_gpu, (2, 0, 1))
        # crop kdata to keep only the first (n_frames*n_arms) data
        kdata = kdata[:n_recon_frames*n_arms,:,:]
        kdata = kdata.reshape(n_recon_frames, n_arms, nc, nk)            # [n_recon_frames, n_arms, n_ch, n_samples]
        kdata = xp.transpose(kdata, (0, 2, 1, 3))                           # [n_recon_frames, n_ch, n_arms, n_samples]

        # kweight = xp.expand_dims(kweight, axis=0)                             # [1, n_samples]
        kweight = w_gpu                             # [1, n_samples]

        kloc = xp.transpose(coord_gpu, (1, 0, 2))
        kloc = kloc[:n_recon_frames*n_arms, :, :]
        kloc = kloc.reshape(n_recon_frames, n_arms, nk, -1)                # [n_recon_frames, n_arms, n_samples, 2]

        print('kdata array shape: {}'.format(kdata.shape))
        print('kweight array shape: {}'.format(kweight.shape))
        print('kloc array shape: {}'.format(kloc.shape))


    ################################################################################
    # Gridding Example for one frame
    #
    with device:

        zero_filed_img = sp.nufft_adjoint(kdata[0, :, :, :] * kweight, kloc[0, :, :, :], (nc, rNy, rNx))
        # pl.ImagePlot(xp.squeeze(zero_filed_img), z=0, title='Multi-channel Gridding')


        ################################################################################
        # CS Reconstruction
        #

        reg_lambda_scaled = reg_lambda * xp.max(xp.abs(zero_filed_img))
        if methods == "nlcg": # non-linear conjugate gradient
            img, fnorm, tnorm, cost = tvrecon.TotalVariationReconNLCG(kdata, kweight, kloc, sens_map, reg_lambda_scaled, max_iter, device=device).run()
        elif methods == "pdhg": # primal dual hybrid gradient
            img = tvrecon.TotalVariationRecon(kdata, kweight, kloc, sens_map, reg_lambda=reg_lambda_scaled, dim_fd=(0,), max_iter=max_iter, device=device).run()
        img = sp.to_device(img)
        image = process_group(grp, img, [], metadata)
        connection.send_image(image)

        pass

    


def process_group(group, data, config, metadata):
   

    data = np.abs(np.flip(data, axis=(2,)))
    data = np.transpose(data, (0, 2, 1))

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    # Normalize and convert to int16
    data *= maxVal/data.max()
    data = np.around(data)
    data = data.astype(np.int16)


    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data, acquisition=group, transpose=False)

    image.image_index = 0

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

def process_group_bkp(group, frames, device, rep, config, metadata):
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


