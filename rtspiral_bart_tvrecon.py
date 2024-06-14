
import ismrmrd
import os
import sys
# import itertools
import logging
import numpy as np
# import numpy.fft as fft
import ctypes

import rtoml._rtoml
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
import rtoml

# Read config file and import BART
with open('configs/rtspiral_bart_tvrecon_config.toml') as jf:
    cfg = rtoml.load(jf)
    n_arm_per_frame = cfg['reconstruction']['arms_per_frame']
    APPLY_GIRF = cfg['reconstruction']['apply_girf']
    gpu_device = cfg['reconstruction']['gpu_device']
    BART_TOOLBOX_PATH = cfg['reconstruction']['BART_TOOLBOX_PATH']
    n_recon_frames = cfg['reconstruction']['num_recon_frames']
    reg_lambda = cfg['reconstruction']['reg_lambda']
    max_iter = cfg['reconstruction']['num_iter']


sys.path.append(f'{BART_TOOLBOX_PATH}/python/')
import bart # type: ignore

# Set BART related env vars
os.environ['BART_TOOLBOX_PATH'] = BART_TOOLBOX_PATH
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

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
    device = sp.Device(0)

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
    # with device:
    #     sens_map = mri.app.JsenseRecon(ksp_gpu,
    #                             coord=coord_gpu, weights=dcf_gpu,device=device, img_shape=(msize, msize)).run()

    nc = ksp_gpu.shape[0]
    nk = ksp_gpu.shape[1]


    kdata = np.transpose(data[:,:,:n_recon_frames*n_arm_per_frame], (2, 0, 1))
    kdata = kdata.reshape(n_recon_frames, n_arm_per_frame, nc, nk)            # [n_recon_frames, n_arms, n_ch, n_samples]
    kdata = np.transpose(kdata, (3, 1, 2, 0))                           # [n_samples, n_arms, n_ch n_recon_frames]
    ksp_all  = np.reshape(np.transpose(np.squeeze(kdata), (0, 1, 3, 2)), [1, nk, -1, nc])

    kdata = kdata[None,:,:,:,None,None,None,None,None,None,:]
    kloc = np.transpose(coord[:,:n_recon_frames*n_arm_per_frame,:], (1, 0, 2))
    kloc = kloc.reshape(n_recon_frames, n_arm_per_frame, nk, -1)                # [n_recon_frames, n_arms, n_samples, 2]
    kloc = np.concatenate((kloc, np.zeros((n_recon_frames, n_arm_per_frame, nk, 1))), axis=3)
    kloc = kloc.transpose((3, 2, 1, 0))
    traj_all = np.reshape(kloc, (3, nk, -1))
    kloc = kloc[:,:,:,None,None,None,None,None,None,None,:]

    # sens_map = sens_map.transpose((1, 2, 0))
    # sens_map = sens_map[:,:,None,:]
    print('kdata array shape: {}'.format(kdata.shape))
    print('kloc array shape: {}'.format(kloc.shape))
    # print('sens_map array shape: {}'.format(sens_map.shape))


    #       0           1           2           3           4           5
    #   READ_DIM,   PHS1_DIM,   PHS2_DIM,   COIL_DIM,   MAPS_DIM,   TE_DIM,
    #       6           7           8           9          10          11 
    #   COEFF_DIM,  COEFF2_DIM, ITER_DIM,   CSHIFT_DIM, TIME_DIM,   TIME2_DIM
    #      12          13          14      
    #   LEVEL_DIM,  SLICE_DIM,  AVG_DIM

    ################################################################################
    # Gridding Example for one frame
    #

    # zero_filed_img = sp.nufft_adjoint(kdata[0, :, :, :] * kweight, kloc[0, :, :, :], (nc, rNy, rNx))
    # pl.ImagePlot(xp.squeeze(zero_filed_img), z=0, title='Multi-channel Gridding')


    ################################################################################
    # CS Reconstruction
    #

    _, rtnlinv_sens_32 = bart.bart(2, 'nlinv -a 32 -b 16  -S -d4 -i13 -x 32:32:1 -t',
            traj_all, ksp_all)

    sens_ksp = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(rtnlinv_sens_32, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    sens_ksp = bart.bart(1, f'resize -c 0 {msize*2} 1 {msize*2}', sens_ksp)
    # sens_ksp = padarray(sens_ksp, [(2*Nx-64)/2 (2*Ny-64)/2]);
    sens = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sens_ksp, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    sens = bart.bart(1, f'resize -c 0 {msize} 1 {msize}', sens)
    # sens = centeredCrop(sens, [Nx/2, Ny/2, 0]);
    sens_map = bart.bart(1, f'normalize 8', sens)
    # sens = sens/vecnorm(sens,2,4)


    img = bart.bart(1, f'pics -g -m -e -S -R T:1024:1024:{reg_lambda*n_recon_frames} -d 4 -i {max_iter} -t', kloc, kdata, sens_map)
    img = np.squeeze(img)
    for ii in range(n_recon_frames):
        image = process_group(grp, img[None,:,:,ii], ii, [], metadata)
        connection.send_image(image)

    connection.send_close()
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    


def process_group(group, data, rep, config, metadata):
   

    data = np.abs(np.flip(data, axis=(1,)))
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


