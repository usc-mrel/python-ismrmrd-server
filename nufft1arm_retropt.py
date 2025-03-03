
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import ctypes
import mrdhelper
from datetime import datetime
import time

from scipy.io import loadmat
# from sigpy.linop import NUFFT
from scipy.ndimage import rotate
import sigpy as sp
from sigpy import fourier
# import cupy as cp
import GIRF
from pt_utils import pt
from connection import Connection
import sys
import socket


# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection: Connection, config, metadata):
    logging.disable(logging.CRITICAL)
    
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

    n_arm_per_frame = 34
    APPLY_GIRF = False

    gbar = 42.576e6
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
    # Load some useful header info into vars
    f0 = metadata.experimentalConditions.H1resonanceFrequency_Hz
    fpt   =  23.8e6 #23.635e6 # [Hz]
    fdiff =  f0-fpt #-45.2e3;   # fpt-f0 [Hz]
    dt = 1e-6 # [s]

    patient_position = metadata.measurementInformation.patientPosition.value
    r_PCS2DCS = GIRF.calculate_matrix_pcs_to_dcs(patient_position)

    gx = np.diff(np.concatenate((np.zeros((1, kx.shape[1])), kx)), axis=0)/dt/gbar
    gy = np.diff(np.concatenate((np.zeros((1, kx.shape[1])), ky)), axis=0)/dt/gbar
    g_nom = np.stack((gx, gy), axis=2)*1e3 # [mT/m] -> [T/m]
    g_gcs = np.concatenate((g_nom, np.zeros((g_nom.shape[0], g_nom.shape[1], 1))), axis=2)*1e-3 # [T/m] -> [mT/m]
    g_gcs = np.concatenate((np.zeros((traj['param']['pre_discard'][0,0][0,0], g_gcs.shape[1], g_gcs.shape[2])), g_gcs), axis = 0)

    sR = {'T': 0.55}
    tRR = 3


    ktraj = np.stack((kx, ky), axis=2)

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
    pt_sig = []
    # pt_sig_q = queue.Queue()
    tt = []
    fcorrmin_l = []
    arm_counter = 0
    rep_counter = 0
    device = sp.Device(1)
    # Plotting
    # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.connect(('127.0.0.1', 12345))  # Use the same host and port as the server
    # client_socket.sendall(nchannel.to_bytes(2, 'big'))

    coord_gpu = sp.to_device(ktraj, device=device)
    w_gpu = sp.to_device(w, device=device)

    for arm in connection:
        if arm is None:
            break # End of acq
        # First arm came, if GIRF is requested, correct trajectories and reupload.
        if type(arm) is ismrmrd.Acquisition:
            if (arm.idx.kspace_encode_step_1 == 0): 
                r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                                        [1,    0,   0],  # [RO] = [1 0 0] * [c]
                                        [0,    0,   1]]) # [SL]   [0 0 1] * [s]
                r_GCS2PCS = np.array([np.array(arm.phase_dir), -np.array(arm.read_dir), arm.slice_dir])
                r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
                PCS_offset = np.array([-1, 1, 1])*np.array(arm.position)*1e-3
                GCS_offset = r_GCS2PCS.T.dot(PCS_offset)
                RCS_offset = r_GCS2RCS.dot(GCS_offset)

                t = np.arange(0, arm.data.shape[1])*dt
                df = 1/(dt*arm.data.shape[1])
                ksp_window = np.ones(arm.data.shape)

                # ================================
                # Demodulate any shifts
                # ================================
                thet = np.exp(-1j*np.cumsum(2*np.pi*gbar*np.sum(g_gcs*RCS_offset, axis=2), axis=0)*dt) # [rad]

                if APPLY_GIRF:
                    sR['R'] = r_GCS2DCS
                    k_pred, _ = GIRF.apply_GIRF(g_nom, dt, sR, tRR)
                    k_pred = np.flip(k_pred[:,:,0:2], axis=2) # Drop the z
                    kmax = np.max(np.abs(k_pred[:,:,0] + 1j * k_pred[:,:,1]))
                    k_pred = np.swapaxes(k_pred, 0, 1)
                    k_pred = 0.5 * (k_pred / kmax) * msize
                    coord_gpu = sp.to_device(k_pred, device=device) # Replace  the original k-space

            # PT processing routine

            # Apply the negative of the phase
            data_demod = arm.data*thet[:, arm_counter]

            # ================================
            # Fine tune the PT frequency. 
            # Fit and subtract the PT.
            # ================================
            fcorrmin = pt.find_freq_qifft(data_demod.T, df, fdiff, 3e3, 4, (1))
            fcorrmin_l.append(fcorrmin)
            ksp_ptsubbed, pt_sig_fit = pt.est_dtft(t, data_demod.T, np.array([fdiff])-fcorrmin, ksp_window)
            pt_sig.append(np.abs(pt_sig_fit))
            # pl.plot(np.abs(pt_sig_fit[0]))
            # data_bytes = pt_sig[-1].tobytes()
            # client_socket.sendall(data_bytes)

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

            if ((arm.idx.kspace_encode_step_1+1) % n_arm_per_frame) == 0:
                # start = time.perf_counter()
                image = process_group(arm, frames, device, rep_counter, config, metadata)
                # end = time.perf_counter()

                # print(f"Elapsed time for frame processing: {end-start} secs.")
                frames = []
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)

                rep_counter += 1
                # plt.plot(pt_sig)
                # plt.draw()
                # plt.pause(0.1)

    pt_waveform = process_pilottone(np.array(pt_sig).T, metadata)
    connection.send_waveform(pt_waveform)

        # if(arm.idx.kspace_encode_step_1 == 600):
        # plt.hold(False)
            # plt.show()
    # pl.plot(None, finished=True)
    # client_socket.sendall(b'Terminate')
    # plt.pause(0.5)
    # client_socket.close()



def process_group(group, frames, device, rep, config, metadata):
    xp = device.xp
    with device:
        data = xp.zeros(frames[0].shape, dtype=np.complex128)
        for g in frames:
            data += g
        # Sum of squares coil combination
        data = np.abs(np.flip(data, axis=(1,2)))
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


def process_pilottone(pt_sig: np.ndarray, metadata):
    dt_pt = metadata.sequenceParameters.TR[0]

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    pt_sig = pt_sig - np.mean(pt_sig, axis=1, keepdims=True)
    # Normalize and convert to int16
    pt_sig *= maxVal/pt_sig.max()
    pt_sig = np.around(pt_sig)
    pt_sig = pt_sig.astype(np.int16)
    pt_wave = ismrmrd.Waveform.from_array(pt_sig)
    pt_wave.sample_time_us = dt_pt*1e3

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1', 12345))  # Use the same host and port as the server
    import pickle
    wvfrmd = {'signal': pt_sig, 'dt': dt_pt*1e-3, 'coil_labels': metadata.acquisitionSystemInformation.coilLabel}
    data_bytes = pickle.dumps(wvfrmd)
    # data_bytes = pt_sig.tobytes()
    client_socket.sendall(data_bytes)
    client_socket.close()
    # Let's attempt hijacking mrd's waveform
    return pt_wave

