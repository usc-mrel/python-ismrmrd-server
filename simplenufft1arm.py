
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
from sigpy.linop import NUFFT
from sigpy import Device
from sigpy import to_device
from scipy.ndimage import rotate

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

    n_arm_per_frame = 23

    if N is None:
        start = time.perf_counter()
        # get the k-space trajectory based on the metadata hash.
        traj_name = metadata.userParameters.userParameterString[1].value

        # load the .mat file containing the trajectory
        traj = loadmat("seq_meta/" + traj_name)

        n_unique_angles = traj['param']['repetitions'][0,0][0,0]

        # truncate the trajectory to the number of interleaves (NOTE: this won't work for golden angle?)
        kx = traj['kx'][:,:]
        ky = traj['ky'][:,:]
        ktraj = np.stack((kx, ky), axis=2)

        # find max ktraj value
        kmax = np.max(np.abs(kx + 1j * ky))

        # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
        ktraj = np.swapaxes(ktraj, 0, 1)

        msize = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0])

        ktraj = 0.5 * (ktraj / kmax) * msize

        nchannel = metadata.acquisitionSystemInformation.receiverChannels
        pre_discard = traj['param']['pre_discard'][0,0][0,0]
        # create the NUFFT operator
        N_ = []
        for arm_i in range(0, n_unique_angles):
            N_.append(NUFFT([nchannel, msize, msize], ktraj[arm_i,:,:]))

        w = traj['w']
        w = np.reshape(w, (1,w.shape[1]))
        end = time.perf_counter()
        # logging.debug("Elapsed time during recon prep: %f secs.", end-start)
        # print(f"Elapsed time during recon prep: {end-start} secs.")
    else:
        interleaves = N.ishape[0]

    # Discard phase correction lines and accumulate lines until we get fully sampled data
    frames = []
    arm_counter = 0
    # for group in conditionalGroups(connection, lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA), lambda acq: ((acq.idx.kspace_encode_step_1+1) % interleaves == 0)):
    for arm in connection:
        # startarm = time.perf_counter()

        # frames.append(process_arm(N_[arm_counter], w, arm, config, metadata))
        frames.append(N_[arm_counter].H * (arm.data[:,pre_discard:] * w))

        # endarm = time.perf_counter()
        # print(f"Elapsed time for arm NUFFT: {(endarm-startarm)*1e3} ms.")

        arm_counter += 1
        if arm_counter == n_unique_angles:
            arm_counter = 0

        if ((arm.idx.kspace_encode_step_1+1) % n_arm_per_frame) == 0:
            # start = time.perf_counter()

            image = process_group(arm, frames, config, metadata)
            # end = time.perf_counter()

            # print(f"Elapsed time for frame processing: {end-start} secs.")
            frames = []
            logging.debug("Sending image to client:\n%s", image)
            connection.send_image(image)





def process_arm(N, w, arm, config, metadata):
    # if len(group) == 0:
    #     return []

    # Create folder, if necessary
    # if not os.path.exists(debugFolder):
    #     os.makedirs(debugFolder)
    #     logging.debug("Created folder " + debugFolder + " for debug output files")


    # Format data into single [cha "PE" RO] array
    # data = arm.data #[acquisition.data for acquisition in group]
    # data = np.stack(data, axis=1)

    # n_samples_keep = N.oshape[1]
    # n_samples_discard = arm.data.shape[1] - n_samples_keep

    # discard the extra samples (in the readout direction)
    # data = arm.data[:,n_samples_discard:]

    data = N.H * (arm.data[:,10:] * w)

    return data

def process_group(group, frames, config, metadata):
    data = np.zeros(frames[0].shape, dtype=np.complex128)
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

    interleaves = 20
    repetition = 0 # np.int32((group[-1].idx.kspace_encode_step_1+1) / interleaves)
    

    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data.transpose(), acquisition=group, transpose=False)

    image.image_index = repetition

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


