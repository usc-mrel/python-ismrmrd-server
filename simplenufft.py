
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import ctypes
import mrdhelper
from datetime import datetime

from scipy.io import loadmat
from sigpy.linop import NUFFT
from sigpy import Device
from sigpy import to_device

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
        # get the k-space trajectory based on the metadata hash.
        traj_name = metadata.userParameters.userParameterString[1].value

        # load the .mat file containing the trajectory
        traj = loadmat("seq_meta/" + traj_name)

        interleaves = 20# np.int16(traj['param']['interleaves'])[0][0]

        # truncate the trajectory to the number of interleaves (NOTE: this won't work for golden angle?)
        kx = traj['kx'][:,:interleaves]
        ky = traj['ky'][:,:interleaves]
        ktraj = np.stack((kx, ky), axis=2)

        # find max ktraj value
        kmax = np.max(np.abs(kx + 1j * ky))

        # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
        ktraj = np.swapaxes(ktraj, 0, 1)

        msize = np.int16(10 * np.float32(traj['param']['fov'])[0][0] / np.float32(traj['param']['spatial_resolution'])[0][0])

        ktraj = 0.5 * (ktraj / kmax) * msize

        nchannel = metadata.acquisitionSystemInformation.receiverChannels

        # create the NUFFT operator
        N = NUFFT([nchannel, msize, msize], ktraj)
        w = traj['w']
        w = np.reshape(w, (1,1,w.shape[1]))
    else:
        interleaves = N.ishape[0]

    # Discard phase correction lines and accumulate lines until we get fully sampled data
    for group in conditionalGroups(connection, lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA), lambda acq: ((acq.idx.kspace_encode_step_1+1) % interleaves == 0)):
        image = process_group(N, w, group, config, metadata)

        logging.debug("Sending image to client:\n%s", image)
        connection.send_image(image)


def process_group(N, w, group, config, metadata):
    if len(group) == 0:
        return []

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")


    # Format data into single [cha "PE" RO] array
    data = [acquisition.data for acquisition in group]
    data = np.stack(data, axis=1)

    n_samples_keep = N.oshape[2]
    n_samples_discard = data.shape[2] - n_samples_keep

    # discard the extra samples (in the readout direction)
    data = data[:,:,n_samples_discard:]

    data = N.H * (data * w)

    # Sum of squares coil combination
    data = np.abs(data)
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

    interleaves = N.ishape[0]
    repetition = np.int32((group[-1].idx.kspace_encode_step_1+1) / interleaves)
    

    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data.transpose(), acquisition=group[0], transpose=False)

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


