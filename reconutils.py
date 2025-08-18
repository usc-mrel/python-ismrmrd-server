import connection
import collections
import threading
import ismrmrd
import logging
import coils
import numpy as np
import numpy.typing as npt
import sigpy as sp
import ctypes
import os
import mrdhelper
from scipy.io import loadmat
import time

def data_acquisition_thread(conn: connection.Connection, data_deque: collections.deque, stop_event: threading.Event):
    """Thread function to acquire data from the connection and put it in the deque."""
    
    arm: ismrmrd.Acquisition | ismrmrd.Waveform | None
    for arm in conn:
        if arm is None or stop_event.is_set():
            data_deque.append(None)
            logging.info("Acquisition thread has stopped.")
            break

        data_deque.append(arm)
        time.sleep(0.001)
    
def process_csm(frames):
    data = np.zeros(frames[0].shape, dtype=np.complex128)
    for g in frames:
        data += sp.to_device(g)
    (csm_est, rho) = coils.calculate_csm_inati_iter(data, smoothing=32)

    return csm_est


def process_group(group, frames: list, sens: npt.NDArray | None, device, rep, config, metadata):
    xp = device.xp
    with device:
        data = xp.zeros(frames[0].shape, dtype=np.complex128)
        for g in frames:
            data += g

        if sens is None:
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
        dscale = maxVal/data.max()
        data *= dscale
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
                         'GriddingWindowShift':    str(config['reconstruction']['window_shift']), 
                         'ImageScaleFactor':       str(dscale)
                         })

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


def process_frame_complex(group, frames: list, sens: npt.NDArray | None, device, rep, config, metadata):
    xp = device.xp
    with device:
        data = xp.zeros(frames[0].shape, dtype=np.complex128)
        for g in frames:
            data += g

        if sens is None:
            logging.error("No coil sensitivity maps found. Cannot perform coil combination.")
            # Sum of squares coil combination
            data = np.abs(np.flip(data, axis=(1,)))
            data = np.square(data)
            data = np.sum(data, axis=0)
            data = np.sqrt(data)
        else:
            # Coil combine
            data = np.flip(np.sum(np.conj(sens) * data, axis=0), axis=(0,))

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
                         'WindowCenter':           str((data.max()+1)/2),
                         'WindowWidth':            str((data.max()+1)),
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

def load_trajectory(metadata, metafile_paths: list[str]) -> dict | None:
    # get the k-space trajectory based on the metadata hash.
    for str_param in metadata.userParameters.userParameterString:
        if str_param.name == "tSequenceVariant":
            traj_name = str_param.value[:32] # Get first 32 chars, because a bug sometimes causes this field to have /OSP added to the end.
            break
    else:
        logging.error("Sequence hash is not found in metadata user parameters.")
        return None

    # load the .mat file containing the trajectory
    # Search for the file in the metafile_paths
    for path in metafile_paths:
        metafile_fullpath = os.path.join(path, traj_name + ".mat")
        if os.path.isfile(metafile_fullpath):
            logging.info(f"Loading metafile {traj_name} from {path}...")
            traj = loadmat(metafile_fullpath)
            return traj
    else:
        logging.error(f"Trajectory file {traj_name}.mat not found in specified paths.")
        return None
    