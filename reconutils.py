import ctypes
import logging
import os
import pathlib
import queue
import threading

import ismrmrd
import numpy as np
import numpy.typing as npt
import rtoml
import sigpy as sp
from scipy.io import loadmat

import coils
import connection
import GIRF
import mrdhelper


def data_acquisition_worker(conn: connection.Connection, data_queue: queue.Queue, stop_event: threading.Event):
    """Worker function for data acquisition using concurrent.futures."""
    try:
        for arm in conn:
            if arm is None or stop_event.is_set():
                data_queue.put(None)
                logging.info("Acquisition worker has stopped.")
                break
            
            data_queue.put(arm)
            
            # Yield control to avoid hogging CPU
            # time.sleep(0.001)
            
    except Exception as e:
        logging.error(f"Error in data acquisition worker: {e}")
        data_queue.put(None)  # Ensure main thread doesn't hang
    finally:
        logging.info("Data acquisition worker finished.")

def data_acquisition_with_save_worker(conn: connection.Connection, data_queue: queue.Queue, stop_event: threading.Event, output_file_path: str, metadata: ismrmrd.xsd.ismrmrdHeader):
    """Worker function for data acquisition that also saves the raw data using concurrent.futures."""
    
    with ismrmrd.Dataset(output_file_path, create_if_needed=True) as dset:
        dset.write_xml_header(ismrmrd.xsd.ToXML(metadata))

        try:
            for arm in conn:
                if arm is None or stop_event.is_set():
                    data_queue.put(None)
                    logging.info("Acquisition worker has stopped.")
                    break
                
                data_queue.put(arm)
                if type(arm) is ismrmrd.Acquisition:
                    dset.append_acquisition(arm)
                elif type(arm) is ismrmrd.Waveform:
                    dset.append_waveform(arm)

                # Yield control to avoid hogging CPU
                # time.sleep(0.001)
                
        except Exception as e:
            logging.error(f"Error in data acquisition worker: {e}")
            data_queue.put(None)  # Ensure main thread doesn't hang
        finally:
            logging.info("Data acquisition worker finished.")

def girf_calibration(g_nom: np.ndarray, patient_position: str, arm: ismrmrd.Acquisition, dt: float, msize: int, girf_file: str) -> np.ndarray:

    r_PCS2DCS = GIRF.calculate_matrix_pcs_to_dcs(patient_position)
    r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                        [1,    0,   0],  # [RO] = [1 0 0] * [c]
                        [0,    0,   1]]) # [SL]   [0 0 1] * [s]
    r_GCS2PCS = np.array([arm.phase_dir, -np.array(arm.read_dir), arm.slice_dir])
    r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
    sR = r_GCS2DCS.dot(r_GCS2RCS)
    tRR = 3*1e-6/dt

    k_pred, _ = GIRF.apply_GIRF(g_nom, dt, sR, girf_file=girf_file, tRR=tRR)
    # k_pred = np.flip(k_pred[:,:,0:2], axis=2) # Drop the z
    k_pred = k_pred[:,:,0:2] # Drop the z

    kmax = np.max(np.abs(k_pred[:,:,0] + 1j * k_pred[:,:,1]))
    k_pred = np.swapaxes(k_pred, 0, 1)
    k_pred = 0.5 * (k_pred / kmax) * msize

    return k_pred

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

def process_waveforms(ecg, ext1, resp_pt) -> tuple[npt.NDArray, npt.NDArray]:
    ecg = np.concatenate(ecg, axis=1).T if len(ecg) > 0 else np.array([])
    if len(ecg) > 0:
        ecg = ecg[:, 0].astype(np.float32)
        ecg -= np.percentile(ecg, 5, axis=0)
        ecg /= np.max(np.abs(ecg), axis=0, keepdims=True)
    resp_pt = np.concatenate(resp_pt, axis=1).T if len(resp_pt) > 0 else np.array([])
    if len(resp_pt) > 0:
        resp_pt = resp_pt[:, 0].astype(np.float32)
        resp_pt -= np.mean(resp_pt, axis=0, keepdims=True)
        resp_pt /= np.max(np.abs(resp_pt), axis=0, keepdims=True)
    ext1 = np.concatenate(ext1, axis=1).T if len(ext1) > 0 else np.array([])
    if len(ext1) > 0:
        ext1 = ext1[:, 1].astype(np.float32)
        ext1[ext1 > 0] = 1

    resp = resp_pt
    card = ecg if len(ecg) > 0 else ext1
    return resp, card

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
    
def load_config(config_file_name: str) -> dict | None:
    path_list = [
        pathlib.Path('/tmp/share/configs'),
        pathlib.Path(__file__).parent / 'configs', 
        ]
    for path in path_list:
        cfg_fullpath = path / config_file_name
        logging.debug(f"Checking for config file at: {cfg_fullpath}")
        if os.path.isfile(cfg_fullpath):
            # We now read these parameters from toml file, so that we won't have to keep restarting the server when we change them.
            logging.info(f"Using the configuration at {cfg_fullpath}")
            try:
                with open(cfg_fullpath) as jf:
                    cfg = rtoml.load(jf)
                return cfg
            except Exception as e:
                logging.error(f"Error loading configuration file {config_file_name}: {e}")
                return None
    logging.error(f"Configuration file {config_file_name} not found in specified paths: {path_list}.")
    return None