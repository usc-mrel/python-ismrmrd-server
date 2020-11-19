
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
import ctypes
import re

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    imgGroup = []
    waveformGroup = []
    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):
                # Accumulate all imaging readouts in a group
                if (not item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION) and
                    not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)):
                    acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                    logging.info("Processing a group of k-space data")
                    image = process_raw(acqGroup, config, metadata)
                    connection.send_image(image)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # Only process magnitude images -- send phase images back without modification
                if item.image_type is ismrmrd.IMTYPE_MAGNITUDE:
                    imgGroup.append(item)
                else:
                    tmpMeta = ismrmrd.Meta.deserialize(item.attribute_string)
                    tmpMeta['Keep_image_geometry']    = 1
                    item.attribute_string = tmpMeta.serialize()

                    connection.send_image(item)
                    continue

                # When this criteria is met, run process_group() on the accumulated
                # data, which returns images that are sent back to the client.
                # TODO: logic for grouping images
                if False:
                    logging.info("Processing a group of images")
                    image = process_image(imgGroup, config, metadata)
                    connection.send_image(image)
                    imgGroup = []

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            image = process_raw(acqGroup, config, metadata)
            connection.send_image(image)
            acqGroup = []

        if len(imgGroup) > 0:
            logging.info("Processing a group of images (untriggered)")
            image = process_image(imgGroup, config, metadata)
            connection.send_image(image)
            imgGroup = []

    finally:
        connection.send_close()


def process_raw(group, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha PE RO phs] array
    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    phs = [acquisition.idx.phase                for acquisition in group]

    # Use the zero-padded matrix size
    data = np.zeros((group[0].data.shape[0], 
                     metadata.encoding[0].encodedSpace.matrixSize.y, 
                     metadata.encoding[0].encodedSpace.matrixSize.x, 
                     max(phs)+1), 
                    group[0].data.dtype)

    rawHead = [None]*(max(phs)+1)

    for acq, lin, phs in zip(group, lin, phs):
        if (lin < data.shape[1]) and (phs < data.shape[3]):
            # TODO: Account for asymmetric echo in a better way
            data[:,lin,-acq.data.shape[1]:,phs] = acq.data

            # center line of k-space is encoded in user[5]
            if (rawHead[phs] is None) or (np.abs(acq.getHead().idx.kspace_encode_step_1 - acq.getHead().idx.user[5]) < np.abs(rawHead[phs].idx.kspace_encode_step_1 - rawHead[phs].idx.user[5])):
                rawHead[phs] = acq.getHead()

    # Flip matrix in RO/PE to be consistent with ICE
    data = np.flip(data, (1, 2))

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Fourier Transform
    data = fft.fftshift( data, axes=(1, 2))
    data = fft.ifft2(    data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
    # Data will be [PE RO phs]
    data = np.abs(data)
    data = np.square(data)
    data = np.sum(data, axis=0)
    data = np.sqrt(data)

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Remove readout oversampling
    offset = int((data.shape[1] - metadata.encoding[0].reconSpace.matrixSize.x)/2)
    data = data[:,offset:offset+metadata.encoding[0].reconSpace.matrixSize.x]

    # Remove phase oversampling
    offset = int((data.shape[0] - metadata.encoding[0].reconSpace.matrixSize.y)/2)
    data = data[offset:offset+metadata.encoding[0].reconSpace.matrixSize.y,:]

    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Format as ISMRMRD image data
    imagesOut = []
    for phs in range(data.shape[2]):
        # Create new MRD instance for the processed image
        # NOTE: from_array() takes input data as [x y z coil], which is
        # different than the internal representation in the "data" field as
        # [coil z y x], so we need to transpose
        tmpImg = ismrmrd.Image.from_array(data[...,phs].transpose())

        # Set the header information
        tmpImg.setHead(update_img_header_from_raw(tmpImg.getHead(), rawHead[phs]))
        tmpImg.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                                ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))
        tmpImg.image_index = phs

        # Set ISMRMRD Meta Attributes
        tmpMeta = ismrmrd.Meta()
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'
        tmpMeta['Keep_image_geometry']    = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        tmpImg.attribute_string = xml
        imagesOut.append(tmpImg)

    # Call process_image() to invert image contrast
    imagesOut = process_image(imagesOut, config, metadata)

    return imagesOut


def process_image(images, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Processing data with %d images of type %s", len(images), ismrmrd.get_dtype_from_data_type(images[0].data_type))

    # Extract image data into a 5D array of size [img cha z y x]
    data = np.stack([img.data                              for img in images])
    head = [img.getHead()                                  for img in images]
    meta = [ismrmrd.Meta.deserialize(img.attribute_string) for img in images]

    # Reformat data to the more intuitive [x y z cha img]
    data = data.transpose()

    # Reformat data again to [y x z cha img], i.e. [row col] for the first two
    # dimensions.  Note we will need to undo this later prior to sending back
    # to the client
    data = data.transpose((1, 0, 2, 3, 4))

    # Display MetaAttributes for first image
    logging.debug("MetaAttributes[0]: %s", ismrmrd.Meta.serialize(meta[0]))

    # Optional serialization of ICE MiniHeader
    if 'IceMiniHead' in meta[0]:
        logging.debug("IceMiniHead[0]: %s", base64.b64decode(meta[0]['IceMiniHead']).decode('utf-8'))

    logging.debug("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    # Normalize and convert to int16
    data = data.astype(np.float64)
    data *= 32767/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Invert image contrast
    data = 32767-data
    data = np.abs(data)
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgInverted.npy", data)

    # Reformat data from [row col z cha img] back to [x y z cha img] before sending back to client
    data = data.transpose((1, 0, 2, 3, 4))

    currentSeries = 0

    # Re-slice back into 2D images
    imagesOut = [None] * data.shape[-1]
    for iImg in range(data.shape[-1]):
        # Create new MRD instance for the inverted image
        # NOTE: from_array() takes input data as [x y z coil], which is
        # different than the internal representation in the "data" field as
        # [coil z y x].  However, we already transposed this data when
        # extracting it earlier.
        imagesOut[iImg] = ismrmrd.Image.from_array(data[...,iImg])
        data_type = imagesOut[iImg].data_type

        # Create a copy of the original fixed header and update the data_type
        # (we changed it to int16 from all other types)
        oldHeader = head[iImg]
        oldHeader.data_type = data_type

        # Increment series number when flag detected (i.e. follow ICE logic for splitting series)
        oldHeader.image_series_index = currentSeries
        if get_meta_value(meta[iImg], 'IceMiniHead') is not None:
            if extract_minihead_bool_param(base64.b64decode(meta[iImg]['IceMiniHead']).decode('utf-8'), 'BIsSeriesEnd') is True:
                currentSeries += 1

        imagesOut[iImg].setHead(oldHeader)

        # Create a copy of the original ISMRMRD Meta attributes and update
        tmpMeta = meta[iImg]
        tmpMeta['DataRole']               = 'Image'
        tmpMeta['ImageProcessingHistory'] = ['FIRE', 'PYTHON']
        tmpMeta['WindowCenter']           = '16384'
        tmpMeta['WindowWidth']            = '32768'
        tmpMeta['Keep_image_geometry']    = 1

        xml = tmpMeta.serialize()
        logging.debug("Image MetaAttributes: %s", xml)
        logging.debug("Image data has %d elements", imagesOut[iImg].data.size)

        imagesOut[iImg].attribute_string = xml

    return imagesOut

def update_img_header_from_raw(imgHead, rawHead):
    if rawHead is None:
        return imgHead

    imgHead.version                = rawHead.version
    imgHead.flags                  = rawHead.flags
    imgHead.measurement_uid        = rawHead.measurement_uid

    # # These fields are not translated from the raw header, but filled in
    # # during image creation by from_array
    # imgHead.data_type            = 
    # imgHead.matrix_size          = 
    # imgHead.field_of_view        = 

    imgHead.image_type             = ismrmrd.IMTYPE_MAGNITUDE

    imgHead.channels               = 1
    imgHead.position               = rawHead.position
    imgHead.read_dir               = rawHead.read_dir
    imgHead.phase_dir              = rawHead.phase_dir
    imgHead.slice_dir              = rawHead.slice_dir
    imgHead.patient_table_position = rawHead.patient_table_position

    imgHead.average                = rawHead.idx.average
    imgHead.slice                  = rawHead.idx.slice
    imgHead.contrast               = rawHead.idx.contrast
    imgHead.phase                  = rawHead.idx.phase
    imgHead.repetition             = rawHead.idx.repetition
    imgHead.set                    = rawHead.idx.set

    imgHead.acquisition_time_stamp = rawHead.acquisition_time_stamp
    imgHead.physiology_time_stamp  = rawHead.physiology_time_stamp

    # Defaults, to be updated by the user
    imgHead.image_type         = ismrmrd.IMTYPE_MAGNITUDE
    imgHead.image_index        = 0
    imgHead.image_series_index = 0

    imgHead.user_float             = rawHead.user_float
    imgHead.user_int               = rawHead.user_int

    return imgHead

def extract_minihead_bool_param(miniHead, name):
    # Extract a bool parameter from the serialized text of the ICE MiniHeader
    # Note: if missing, return false (following ICE logic)
    expr = r'(?<=<ParamBool."' + name + r'">{)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return False
    else:
        if res.group(0).strip().lower() == '"true"'.lower():
            return True
        else:
            return False

def extract_minihead_long_param(miniHead, name):
    # Extract a long parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamLong."' + name + r'">{)\s*\d*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    elif res.group(0).isspace():
        return 0
    else:
        return int(res.group(0))

def extract_minihead_string_param(miniHead, name):
    # Extract a string parameter from the serialized text of the ICE MiniHeader
    expr = r'(?<=<ParamString."' + name + r'">{)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None
    else:
        return res.group(0).strip()

def get_meta_value(meta, key):
    # Get a value from MRD Meta Attributes (blank if key not found)
    if key in meta.keys():
        return meta[key]
    else:
        return None
