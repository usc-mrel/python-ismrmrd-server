
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    try:
        logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

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

    for group in process_data(connection):
        if isinstance(group[0], ismrmrd.Image):
            logging.info("Processing an image")
            image = process_image(group[0], config, metadata)
        else:
            logging.info("Processing a group of k-space data")
            image = process_raw(group, config, metadata)

        logging.debug("Sending image to client:\n%s", image)
        connection.send_image(image)


# Continuously parse incoming data parsed from MRD messages
def process_data(iterable):
    group = []
    try:
        for item in iterable:
            if item is None:
                break

            elif isinstance(item, ismrmrd.Acquisition):
                if (not item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA)):
                    group.append(item)

                if (item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)):
                    yield group
                    group = []

            elif isinstance(item, ismrmrd.Image):
                group.append(item)
                yield group
                group = []

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

    finally:
        iterable.send_close()


def process_raw(group, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Sort by line number (incoming data may be interleaved)
    lin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    logging.debug("Incoming lin ordering: " + str(lin))

    group.sort(key = lambda acq: acq.idx.kspace_encode_step_1)
    sortedLin = [acquisition.idx.kspace_encode_step_1 for acquisition in group]
    logging.debug("Sorted lin ordering: " + str(sortedLin))

    # Format data into single [cha RO PE] array
    data = [acquisition.data for acquisition in group]
    data = np.stack(data, axis=-1)

    logging.debug("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Fourier Transform
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data)
    data = fft.ifftshift(data, axes=(1, 2))

    # Sum of squares coil combination
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

    # Invert image contrast
    data = 32767-data
    data = np.abs(data)
    data = data.astype(np.int16)
    np.save(debugFolder + "/" + "imgInverted.npy", data)

    # Remove phase oversampling
    nRO = np.size(data,0)
    data = data[int(nRO/4):int(nRO*3/4),:]
    logging.debug("Image without oversampling is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgCrop.npy", data)

    # Format as ISMRMRD image data
    image = ismrmrd.Image.from_array(data, acquisition=group[0])
    image.image_index = 1

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})
    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image


def process_image(image, config, metadata):
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    logging.debug("Incoming image data of type %s", ismrmrd.get_dtype_from_data_type(image.data_type))

    # Extract image data itself
    data = image.data
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

    # Create new MRD instance for the inverted image
    imageInverted = ismrmrd.Image.from_array(data.transpose())
    data_type = imageInverted.data_type

    np.save(debugFolder + "/" + "imgInvertedMrd.npy", imageInverted.data)

    # Copy the fixed header information
    oldHeader = image.getHead()
    oldHeader.data_type = data_type
    imageInverted.setHead(oldHeader)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})
    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    imageInverted.attribute_string = xml

    return imageInverted