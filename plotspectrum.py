import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import ismrmrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import connection

matplotlib.use("Agg")  # Use non-interactive backend for saving figures

from scipy.fft import fftfreq, fftshift, fft, ifftshift

import reconutils

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(conn: connection.Connection, config, metadata):
    logging.disable(logging.DEBUG)

    logging.info("Config: \n%s", config)

    cfg = reconutils.load_config("plotspectrum.toml")
    if cfg is None:
        logging.error("Failed to load configuration file.")
        return

    skip_TR = cfg["skip_TR"]
    ch_i = cfg["channel"]

    # Set plot
    # Properties of report image to create
    width = 512  # pixels
    height = 512  # pixels

    # Create a blank image with white background
    plt.style.use("dark_background")
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    line1 = None

    rep_counter = 0
    acq_counter = 0

    # Create thread-safe queue and stop event
    data_queue = queue.Queue(
        maxsize=100
    )  # Limit queue size to prevent excessive memory usage
    stop_event = threading.Event()

    # Use ThreadPoolExecutor for better resource management
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="DataAcquisition"
    ) as executor:
        # Submit the data acquisition worker
        if cfg["save_raw"]:
            output_file_path = os.path.join(conn.savedataFolder)
            if metadata.measurementInformation.protocolName != "":
                output_file_path = os.path.join(
                    conn.savedataFolder,
                    f"meas_MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_{metadata.measurementInformation.protocolName}_{datetime.now().strftime('%H%M%S')}.h5",
                )
            else:
                output_file_path = os.path.join(
                    conn.savedataFolder,
                    f"meas_MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_UnknownProtocol_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.h5",
                )
            future = executor.submit(
                reconutils.data_acquisition_with_save_worker,
                conn,
                data_queue,
                stop_event,
                output_file_path,
                metadata,
            )
        else:
            future = executor.submit(
                reconutils.data_acquisition_worker, conn, data_queue, stop_event
            )

        wf_list = []
        arm: ismrmrd.Acquisition | ismrmrd.Waveform | None

        try:
            while True:
                try:
                    # Use blocking get with timeout to allow checking future status
                    arm = data_queue.get(timeout=0.1)
                except queue.Empty:
                    # No data available, check if acquisition worker is still running
                    if future.done():
                        # Check if there was an exception
                        try:
                            future.result()  # This will raise any exception that occurred
                        except Exception as e:
                            logging.error(f"Data acquisition worker failed: {e}")
                        break
                    continue

                # Signal that we've processed this item
                data_queue.task_done()

                if arm is None:
                    # End of data signal
                    break
                elif type(arm) is ismrmrd.Waveform:
                    # Accumulate waveforms to send at the end
                    wf_list.append(arm)
                    continue
                elif type(arm) is not ismrmrd.Acquisition:
                    continue

                acq_counter += 1
                start_iter = time.perf_counter()

                if arm.scan_counter % 1000 == 0:
                    logging.info("Processing acquisition %d", arm.scan_counter)

                if arm.scan_counter == 1:
                    # Display the blank image to set image size
                    dt = arm.sample_time_us*1e-6
                    freqs = fftshift(fftfreq(arm.data.shape[1], d=dt))
                    (line1,) = ax.plot(freqs*1e-3, np.zeros_like(freqs))
                    plt.xlabel("Frequency [kHz]")
                    if ch_i < 0:
                        plt.title("RSS magnitude")
                    else:
                        plt.title(f"Channel {metadata.acquisitionSystemInformation.coilLabel[ch_i].coilName} magnitude")
                    
                    ax.set_ylim(0, 14)

                if ((arm.scan_counter) % skip_TR) == 0:
                    spec = cfft(arm.data, axis=1)
                    if ch_i < 0:
                        y_data = rssq(spec, axis=0)
                    else:
                        y_data = np.abs(spec[ch_i])
                    image = process_spectrum(fig, line1, y_data, arm, rep_counter)
                    logging.debug("Sending image to client:\n%s", image)
                    conn.send_image(image)

                    rep_counter += 1

                    end_iter = time.perf_counter()
                    logging.debug(
                        "Elapsed time for per iteration: %f secs.",
                        end_iter - start_iter,
                    )

        except KeyboardInterrupt:
            logging.info("Received interrupt signal, stopping acquisition...")
            stop_event.set()

        except Exception as e:
            logging.error(f"Error in main processing loop: {e}")
            stop_event.set()

        finally:
            # Ensure clean shutdown
            stop_event.set()

            # Wait for the acquisition task to complete with timeout
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logging.warning(
                    f"Error while waiting for acquisition task to complete: {e}"
                )

    conn.send_close()
    logging.info("Reconstruction is finished.")


def process_spectrum(fig, line1, y_data, acq, rep_counter):

    line1.set_ydata(y_data)
    
    # Get current y-axis limits
    ax = fig.gca()
    current_ylim = ax.get_ylim()
    
    # Get the actual data limits
    data_max = np.max(y_data)
    
    # Add some padding (5%) to the data limits for better visualization
    data_max_padded = data_max + 0.05 * data_max
    
    # Check if we need to update y-limits (20% threshold)
    update_ylim = False
    
    # Check if current limits are significantly different from data limits
    if current_ylim[1] == 0:  # First time or zero range
        update_ylim = True
    else:
        # Calculate relative change needed
        max_change = abs(data_max_padded - current_ylim[1]) / current_ylim[1]
        
        # Update if either limit needs to change by more than 20%
        if max_change > 0.2:
            update_ylim = True
        
        # Also update if data goes outside current limits
        if data_max > current_ylim[1]:
            update_ylim = True
    
    if update_ylim:
        ax.set_ylim(0, data_max_padded)
        logging.debug(f"Updated y-limits to [0, {data_max_padded:.3f}] for rep {rep_counter}")
    
    # Invoke a draw to create a buffer we can copy the pixel data from
    fig.canvas.draw()

    imgReport = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
    w, h = fig.canvas.get_width_height()
    imgReport = imgReport.reshape((int(h), int(w), -1))

    # plt.imsave(os.path.join(debugFolder, "report.png"), imgReport)

    # Conversion as per CCIR 601 (https://en.wikipedia.org/wiki/Luma_(video)
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    # Must convert to grayscale float32 or uint16 (float64s are not supported)
    imgGray = rgb2gray(np.asarray(imgReport))
    imgGray = imgGray.astype(np.float32)

    imagesOut = []

    # Create new MRD instance for the report image
    # data has shape [PE RO phs], i.e. [y x].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    mrdImg = ismrmrd.Image.from_array(imgGray, transpose=False)

    # Set the minimum appropriate ImageHeader information without using a reference acquisition/image as a starting point
    # Note mrdImg.getHead().matrix_size should be used instead of the convenience mrdImg.matrix_size because the latter
    # returns a transposed result.  See: https://github.com/ismrmrd/ismrmrd-python/pull/54
    mrdImg.field_of_view = (
        mrdImg.getHead().matrix_size[0],
        mrdImg.getHead().matrix_size[1],
        mrdImg.getHead().matrix_size[2],
    )

    # Set image orientation dimensions.  Note that the default initialized values (0,0,0) are invalid
    # because they are not unit vectors
    mrdImg.read_dir = (1, 0, 0)
    mrdImg.phase_dir = (0, 1, 0)
    mrdImg.slice_dir = (0, 0, 1)

    # mrdImg.position is optional, but is relative to the patient_table_position
    # Setting patient_table_position is recommended, otherwise the report may
    # significantly shifted from other images in the series_
    mrdImg.patient_table_position = acq.patient_table_position

    # Optional, but recommended.  Default value (0) corresponds to midnight
    mrdImg.acquisition_time_stamp = acq.acquisition_time_stamp

    # Default value of image_type (0) is invalid
    mrdImg.image_type = ismrmrd.IMTYPE_MAGNITUDE

    # Use a different image_series_index to have a separate series than the main
    # images.  Absolute value does not matter, but images with the same
    # image_series_index are grouped together in the same DICOM SeriesNumber
    mrdImg.image_series_index = 0

    # DICOM InstanceNumber. Should be incremented if multiple images in a series
    mrdImg.image_index = rep_counter
    mrdImg.repetition = rep_counter

    # Set MRD MetaAttributes
    tmpMeta = ismrmrd.Meta()
    tmpMeta["DataRole"] = "Image"
    tmpMeta["ImageProcessingHistory"] = ["FIRE", "PYTHON"]
    tmpMeta["Keep_image_geometry"] = 1

    # Add image orientation directions to MetaAttributes
    # Note that DICOM image orientation is in LPS coordinates, so if another set of image directional
    # cosines are chosen, they may be flipped/rotated to bring them into LPS coordinate space
    tmpMeta["ImageRowDir"] = [
        "{:.18f}".format(mrdImg.read_dir[0]),
        "{:.18f}".format(mrdImg.read_dir[1]),
        "{:.18f}".format(mrdImg.read_dir[2]),
    ]
    tmpMeta["ImageColumnDir"] = [
        "{:.18f}".format(mrdImg.phase_dir[0]),
        "{:.18f}".format(mrdImg.phase_dir[1]),
        "{:.18f}".format(mrdImg.phase_dir[2]),
    ]

    xml = tmpMeta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    mrdImg.attribute_string = xml
    imagesOut.append(mrdImg)

    return imagesOut

def cfft(data, axis):
    '''Centered FFT.'''
    return fftshift(fft(ifftshift(data, axis), None, axis), axis)

def rssq(data, axis):
    '''Root sum of squares along the given axis.'''
    return np.sqrt(np.sum(np.abs(data)**2, axis=axis))