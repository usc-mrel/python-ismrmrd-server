import connection
import logging
import ismrmrd
import time
import os
from datetime import datetime

def process(conn: connection.Connection, config, metadata):
    logging.disable(logging.DEBUG)
    
    logging.info("Config: \n%s", config)

    acq: ismrmrd.Acquisition | ismrmrd.Waveform | None
    acq_counter = 0
    wf_counter = 0
    startarm = time.perf_counter()

    output_file_path = os.path.join(conn.savedataFolder)
    try:
        if (metadata.measurementInformation.protocolName != ""):
            output_file_path = os.path.join(conn.savedataFolder, f"noise_meas_MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_{metadata.measurementInformation.protocolName}_{datetime.now().strftime('%H%M%S')}.h5")
        else:
            output_file_path = os.path.join(conn.savedataFolder, f"noise_meas_MID{int(metadata.measurementInformation.measurementID.split('_')[-1]):05d}_UnknownProtocol_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.h5")
    except Exception as e:
        logging.error("Error constructing output file path: %s", e)
        # Fallback to a default file name if there's an error
        output_file_path = os.path.join(conn.savedataFolder, f"noise_meas_default_{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.h5")

    with ismrmrd.Dataset(output_file_path, create_if_needed=True) as dset:
        
        if metadata != '':
            try:
                dset.write_xml_header(ismrmrd.xsd.ToXML(metadata))
            except Exception as e:
                logging.error("Error writing XML header: %s", e)
        for acq in conn:
            if acq is None:
                break
            
            if type(acq) is ismrmrd.Acquisition:
                acq_counter += 1
                dset.append_acquisition(acq)

                if acq_counter % 1000 == 0:
                    endarm = time.perf_counter()
                    logging.info("Received acquisition %d at time %f secs.", acq_counter, endarm-startarm)

            elif type(acq) is ismrmrd.Waveform:
                wf_counter += 1
                dset.append_waveform(acq)


    endarm = time.perf_counter()

    conn.send_close()
    logging.info("Received %d acquisitions and %d waveforms.", acq_counter, wf_counter)
    logging.info("Total time for receiving data: %f secs.", endarm - startarm)
