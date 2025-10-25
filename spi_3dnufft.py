
import ismrmrd
import os
import sys
import logging
import numpy as np
import ctypes

import mrdhelper
import time

import GIRF.GIRF as GIRF
import reconutils
from coils import rssq

# Folder for debug output files
debugFolder = "/tmp/share/debug"


def process(connection, config, metadata):
    logging.disable(logging.WARNING)
    
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

    # Read config file and import BART
    cfg = reconutils.load_config('spi_3dnufft.toml')
    if cfg is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    APPLY_GIRF = cfg['reconstruction']['apply_girf']
    gpu_device = cfg['reconstruction']['gpu_device']
    BART_TOOLBOX_PATH   = cfg['reconstruction']['BART_TOOLBOX_PATH']
    metafile_paths = cfg['metafile_paths']

    sys.path.append(f'{BART_TOOLBOX_PATH}/python/')
    import bart # type: ignore

    # Set BART related env vars
    os.environ['BART_TOOLBOX_PATH'] = BART_TOOLBOX_PATH
    os.environ['OPENBLAS_NUM_THREADS'] = '32'
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    # start = time.perf_counter()
    # get the k-space trajectory based on the metadata hash.

    # load the .mat file containing the trajectory
    traj = reconutils.load_trajectory(metadata, metafile_paths)
    if traj is None:
        logging.error("Failed to load trajectory.")
        return

    n_unique_angles = traj['param']['repetitions'][0,0][0,0]

    kx = traj['kx'][:,:]
    ky = traj['ky'][:,:]
    kz = traj['kz'][:,:]

    dt = traj['param']['dt'][0,0][0,0]
    msize_inplane = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0] * cfg['reconstruction']['fov_oversampling'])
    msize_slab = np.int16(10 * traj['param']['fov'][0,0][0,1] / traj['param']['spatial_resolution'][0,0][0,0] * cfg['reconstruction']['fov_oversampling'])
    delta_r = traj['param']['spatial_resolution'][0,0][0,0]
    # Prepare gradients and variables if GIRF is requested. 
    # Unfortunately, we don't know rotations until the first data, so we can't prepare them yet.
    if APPLY_GIRF:
        gx = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(kx, axis=0)))/dt/42.58e6
        gy = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(ky, axis=0)))/dt/42.58e6
        gz = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(kz, axis=0)))/dt/42.58e6
        g_nom = np.stack((gx, -gy, gz), axis=2)

    ktraj = np.stack((kx, -ky, kz), axis=2)

    # find max ktraj value
    kmax = np.max(np.linalg.vector_norm(ktraj, axis=2, ord=2))
    # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
    ktraj = np.swapaxes(ktraj, 0, 1)
    ktraj = 0.5 * (ktraj / kmax) * msize_inplane

    # nchannel = metadata.acquisitionSystemInformation.receiverChannels
    pre_discard = traj['param']['pre_discard'][0,0][0,0]
    w = traj['w']
    w = np.reshape(w, (1,w.shape[1]))
    # end = time.perf_counter()
    # logging.debug("Elapsed time during recon prep: %f secs.", end-start)
    # print(f"Elapsed time during recon prep: {end-start} secs.")


    # Discard phase correction lines and accumulate lines until we get fully sampled data
    arm_counter = 0

    start_acq = time.time()
    data = []
    coord = []
    dcf = []
    grp = None
    wf_list = []

    for arm in connection:
        # start_iter = time.perf_counter()
        if arm is None:
            break

        if type(arm) is ismrmrd.Acquisition:

            # First arm came, if GIRF is requested, correct trajectories and reupload.
            if (arm.getHead().scan_counter == 1) and APPLY_GIRF:
                r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                                        [1,    0,   0],  # [RO] = [1 0 0] * [c]
                                        [0,    0,   1]]) # [SL]   [0 0 1] * [s]
                r_GCS2PCS = np.array([arm.phase_dir, -np.array(arm.read_dir), arm.slice_dir])
                r_PCS2DCS = GIRF.calculate_matrix_pcs_to_dcs(metadata.measurementInformation.patientPosition.value)
                r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
                sR = r_GCS2DCS.dot(r_GCS2RCS)
                tRR = 3e-6/dt
                k_pred, _ = GIRF.apply_GIRF(g_nom, dt, sR, tRR=tRR)
                
                kmax = np.max(np.linalg.vector_norm(k_pred, axis=2, ord=2))
                k_pred = np.swapaxes(k_pred, 0, 1)
                k_pred = 0.5 * (k_pred / kmax) * msize_inplane
                ktraj = k_pred

            if (arm.getHead().scan_counter == 1):
                grp = arm

            data.append(arm.data[:,pre_discard:])
            coord.append(ktraj[arm_counter,:,:])
            dcf.append(w[0,:])

            arm_counter += 1
            if arm_counter == n_unique_angles:
                arm_counter = 0

        elif type(arm) is ismrmrd.Waveform:
            wf_list.append(arm)


    end_acq = time.time()
    logging.info(f'Acquiring the data took {end_acq-start_acq} secs.')

    ################################################################################
    # TODO: Fix the data shapes for 3D SPI here.
    data = np.array(data)
    data = np.transpose(data, axes=(1, 2, 0))
    coord = np.array(coord, dtype=np.float32)
    coord = np.transpose(coord, axes=(1, 0, 2))

    nc = data.shape[0]
    nk = data.shape[1]

    kdata = np.transpose(data, (2, 0, 1))           # [ndisks*n_arms, n_ch, n_samples]
    kdata = np.transpose(kdata, (2, 0, 1))       # [n_samples, ndisks*n_arms, n_ch]
    # ksp_all  = np.reshape(np.transpose(np.squeeze(kdata), (0, 1, 3, 2)), [1, nk, -1, nc])

    kdata = kdata[None,:,:,:,None,None,None,None,None,None,None]
    kloc = np.transpose(coord, (1, 0, 2)) # [ndisks*n_arms, n_samples, 2]
    kloc = kloc.transpose((2, 1, 0))
    kloc = kloc[:,:,:,None,None,None,None,None,None,None,None]

    logging.info('kdata array shape: {}'.format(kdata.shape))
    logging.info('kloc array shape: {}'.format(kloc.shape))


    #       0           1           2           3           4           5
    #   READ_DIM,   PHS1_DIM,   PHS2_DIM,   COIL_DIM,   MAPS_DIM,   TE_DIM,
    #       6           7           8           9          10          11 
    #   COEFF_DIM,  COEFF2_DIM, ITER_DIM,   CSHIFT_DIM, TIME_DIM,   TIME2_DIM
    #      12          13          14      
    #   LEVEL_DIM,  SLICE_DIM,  AVG_DIM


    ################################################################################
    # CS Reconstruction
    #
    # TODO: nothing limits the time frame we add here.
    # RSSQ for now.
    # _, rtnlinv_sens_32 = bart.bart(2, 'nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t',
    #         traj_all, ksp_all)

    # sens_ksp = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(rtnlinv_sens_32, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    # sens_ksp = bart.bart(1, f'resize -c 0 {msize*2} 1 {msize*2}', sens_ksp)
    # # sens_ksp = padarray(sens_ksp, [(2*Nx-64)/2 (2*Ny-64)/2]);
    # sens = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sens_ksp, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    # sens = bart.bart(1, f'resize -c 0 {msize} 1 {msize}', sens)
    # # sens = centeredCrop(sens, [Nx/2, Ny/2, 0]);
    # sens_map = bart.bart(1, 'normalize 8', sens)
    # sens = sens/vecnorm(sens,2,4)


    # Do NUFFT
    img = bart.bart(1, f"nufft -i -g -t -m {cfg['reconstruction']['num_iter']} -x {msize_inplane}:{msize_inplane}:{msize_inplane}", kloc, kdata)
    img = rssq(img, axis=3)
    maxVal = 2**12 - 1
    img *= maxVal/img.max()
    img = np.around(img)
    # Reshape, process and send images.

    for ii in range(img.shape[2]):
        # update time stamp
        image = process_group(grp, img[None,:,:,ii], ii, 0, img.shape[2], delta_r, metadata)
        connection.send_image(image)

    # for ii in range(img.shape[1]):
    #     # update time stamp
    #     image = process_group(grp, (img[:,ii,:])[None,:,:], ii, 1, traj['param']['spatial_resolution'][0,0][0,0], metadata)
    #     connection.send_image(image)

    # for ii in range(img.shape[0]):
    #     # update time stamp
    #     image = process_group(grp, (img[ii,:,:])[None,:,:], ii, 2, traj['param']['spatial_resolution'][0,0][0,0], metadata)
    #     connection.send_image(image)

    # Send waveforms back to save them with images
    # for wf in wf_list:
    #     connection.send_waveform(wf)

    connection.send_close()
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    logging.info('Reconstruction is finished.')


def process_group(group, data, rep, image_series_idx, Nslc, resolution, metadata):
   
    data = np.abs(np.flip(data, axis=(1,)))
    data = np.transpose(data, (0, 2, 1))

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    # Normalize and convert to int16
    data = data.astype(np.int16)


    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data, acquisition=group, transpose=False)

    image.image_index = rep+1
    image.image_series_index = image_series_idx
    image.slice = rep

    # Set field of view
    # Field-of-view in the metadata is as given in .seq file, logical coordinates in mm.
    # Similarly FoV in image header is, physical size (in mm) in each of the 3 dimensions in the image
    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(resolution))

    # Set slice position
    # Center of the excited volume, in (left, posterior, superior) (LPS) coordinates relative to isocenter in millimeters. 
    # NB this is different than DICOM’s ImageOrientationPatient, which defines the center of the first (typically top-left) voxel.
    image.position[1] = ctypes.c_float(image.position[1] + resolution * (rep-Nslc//2))

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

