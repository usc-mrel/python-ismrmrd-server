
import ismrmrd
import os
import sys
import logging
import numpy as np
import ctypes

import mrdhelper
import time

from scipy.io import loadmat

import GIRF
import rtoml

# Folder for debug output files
debugFolder = "/tmp/share/debug"


def process(connection, config, metadata):
    logging.disable(logging.WARNING)
    
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

    # Read config file and import BART
    with open('configs/rtspiral_bart_tvrecon_config.toml') as jf:
        cfg = rtoml.load(jf)

    APPLY_GIRF = cfg['reconstruction']['apply_girf']
    gpu_device = cfg['reconstruction']['gpu_device']
    BART_TOOLBOX_PATH   = cfg['reconstruction']['BART_TOOLBOX_PATH']
    n_arm_per_frame     = cfg['reconstruction']['arms_per_frame']
    n_recon_frames      = cfg['reconstruction']['num_recon_frames']
    n_chunks            = cfg['reconstruction']['num_chunks']
    reg_lambda          = cfg['reconstruction']['reg_lambda']
    max_iter            = cfg['reconstruction']['num_iter']

    if n_recon_frames % n_chunks != 0:
        logging.error('Number of frames is not divisible by number of chunks. Check and correct the config.')

    sys.path.append(f'{BART_TOOLBOX_PATH}/python/')
    import bart # type: ignore

    # Set BART related env vars
    os.environ['BART_TOOLBOX_PATH'] = BART_TOOLBOX_PATH
    os.environ['OPENBLAS_NUM_THREADS'] = '32'
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

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
    if APPLY_GIRF:
        # We get dwell time too late from MRD, as it comes with acquisition.
        # So we ask it from the metadata.
        try:
            dt = traj['param']['dt'][0,0][0,0]
        except KeyError:
            dt = 1e-6 # [s]

        patient_position = metadata.measurementInformation.patientPosition.value
        r_PCS2DCS = GIRF.calculate_matrix_pcs_to_dcs(patient_position)

        gx = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(kx, axis=0)))/dt/42.58e6
        gy = 1e3*np.concatenate((np.zeros((1, kx.shape[1])), np.diff(ky, axis=0)))/dt/42.58e6
        g_nom = np.stack((gx, -gy), axis=2)

        sR = {'T': 0.55}
        tRR = 3e-6/dt


    ktraj = np.stack((kx, -ky), axis=2)

    # find max ktraj value
    kmax = np.max(np.abs(kx + 1j * ky))

    # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
    ktraj = np.swapaxes(ktraj, 0, 1)

    msize = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0] * cfg['reconstruction']['fov_oversampling'])

    ktraj = 0.5 * (ktraj / kmax) * msize

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
            if (arm.idx.kspace_encode_step_1 == 0) and APPLY_GIRF:
                r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
                                        [1,    0,   0],  # [RO] = [1 0 0] * [c]
                                        [0,    0,   1]]) # [SL]   [0 0 1] * [s]
                r_GCS2PCS = np.array([arm.phase_dir, -np.array(arm.read_dir), arm.slice_dir])
                r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
                sR['R'] = r_GCS2DCS.dot(r_GCS2RCS)
                k_pred, _ = GIRF.apply_GIRF(g_nom, dt, sR, tRR)
                # k_pred = np.flip(k_pred[:,:,0:2], axis=2) # Drop the z
                k_pred = k_pred[:,:,0:2] # Drop the z

                kmax = np.max(np.abs(k_pred[:,:,0] + 1j * k_pred[:,:,1]))
                k_pred = np.swapaxes(k_pred, 0, 1)
                k_pred = 0.5 * (k_pred / kmax) * msize
                ktraj = k_pred
                
                if (arm.data.shape[1]-pre_discard/2) == w.shape[1]/2:
                    # Check if the OS is removed. Should only happen with offline recon.
                    ktraj = ktraj[:,::2,:]
                    w = w[:,::2]
                    pre_discard = int(pre_discard//2)
                
            if (arm.idx.kspace_encode_step_1 == 0):
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
    print(f'Acquiring the data took {end_acq-start_acq} secs.')

    data = np.array(data)
    data = np.transpose(data, axes=(1, 2, 0))
    coord = np.array(coord, dtype=np.float32)
    coord = np.transpose(coord, axes=(1, 0, 2))

    nc = data.shape[0]
    nk = data.shape[1]

    kdata = np.transpose(data[:,:,:n_recon_frames*n_arm_per_frame], (2, 0, 1))
    kdata = kdata.reshape(n_recon_frames, n_arm_per_frame, nc, nk)            # [n_recon_frames, n_arms, n_ch, n_samples]
    kdata = np.transpose(kdata, (3, 1, 2, 0))                           # [n_samples, n_arms, n_ch n_recon_frames]
    ksp_all  = np.reshape(np.transpose(np.squeeze(kdata), (0, 1, 3, 2)), [1, nk, -1, nc])

    kdata = kdata[None,:,:,:,None,None,None,None,None,None,:]
    kloc = np.transpose(coord[:,:n_recon_frames*n_arm_per_frame,:], (1, 0, 2))
    kloc = kloc.reshape(n_recon_frames, n_arm_per_frame, nk, -1)                # [n_recon_frames, n_arms, n_samples, 2]
    kloc = np.concatenate((kloc, np.zeros((n_recon_frames, n_arm_per_frame, nk, 1))), axis=3)
    kloc = kloc.transpose((3, 2, 1, 0))
    traj_all = np.reshape(kloc, (3, nk, -1))
    kloc = kloc[:,:,:,None,None,None,None,None,None,None,:]

    print('kdata array shape: {}'.format(kdata.shape))
    print('kloc array shape: {}'.format(kloc.shape))


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
    _, rtnlinv_sens_32 = bart.bart(2, 'nlinv -a 32 -b 16 -S -d4 -i13 -x 32:32:1 -t',
            traj_all, ksp_all)

    sens_ksp = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(rtnlinv_sens_32, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    sens_ksp = bart.bart(1, f'resize -c 0 {msize*2} 1 {msize*2}', sens_ksp)
    # sens_ksp = padarray(sens_ksp, [(2*Nx-64)/2 (2*Ny-64)/2]);
    sens = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sens_ksp, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    sens = bart.bart(1, f'resize -c 0 {msize} 1 {msize}', sens)
    # sens = centeredCrop(sens, [Nx/2, Ny/2, 0]);
    sens_map = bart.bart(1, 'normalize 8', sens)
    # sens = sens/vecnorm(sens,2,4)

    # Estimate data scaling outside of the bart for:
    # 1) We can exclude first 5 frames, they seem to dominate the scaling due to steady-state artifacts.
    # 2) We can use smaller amount of data to estimate scale of all the data. Scaling won't change in time dimension (hopefully).
    n_frame_per_chunk = n_recon_frames//n_chunks

    inv_scl = estimate_scale_bart(bart, kloc[:,:,:,:,:,:,:,:,:,:,5:n_frame_per_chunk], kdata[:,:,:,:,:,:,:,:,:,:,5:n_frame_per_chunk], sens_map)
    
    # Divide data into chunks to fit intof a single GPU. Things to care: 
    # 1. First and last frames has artifacts (due to diff operation), so there should be a single frame overlap.
    # 2. Single scale should be used.
    # 3. Track the actual frame number.
    start_time_stamp = grp.getHead().acquisition_time_stamp
    frame_idx = 0
    for chunk_i in range(n_chunks):
        
        if n_chunks == 1: # No chunking, so no overlap.
            slc = np.arange(chunk_i*n_frame_per_chunk, (chunk_i+1)*n_frame_per_chunk)
        elif chunk_i == 0: # First chunk, only overlap at the end.
            slc = np.arange(n_frame_per_chunk+3)
        elif chunk_i == (n_chunks-1): # Last chunk, only overlap at the beginning.
            slc = np.arange(chunk_i*n_frame_per_chunk-3, (chunk_i+1)*n_frame_per_chunk)
        else: # Mid chunks
            slc = np.arange(chunk_i*n_frame_per_chunk-3, (chunk_i+1)*n_frame_per_chunk+3)

        img = bart.bart(1, f'pics -g -m -w {inv_scl} -e -S -R T:1024:1024:{reg_lambda*n_frame_per_chunk} -d 4 -i {max_iter} -t', 
                        kloc[:,:,:,:,:,:,:,:,:,:,slc], 
                        kdata[:,:,:,:,:,:,:,:,:,:,slc], 
                        sens_map)

        img = np.squeeze(img)

        if n_chunks > 1: 
            if chunk_i == 0: # First chunk, only overlap at the end.
                img = img[:,:,:-3]
            elif chunk_i == (n_chunks-1): # Last chunk, only overlap at the beginning.
                img = img[:,:,3:]
            else: # Mid chunks
                img = img[:,:,3:-3]

        # Reshape, process and send images.
        for ii in range(img.shape[2]):
            # update time stamp
            grp.acquisition_time_stamp = start_time_stamp + int(frame_idx*(n_arm_per_frame*metadata.sequenceParameters.TR[0]/2.5))
            image = process_group(grp, img[None,:,:,ii], frame_idx, [], metadata)
            connection.send_image(image)
            frame_idx += 1


    # Send waveforms back to save them with images
    for wf in wf_list:
        connection.send_waveform(wf)

    connection.send_close()
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    print('Reconstruction is finished.')

def estimate_scale_bart(bart, traj, ksp, sens_map) -> float:
    ''' Estimates inverse scaling of the data as BART does. 
    '''
    first_it = bart.bart(1, f'nufft -g -x {sens_map.shape[0]}:{sens_map.shape[1]}:1 -a ', traj, ksp)
    first_it = np.sum(first_it*np.conj(sens_map[:,:,:,:,None,None,None,None,None,None,None]), axis=3)
    med_ = np.median(np.abs(first_it[:]))
    p90_ = np.percentile(np.abs(first_it[:]), 90)
    max_ = np.max(np.abs(first_it[:]))

    scl = p90_ if ((max_-p90_) < 2*(p90_-med_)) else max_
    print(f"Scaling: {scl} (max = {max_}/p90 = {p90_}/median = {med_})\n")
    return scl

def power_iteration_sigpy(sens_map, traj):
    dims = sens_map.shape
    import sigpy as sp
    from sigpy import linop, alg

    device = sp.Device(0)
    sens_map_gpu = sp.to_device(sens_map[:,:,:,None], device=device)
    traj_sp = sp.to_device(np.tile(traj.squeeze()[:,:,:,:,None], (1,1,1,1,dims[2])).transpose(1,2,4,3,0), device=device)
    n_frame = traj_sp.shape[3]
    S = linop.Multiply((dims[0], dims[1], 1, n_frame), sens_map_gpu)

    NUFFTop = linop.NUFFTAdjoint((dims[0], dims[1], dims[2], n_frame), traj_sp*2).H
    NUFFTop.toeplitz = True

    pm = alg.PowerMethod(S.H(NUFFTop.N(S)), sp.to_device(np.random.rand(dims[0], dims[1], 1, n_frame) + 1j*np.random.rand(dims[0], dims[1], 1, n_frame), device=device), max_iter=30)
    print('PM obj created.')

    while not pm.done():
        print(f'{pm.iter} - {pm.max_eig/n_frame}')
        pm.update()

    max_eig = pm.max_eig/n_frame
    del traj_sp
    del sens_map_gpu
    del NUFFTop
    del S
    del pm

    return max_eig

def process_group(group, data, rep, config, metadata):
   

    data = np.abs(np.flip(data, axis=(1,)))
    data = np.transpose(data, (0, 2, 1))

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    # Normalize and convert to int16
    data *= maxVal/data.max()
    data = np.around(data)
    data = data.astype(np.int16)


    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data, acquisition=group, transpose=False)

    image.image_index = rep
    image.repetition = rep

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

