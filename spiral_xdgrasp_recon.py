
import ismrmrd
import os
import sys
import logging
import numpy as np
import numpy.typing as npt
import ctypes

import mrdhelper
import time

from scipy.io import loadmat
import GIRF.GIRF as GIRF
import rtoml
import math
import constants
from connection import Connection

# Folder for debug output files
debugFolder = "/tmp/share/debug"

def generate_cardiac_bins(triggers: npt.NDArray[np.double], n_cardiac_bins: int) -> npt.NDArray[np.int32]:
    n_acq = triggers.shape[0]
    trig_idxs = np.nonzero(triggers)[0]
    # Bin k-space with ECG
    # Acquisition window is at most 90% of the shortest heartbeat, divisible by n_cardiac_bins
    acq_window = (int(min(np.diff(trig_idxs))*0.9)//n_cardiac_bins)*n_cardiac_bins
    n_arm_per_bin_per_frame = acq_window//n_cardiac_bins
    n_beat = trig_idxs.shape[0]-1

    cardiac_bins = np.zeros((n_acq,), dtype=int)
    cardiac_bins[:] = -1

    for beat_idx in range(n_beat):
        for bin_idx in range(n_cardiac_bins):
            beat_trig_idx = trig_idxs[beat_idx]
            arm_slc = np.s_[(beat_trig_idx+bin_idx*n_arm_per_bin_per_frame):(beat_trig_idx+(bin_idx+1)*n_arm_per_bin_per_frame)]
            cardiac_bins[arm_slc] = bin_idx

    return cardiac_bins

def generate_respiratory_bins(resp_waveform: npt.NDArray[np.double], n_resp_bins: int, resp_discard: float = 0) -> npt.NDArray[np.int32]:
    n_acq = resp_waveform.shape[0]
    n_resp_discard =  int(n_acq*resp_discard)
    I_resp = np.argsort(resp_waveform)
    I_resp_rev = np.argsort(I_resp)

    n_resp_per_bin = int((n_acq-n_resp_discard)//n_resp_bins)
    n_ext_resp = n_acq - n_resp_per_bin*n_resp_bins

    curr_bin_start = (n_ext_resp//2)
    resp_bins = np.zeros((n_acq,), dtype=int)
    resp_bins[:] = -1

    for bin_i in range(n_resp_bins):
        curr_bin_slc = np.s_[curr_bin_start:(curr_bin_start + n_resp_per_bin)]
        resp_bins[curr_bin_slc] = bin_i
        curr_bin_start += n_resp_per_bin

    resp_bins = resp_bins[I_resp_rev]
    return resp_bins

def generate_cyclic_respiratory_bins(resp_waveform: npt.NDArray[np.double], n_resp_bins: int, resp_discard: float = 0) -> npt.NDArray[np.int32]:
    resp_waveform = resp_waveform.astype(float)
    resp_wf_diff = np.diff(resp_waveform.astype(float), prepend=0)

    I_resp = np.sort(resp_waveform) # To be used for determining intervals
    n_acq = resp_waveform.shape[0]
    n_resp_discard =  int(n_acq*resp_discard)
    max_amp = I_resp[-(n_resp_discard//2+1)]
    min_amp = I_resp[(n_resp_discard//2)]

    # Normalize the waveform and multiply with half nbins -1 (half because we will distinguish positive and negative slope).
    resp_bins = (n_resp_bins//2 - 1)*(resp_waveform - min_amp)/(max_amp-min_amp)
    resp_bins[resp_bins > (n_resp_bins//2-1)] = -1 # Discard bins that are upper extreme
    resp_bins[resp_bins < 0] = -1 # Discard bins that are lower extreme
    resp_bins = np.round(resp_bins).astype(int)  # Rounding should give us the bin numbers.

    resp_bins[(resp_wf_diff < 0) & (resp_bins != -1)] = (n_resp_bins-1)-resp_bins[(resp_wf_diff < 0) & (resp_bins != -1)] # Fix the assignment of negative slope bins

    return resp_bins

def extract_ecg_waveform(wf_list: list, acq_list: list, metadata: ismrmrd.xsd.ismrmrdHeader) -> npt.NDArray[np.int32]:
    '''From a list of mrd waveform objects, extracts the ECG triggers and puts them in the same raster as the acquisition.'''
    n_acq = acq_list.__len__()
    #########################
    # Read ECG triggers
    #########################
    ecg_triggers = []
    wf_ts = 0
    for wf in wf_list:
        if wf.getHead().waveform_id == 0:
            ecg_triggers.append(wf.data[4,:])
            if wf_ts == 0:
                wf_ts = wf.time_stamp
                ecg_sampling_time = wf_list[0].getHead().sample_time_us*1e-6 # [us] -> [s]


    ecg_triggers = np.array(np.concatenate(ecg_triggers, axis=0) > 0, dtype=int)
    time_ecg = np.arange(ecg_triggers.shape[0])*ecg_sampling_time - (acq_list[0].acquisition_time_stamp - wf_ts)*1e-3

    # Make ECG same raster time with the acquisition. For every value 1, make closest time value 1 in the finer grid.
    ecg_trig_times = time_ecg[ecg_triggers == 1]
    acq_sampling_time = metadata.sequenceParameters.TR[0]*1e-3
    time_acq = np.arange(n_acq)*acq_sampling_time

    ecg_acq_trig_times = []
    ecg_acq_trig_idxs = []
    for trg in ecg_trig_times:
        idx = np.searchsorted(time_acq, trg, side="left")
        if idx > 0 and (idx == time_acq.shape[0] or math.fabs(trg - time_acq[idx-1]) < math.fabs(trg - time_acq[idx])):
            ecg_acq_trig_times.append(time_acq[idx-1])
            ecg_acq_trig_idxs.append(idx-1)
        else:
            ecg_acq_trig_times.append(time_acq[idx])
            ecg_acq_trig_idxs.append(idx)

    ecg_acq_trig_idxs = np.array(ecg_acq_trig_idxs, dtype=int)

    ecg_acq_triggers = np.zeros((n_acq,), dtype=int)

    ecg_acq_triggers[ecg_acq_trig_idxs] = 1
    return ecg_acq_triggers

def process(connection: Connection, config, metadata):
    logging.disable(logging.DEBUG)

    # Read config file and import BART
    with open('configs/spiral_xdgrasp_config.toml') as jf:
        cfg = rtoml.load(jf)

    APPLY_GIRF          = cfg['reconstruction']['apply_girf']
    gpu_device          = cfg['reconstruction']['gpu_device']
    BART_TOOLBOX_PATH   = cfg['reconstruction']['BART_TOOLBOX_PATH']
    lambda_tr           = cfg['reconstruction']['reg_lambda_resp']
    lambda_tc           = cfg['reconstruction']['reg_lambda_card']
    n_iter              = cfg['reconstruction']['num_iter']
    arm_discard_pre     = cfg['reconstruction']['arm_discard_pre']
    arm_discard_post    = cfg['reconstruction']['arm_discard_post']
    trigger_source  = cfg['binning']['trigger_source']
    ecg_pt_delay    = cfg['binning']['ecg_pilottone_delay']
    n_resp_bins     = cfg['binning']['n_respiratory_bins']
    n_cardiac_bins  = cfg['binning']['n_cardiac_bins']
    resp_discard    = cfg['binning']['resp_discard']
    fill_small_bins = cfg['binning']['fill_small_bins']


    sys.path.append(f'{BART_TOOLBOX_PATH}/python/')
    import bart # type: ignore

    # Set BART related env vars
    os.environ['BART_TOOLBOX_PATH'] = BART_TOOLBOX_PATH
    os.environ['OPENBLAS_NUM_THREADS'] = '32'
    os.environ['OMP_NUM_THREADS'] = '32'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)

    
    logging.info("Config: \n%s", config)
    logging.info("Metadata: \n%s", metadata)

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
        tRR = 3*1e-6/dt


    ktraj = np.stack((kx, -ky), axis=2)

    # find max ktraj value
    kmax = np.max(np.abs(kx + 1j * ky))

    # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
    ktraj = np.swapaxes(ktraj, 0, 1)

    msize = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0] * cfg['reconstruction']['fov_oversampling'])

    ktraj = 0.5 * (ktraj / kmax) * msize

    pre_discard = traj['param']['pre_discard'][0,0][0,0]
    w = traj['w']
    w = np.reshape(w, (1,w.shape[1]))
    # end = time.perf_counter()
    # logging.debug("Elapsed time during recon prep: %f secs.", end-start)
    # print(f"Elapsed time during recon prep: {end-start} secs.")


    # Discard phase correction lines and accumulate lines until we get fully sampled data
    print("Receiving the data....")
    start_acq = time.time()
    grp = None
    acq_list = []
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
                k_pred = k_pred[:,:,0:2] # Drop the z

                kmax = np.max(np.abs(k_pred[:,:,0] + 1j * k_pred[:,:,1]))
                k_pred = np.swapaxes(k_pred, 0, 1)
                k_pred = 0.5 * (k_pred / kmax) * msize
                ktraj = k_pred
                
            if (arm.idx.kspace_encode_step_1 == 0):
                grp = arm
            
            acq_list.append(arm)
        elif type(arm) is ismrmrd.Waveform:
            wf_list.append(arm)


    end_acq = time.time()
    print(f'Acquiring the data took {end_acq-start_acq} secs.')

    # Extract navigators and generate bins per arm.

    ecg_acq_triggers = extract_ecg_waveform(wf_list, acq_list, metadata)

    resp_waveform = []
    pt_card_triggers = []

    for wf in wf_list:
        if wf.getHead().waveform_id == 1025:
            resp_waveform = wf.data[0,:]
            pt_card_triggers = ((wf.data[2,:]-2**31) != 0).astype(int)#np.round(((wf.data[2,:] - 2**31)/2**31)).astype(int)
            pt_card_derivative_triggers = ((wf.data[4,:]-2**31) != 0).astype(int)#np.round(((wf.data[2,:] - 2**31)/2**31)).astype(int)
            pt_sampling_time = wf.getHead().sample_time_us*1e-6 # [us] -> [s]

    # time_pt = np.arange(resp_waveform.shape[0])*resp_sampling_time
    if trigger_source == 'ecg':
        cardiac_triggers = ecg_acq_triggers
    elif trigger_source == 'pilottone':
        roll_amount = int(ecg_pt_delay//pt_sampling_time)
        cardiac_triggers = np.roll(pt_card_triggers, -roll_amount)
        cardiac_triggers[-roll_amount:] = 0
    elif trigger_source == 'pilottone_derivative':
        roll_amount = int(ecg_pt_delay//pt_sampling_time)
        cardiac_triggers = np.roll(pt_card_derivative_triggers, -roll_amount)
        cardiac_triggers[-roll_amount:] = 0
    else:
        raise ValueError(f"Unknown trigger source: {trigger_source}")

    resp_bins = generate_cyclic_respiratory_bins(resp_waveform, n_resp_bins, resp_discard=resp_discard)
    cardiac_bins = generate_cardiac_bins(cardiac_triggers, n_cardiac_bins)

    data = []
    coord = []
    dcf = []
    arm_counter = 0

    if (acq_list[0].data.shape[1]-pre_discard/2) == w.shape[1]/2:
        # Check if the OS is removed. Should only happen with offline recon.
        ktraj = ktraj[:,::2,:]
        w = w[:,::2]
        pre_discard = int(pre_discard//2)

    for arm in acq_list:
        
        data.append(arm.data[:,pre_discard:])
        coord.append(ktraj[arm_counter,:,:])
        dcf.append(w)
        
        arm_counter += 1
        if arm_counter == n_unique_angles:
            arm_counter = 0


    data = np.array(data)
    data = np.transpose(data, axes=(2, 0, 1))
    coord = np.array(coord, dtype=np.float32)
    coord = np.transpose(coord, axes=(2, 1, 0))

    nk = data.shape[0]
    nc = data.shape[2]

    # Discard requested amount from the start
    acq_sampling_time = metadata.sequenceParameters.TR[0]*1e-3
    n_arm_discard_pre = int(arm_discard_pre//acq_sampling_time)
    n_arm_discard_post= int(arm_discard_post//acq_sampling_time)

    data = data[:,n_arm_discard_pre:-n_arm_discard_post,:]
    coord = coord[:,:,n_arm_discard_pre:-n_arm_discard_post]
    resp_bins = resp_bins[n_arm_discard_pre:-n_arm_discard_post]
    cardiac_bins = cardiac_bins[n_arm_discard_pre:-n_arm_discard_post]

    n_acq = data.shape[1]

    print(f"Discarded {n_arm_discard_pre} arms from beginning and {n_arm_discard_post} from end, {n_acq} arms left.")

    # Sensitivity map estimation
    ksp_all = np.reshape(data, [1, nk, -1, nc])
    traj_all = np.concatenate((coord, np.zeros((1, nk, n_acq))), axis=0)

    _, rtnlinv_sens_32 = bart.bart(2, 'nlinv -a 32 -b 16  -S -d4 -i14 -x 32:32:1 -t',
            traj_all, ksp_all)

    sens_ksp = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(rtnlinv_sens_32, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    sens_ksp = bart.bart(1, f'resize -c 0 {msize*2} 1 {msize*2}', sens_ksp)
    sens = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(sens_ksp, axes=(0,1)), axes=(0, 1)), axes=(0,1))
    sens = bart.bart(1, f'resize -c 0 {msize} 1 {msize}', sens)
    sens = bart.bart(1, 'normalize 8', sens)


    ksp_grps = []
    trj_grps = []
    n_min_arms = n_acq
    n_max_arms = 0

    for r_i in range(n_resp_bins):
        ksp_grps.append([])
        trj_grps.append([])
        for c_i in range(n_cardiac_bins):
            grp_idxs = (resp_bins==r_i) & (cardiac_bins==c_i)
            n_min_arms = min(n_min_arms, np.sum(grp_idxs))
            n_max_arms = max(n_max_arms, np.sum(grp_idxs))

            ksp_grps[r_i].append(data[:,grp_idxs,:])
            trj_grps[r_i].append(np.concatenate((coord[:,:,grp_idxs], np.zeros((1, nk, np.sum(grp_idxs)))), axis=0))


    
    if fill_small_bins is True:
        print(f'There are at least {n_min_arms} arms and at most {n_max_arms} per bin.')
        ksp_grps2 = np.zeros((n_resp_bins, n_cardiac_bins, nk, n_max_arms, nc), dtype=data.dtype)
        trj_grps2 = np.zeros((n_resp_bins, n_cardiac_bins, 3, nk, n_max_arms), dtype=coord.dtype)
        # Second pass for zero filling the missing arms.
        for r_i in range(n_resp_bins):
            for c_i in range(n_cardiac_bins):
                n_carms = ksp_grps[r_i][c_i].shape[1]
                ksp_grps2[r_i, c_i, :, :n_carms, :] = ksp_grps[r_i][c_i]
                trj_grps2[r_i, c_i, :, :, :n_carms] = trj_grps[r_i][c_i]

        ksp_grps = ksp_grps2
        trj_grps = trj_grps2
    else:
        print(f'There are {n_min_arms} arms per bin.')
        # Second pass for discarding the excess arms.
        for r_i in range(n_resp_bins):
            for c_i in range(n_cardiac_bins):
                ksp_grps[r_i][c_i] = ksp_grps[r_i][c_i][:,0:n_min_arms,:]
                trj_grps[r_i][c_i] = trj_grps[r_i][c_i][:,:,0:n_min_arms]
                
    #       0           1           2           3           4           5
    #   READ_DIM,   PHS1_DIM,   PHS2_DIM,   COIL_DIM,   MAPS_DIM,   TE_DIM,
    #       6           7           8           9          10          11 
    #   COEFF_DIM,  COEFF2_DIM, ITER_DIM,   CSHIFT_DIM, TIME_DIM,   TIME2_DIM
    #      12          13          14      
    #   LEVEL_DIM,  SLICE_DIM,  AVG_DIM

    # Put respiratory into TIME_DIM [10] cardiac into TIME2_DIM [11]

    ksp_grps = np.array(ksp_grps) # [nresp, ncard, nch, nsamp, narms]
    ksp_grps = np.transpose(ksp_grps, (2, 3, 4, 0, 1)) # [nsamp, narms, nch, nresp, ncard] 
    ksp_grps = ksp_grps[None,:,:,:,None,None,None,None,None,None,:,:]

    trj_grps = np.array(trj_grps) # [nresp, ncard, nsamp, narms, 2]
    trj_grps = np.transpose(trj_grps, (2, 3, 4, 0, 1)) # [2, nsamp, narms, nresp, ncard]
    trj_grps = trj_grps[:,:,:,None,None,None,None,None,None,None,:,:]

    from scipy.signal.windows import tukey

    ksp_win = tukey(2*nk, alpha=0.1)
    ksp_win = ksp_win[None,(nk):,None,None,None,None,None,None,None,None,None,None]

    img = bart.bart(1, f'pics -g -e -r 0 -R T:1024:0:{lambda_tr} -R T:2048:0:{lambda_tc} -d 4 -i {n_iter} -t',
                trj_grps, ksp_grps*ksp_win*1e3, sens)

    # img = np.reshape(np.squeeze(img), (msize, msize, -1))
    img = np.squeeze(img)
    imgint16 = normalize_convert_toint16(img, metadata)
    connection.send_logging(constants.MRD_LOGGING_INFO, 'Reconstruction is finished. Sending images...')
    for ri in range(n_resp_bins):
        for ci in range(n_cardiac_bins):
            image = process_group(grp, imgint16[None,:,:,ri,ci], ri, ci, metadata)
            connection.send_image(image)

    # Send waveforms back to save them with images
    for wf in wf_list:
        connection.send_waveform(wf)

    connection.send_close()
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

def normalize_convert_toint16(data: np.array, metadata: ismrmrd.xsd.ismrmrdHeader) -> np.array:
    data = np.abs(data)
    
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    # Normalize and convert to int16
    data *= maxVal/data.max()
    data = np.around(data)
    data = data.astype(np.int16)
    return data

def process_group(group, data, resp_i, card_i, metadata):

    data = np.flip(data, axis=(1,))
    data = np.transpose(data, (0, 2, 1))

    # Determine max value (12 or 16 bit)
    BitsStored = 12
    if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
        BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
    maxVal = 2**BitsStored - 1

    # # Normalize and convert to int16
    # data *= maxVal/data.max()
    # data = np.around(data)
    # data = data.astype(np.int16)


    # Format as ISMRMRD image data
    # data has shape [RO PE], i.e. [x y].
    # from_array() should be called with 'transpose=False' to avoid warnings, and when called
    # with this option, can take input as: [cha z y x], [z y x], or [y x]
    image = ismrmrd.Image.from_array(data, acquisition=group, transpose=False)

    image.image_index = resp_i
    image.phase = card_i
    image.set = resp_i

    # Set field of view
    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           str((maxVal+1)/2),
                         'WindowWidth':            str((maxVal+1)),
                         'Keep_image_geometry': 1})

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

