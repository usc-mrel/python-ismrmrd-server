
import ismrmrd
import logging
import numpy as np
import numpy.typing as npt
import ctypes
import mrdhelper
import time
import os
import rtoml

from scipy.io import loadmat
import sigpy as sp
from sigpy.linop import NUFFT, FiniteDifference
from sigpy import fourier
from sigpy.app import MaxEig
from sigpy.mri.app import EspiritCalib

from tcr_utils import online_STCR_ISTA_2_timed, calculate_csm_walsh_gpu
import GIRF
import coils
from einops import rearrange
import connection

from dmd_utils import dmd, dmd_filt

# Folder for debug output files
debugFolder = "/tmp/share/debug"


def process(conn: connection.Connection, config, metadata):

    end_iter = 0 # dummy define

    logging.disable(logging.DEBUG)
    
    logging.info("Config: \n%s", config)
    # logging.info("Metadata: \n%s", metadata)

    # We now read these parameters from toml file, so that we won't have to keep restarting the server when we change them.
    logging.info('''Loading and applying file configs/rtspiral_vs_config.toml''')
    with open('configs/rtspiral_stcr.toml') as jf:
        cfg = rtoml.load(jf)

    n_arm_per_frame = cfg['reconstruction']['arms_per_frame']
    APPLY_GIRF      = cfg['reconstruction']['apply_girf']
    gpu_device      = cfg['reconstruction']['gpu_device']
    lambdat         = cfg['reconstruction']['lambdat']
    lambdas         = cfg['reconstruction']['lambdas']
    alg_type         = cfg['reconstruction']['alg_type']
    metafile_paths =  cfg['metafile_paths']

    logging.info(f'''
                 ================================================================
                 Arms per frame: {n_arm_per_frame}
                 Apply GIRF?: {APPLY_GIRF}
                 GPU Device: {gpu_device}
                 =================================================================''')
    
    # start = time.perf_counter()
    # get the k-space trajectory based on the metadata hash.
    #traj_name = metadata.userParameters.userParameterString[1].value[:32] # Get the first 32 chars, because a bug sometimes causes this field to have /OSP added to the end.
    traj_name = metadata.userParameters.userParameterString[1].value[:32] # Get the first 32 chars, because a bug sometimes causes this field to have /OSP added to the end.


    # load the .mat file containing the trajectory
    # Search for the file in the metafile_paths
    for path in metafile_paths:
        metafile_fullpath = os.path.join(path, traj_name + ".mat")
        if os.path.isfile(metafile_fullpath):
            logging.info(f"Loading metafile {traj_name} from {path}...")
            traj = loadmat(metafile_fullpath)
            break
    else:
        logging.error(f"Trajectory file {traj_name}.mat not found in specified paths.")
        return

    n_unique_angles = int(traj['param']['interleaves'][0,0][0,0])
    nTRs = traj['param']['repetitions'][0,0][0,0]

    kx = traj['kx'][:,:n_unique_angles]
    ky = traj['ky'][:,:n_unique_angles]


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

    msize = np.int16(10 * traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0])
    msize = (np.int16(msize * 2) // 2) * 2

    ktraj = 0.5 * (ktraj / kmax) * msize

    nchannel = metadata.acquisitionSystemInformation.receiverChannels
    pre_discard = traj['param']['pre_discard'][0,0][0,0]
    w = traj['w']
    w = np.reshape(w, (1,w.shape[1]))


    if 'n_slices' in traj['param'].dtype.names:
        n_slices = traj['param']['n_slices'][0][0][0][0]
        mode = "multislice"
    else:
        n_slices = 1
        mode = "RT"

    if 'slice_shift' in traj['param'].dtype.names:
        slice_shift = traj['param']['slice_shift'][0][0][0][0]
    else:
        slice_shift = 0

    # buffer for collecting data... 
    frames = []
    arm_counters = []
    datas = []
    L = None

    # buffers for dmd
    dmd_buff = []

    image_buffer_per_slice = []

    arm_counter = 0
    rep_counter = 0
    past_slice = -1 
    device = sp.Device(gpu_device)

    coord_gpu = sp.to_device(ktraj, device=device)
    w_gpu = sp.to_device(w, device=device)

    sens = []
    wf_list = []

    arm: ismrmrd.Acquisition | ismrmrd.Waveform | None
    for arm in conn:

        time_to_receive = time.perf_counter() - end_iter 
        logging.debug("Time to receive data: %f ms.", time_to_receive*1000)

        if arm is None:
            break
        start_iter = time.perf_counter()
        
        if type(arm) is ismrmrd.Acquisition:
            # First arm came, if GIRF is requested, correct trajectories and reupload.
            if (arm.scan_counter == 1) and APPLY_GIRF and (rep_counter == 0):
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
                coord_gpu = sp.to_device(k_pred, device=device) # Replace  the original k-space

            if (arm.scan_counter == 1) and (arm.data.shape[1]-pre_discard/2) == coord_gpu.shape[1]/2:
                # Check if the OS is removed. Should only happen with offline recon.
                coord_gpu = coord_gpu[:,::2,:]
                w_gpu = w_gpu[:,::2]
                pre_discard = int(pre_discard//2)
                

            startarm = time.perf_counter()
            adata = sp.to_device(arm.data[:,pre_discard:], device=device)

            n_arm_fs = 144 

            if mode == 'multislice':
                #coil_map_bool = arm.idx.kspace_encode_step_1 < n_arm_fs
                current_slice = arm.idx.slice
                if past_slice != current_slice:

                    # When switching slices, filter wf_list to keep only waveforms within the time range of the current slice's images
                    if image_buffer_per_slice:
                        # Get acquisition_time_stamp range for this slice
                        start_ts = image_buffer_per_slice[0].acquisition_time_stamp
                        end_ts = image_buffer_per_slice[-1].acquisition_time_stamp

                        ecg_sampling_time = wf_list[0].getHead().sample_time_us*1e-6 # [us] -> [s]

                        # Filter wf_list for waveforms within this time range
                        wf_list_filtered = [wf for wf in wf_list if start_ts <= wf.time_stamp <= end_ts]

                        # filter the wf_list_filtered for only waveforms with wf.getHead().waveform_id == 0
                        wf_list_filtered = [wf for wf in wf_list_filtered if wf.getHead().waveform_id == 0]

                        wf_ts = wf_list_filtered[0].time_stamp

                        # grab only the wf.data[4,:] 
                        wf_list_filtered = [wf.data[4,:] for wf in wf_list_filtered]

                        ecg_triggers = np.array(np.concatenate(wf_list_filtered, axis=0) > 0, dtype=int)

                        time_ecg = np.arange(ecg_triggers.shape[0])*ecg_sampling_time - (image_buffer_per_slice[0].acquisition_time_stamp - wf_ts)*1e-3

                        # Make ECG same raster time with the acquisition. For every value 1, make closest time value 1 in the finer grid.
                        ecg_trig_times = time_ecg[ecg_triggers == 1]
                        acq_sampling_time = metadata.sequenceParameters.TR[0]*1e-3

                        # PK HACK below n_arm_fs * 3
                        n_acq = n_arm_fs * 3
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



                    coil_map_bool = True
                    del frames[:len(frames)]
                    del arm_counters[:len(arm_counters)]
                    del datas[:len(datas)]
                    #del dmd_buff[:len(dmd_buff)]
                past_slice = current_slice
            elif mode == 'RT':
                coil_map_bool = rep_counter == 0

            with device:
                if coil_map_bool:
                    frames.append(fourier.nufft_adjoint(
                            adata*w_gpu,
                            coord_gpu[arm_counter,:,:],
                            (nchannel, msize, msize)))
                arm_counters.append(arm_counter)
                datas.append(adata)
                
            endarm = time.perf_counter()
            logging.debug("Elapsed time for arm %d NUFFT: %f ms.", arm_counter, (endarm-startarm)*1e3)

            arm_counter += 1
            if arm_counter == n_unique_angles:
                arm_counter = 0

            start = time.perf_counter()
            if coil_map_bool:
                if ((arm.idx.kspace_encode_step_1+1)%n_arm_fs == 0): 
                    xp = device.xp
                    with device:
                        time_for_sens = time.perf_counter()
                        sens = sp.to_device(process_csm(frames), device=device)
                        logging.debug("Elapsed time for coil map estimation: %f ms.", (time.perf_counter()-time_for_sens)*1e3)
                        S = sp.linop.Multiply(sens.shape, sens, conj=True)
                        R = sp.linop.Sum(sens.shape, [0])
                        What = sp.linop.Multiply([sens.shape[0], adata.shape[1]*n_arm_per_frame], xp.tile(xp.sqrt(w_gpu), [1, n_arm_per_frame]))
                        xn_1 = xp.zeros(sens.shape, xp.complex128)
                        for g in frames:
                            xn_1 += g
                        xn_1 = R*S*xn_1 * n_arm_per_frame/n_arm_fs
                        xn = xp.copy(xn_1)

                    del frames[:n_arm_fs]
                    del datas[:n_arm_fs]
                    del arm_counters[:n_arm_fs]

                    coil_map_bool = False

                    image = process_group(arm, xn, device, rep_counter, cfg, metadata, n_slices=n_slices, slice_shift=slice_shift)
                    rep_counter += 1
                    logging.debug("Sending image to client:\n%s", image)
                    conn.send_image(image)

            else:
                #if (((arm.idx.kspace_encode_step_1 + 1) % n_arm_per_frame) == 0):
                if len(arm_counters) == n_arm_per_frame:
                    with device:
                        prep_time_start = time.perf_counter()
                        # online STCR
                        xp = device.xp
                        trajectory_frame = sp.to_device(coord_gpu[sp.to_device(arm_counters,device=device), :, :], device=device)
                        trajectory_frame = rearrange(trajectory_frame, 's r t -> (s r) t')
                        F = NUFFT(sens.shape, trajectory_frame, toeplitz=True)
                        G = FiniteDifference(sens.shape[1:])

                        Aframe = What * F * S.H * R.H
                        data_frame = rearrange(xp.array(datas), 'f c r -> c (f r)')
                        data_frame = What.H * data_frame 
                        xn = Aframe.H * data_frame

                        prep_time_end = time.perf_counter()
                        logging.debug("Elapsed time for preparation: %f ms.", (prep_time_end-prep_time_start)*1e3)

                        if alg_type == "cg":
                            cg_alg = sp.alg.ConjugateGradient(Aframe.H*Aframe, xn, xn_1, max_iter=3)
                            while not cg_alg.done():
                                cg_alg.update()
                            xn = cg_alg.x
                        elif alg_type == "stcr":
                            if L is None:
                                lipschitz_start = time.perf_counter()
                                L = MaxEig(Aframe.N, max_iter=40, dtype=xn.dtype, device=device).run()
                                logging.debug("Time for lipschitz constant: %f ms.", (time.perf_counter()-lipschitz_start)*1e3)
                            del_0 = xp.zeros(xn.shape, dtype=xp.complex128)
                            time_recon = metadata.sequenceParameters.TR[0] * n_arm_per_frame * 0.65

                            max_image = xp.abs(xn).max()
                            lamt = lambdat * max_image 
                            lams = lambdas * max_image

                            del_0 = online_STCR_ISTA_2_timed(Aframe, G, xn_1, xn, lamt, lams, 1/L, mu=0.2, yn=data_frame, time_recon=time_recon, deln=del_0) #, deln=del_0)
                            xn = xn_1 + del_0
                        elif alg_type == "dmd":
                            # timing just the dmd
                            start_dmd = time.perf_counter()
                            f_threshold = 3 # TODO: make this configurable
                            N_frames_bin = 30 
                            dt_dmd = metadata.sequenceParameters.TR[0] * n_arm_per_frame * 1e-3 
                            dmd_buff.append(xn)
                            if len(dmd_buff) >= N_frames_bin:
                                # TODO: it is proably more efficeint to create the buffer and simply index it (circular)
                                xn = dmd_filt(xp.array(dmd_buff).transpose((1,2,0)), dt_dmd, f_threshold)[:,:,-1]
                                logging.debug("Frame Finished")
                            # if the lenght of dmd_buff is more than N_frames_bin, remove the first element
                            if len(dmd_buff) > N_frames_bin:
                                dmd_buff.pop(0)
                            end_dmd = time.perf_counter()
                        else:
                            pass
                        xn_1 = xp.copy(xn)

                    image = process_group(arm, xn, device, rep_counter, cfg, metadata, n_slices=n_slices, slice_shift=slice_shift)
                    end = time.perf_counter()

                    logging.debug("Elapsed time for frame processing: %f ms.", (end-start)*1000)
                    del frames[:n_arm_per_frame]
                    del datas[:n_arm_per_frame]
                    del arm_counters[:n_arm_per_frame]

                    logging.debug("Sending image to client:\n%s", image)

                    # send time
                    send_time_start = time.perf_counter()
                    image_buffer_per_slice.append(image)
                    conn.send_image(image)
                    send_time_end = time.perf_counter()
                    logging.debug("Elapsed time for sending image: %f ms.", (send_time_end-send_time_start)*1000)
                    
                    rep_counter += 1
            end_iter = time.perf_counter()
            logging.debug("Elapsed time for per recon frame: %f ms.", (end_iter-start_iter)*1000)
        elif type(arm) is ismrmrd.Waveform:
            wf_list.append(arm)

    # Send waveforms back to save them with images
    for wf in wf_list:
        conn.send_waveform(wf)

    conn.send_close()
    logging.debug('Reconstruction is finished.')


def process_csm(frames):

    xp = sp.get_device(frames[0]).xp
    device = frames[0].device
    with device:
        data = xp.zeros(frames[0].shape, dtype=np.complex128)
        for g in frames:
            data += sp.to_device(g, device=device)
        #csm_est=EspiritCalib(sp.fft(data,axes=[1,2]), max_iter=60, device=device).run()
        #(csm_est, rho) = coils.calculate_csm_inati_iter(data.get(), smoothing=32)
        (csm_est, rho) = calculate_csm_walsh_gpu(data, smoothing=20)
        csm_est = xp.squeeze(csm_est)

    return csm_est


def process_group(group, data, device, rep, config, metadata,n_slices=1, slice_shift=0):
    xp = device.xp
    with device:
        #data = xp.abs(xp.flipud(data.T))
        #data = xp.abs(xp.fliplr(data.T))
        data = xp.abs(xp.fliplr(xp.flipud(data.T)))

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

    par_thickness = slice_shift 
    partition = group.idx.slice

    # Set field of view
    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.z))

    image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                            ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.y), 
                            ctypes.c_float(8))



    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON', 'online_stcr_2D'],
                         'WindowCenter':           str((maxVal+1)/2),
                         'WindowWidth':            str((maxVal+1)),
                         'NumArmsPerFrame':        str(config['reconstruction']['arms_per_frame']),
                         'GriddingWindowShift':    str(config['reconstruction']['arms_per_frame']), 
                         'ImageScaleFactor':       str(dscale)
                         })

    # Add image orientation directions to MetaAttributes if not already present
    if meta.get('ImageRowDir') is None:
        meta['ImageRowDir'] = ["{:.18f}".format(image.getHead().read_dir[0]), "{:.18f}".format(image.getHead().read_dir[1]), "{:.18f}".format(image.getHead().read_dir[2])]

    if meta.get('ImageColumnDir') is None:
        meta['ImageColumnDir'] = ["{:.18f}".format(image.getHead().phase_dir[0]), "{:.18f}".format(image.getHead().phase_dir[1]), "{:.18f}".format(image.getHead().phase_dir[2])]

    R_matrix = np.hstack((np.array(image.getHead().phase_dir).reshape(3,1),np.array(image.getHead().read_dir).reshape(3,1),np.array(image.getHead().slice_dir).reshape(3,1)))

    logging.debug("slice: %i", partition)
    partition_vector = np.array([0,0,-1 * ((partition-(n_slices/2.0))*par_thickness)])
    position_offset = np.matmul(R_matrix,partition_vector)

    for ii in range(3):
        image.position[ii] = image.position[ii] + position_offset[ii]

    xml = meta.serialize()
    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image
