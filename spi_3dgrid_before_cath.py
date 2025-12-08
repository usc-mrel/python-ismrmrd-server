
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
from sigpy.mri.dcf import pipe_menon_dcf
import sigpy as sp
from sigpy import fourier
import cupy as cp
from scipy.io import savemat

# Folder for debug output files
debugFolder = "/tmp/share/debug"
DEBUG = False

def window_prctile(I, WL = 100):

	WU = np.percentile(np.abs(I), WL)
	WL = np.percentile(np.abs(I), 100 - WL)

	I[I > WU] = WU
	I[I < WL] = WL
	I = (I - WL) / (WU - WL)
	return I

def get_ref_proj_filename(metadata):
	for str_param in metadata.userParameters.userParameterString:
		if str_param.name == "tSequenceVariant":
			traj_name = str_param.value[:32] # Get first 32 chars, because a bug sometimes causes this field to have /OSP added to the end.
			break
	fn = f"SPI_ref_proj_{traj_name}.mat"
	return fn

def get_projections(kloc, kdata, w, msize_inplane, nr_interleaves):
	
	xp = np if kloc.device == sp.cpu_device else cp
	device = kloc.device

	nr_planes = kloc.shape[0] // nr_interleaves
	nchannel  = kdata.shape[0]
	# Hexagonal sampling
	hx_theta       = xp.pi / nr_interleaves

	kloc_ = kloc[:nr_interleaves, :, :2]

	# ktraj = np.stack((kx, -ky, kz), axis=2)
	
	Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
								[np.sin(theta),  np.cos(theta), 0],
								[0,              0,             1]])

	get_arm_idx = lambda plane_idx, nr_interleaves: np.arange( plane_idx * nr_interleaves,  (plane_idx+1) * nr_interleaves )

	# Get projection data for plane idx
	Plist = []
	for plane_idx in range(nr_planes):
		arm_idx_   = get_arm_idx(plane_idx, nr_interleaves)

		with device:
			Plist.append( fourier.nufft_adjoint(
			kdata[:, arm_idx_, :] * w[xp.newaxis, :, :],
			kloc_,
			(nchannel, msize_inplane, msize_inplane)))
	
	return xp.array(Plist)

def process(connection, config, metadata):
	logging.disable(logging.WARNING)
	
	logging.info("Config: \n%s", config)
	logging.info("Metadata: \n%s", metadata)

	# Read config file and import BART
	cfg = reconutils.load_config('spi_config.toml')
	if cfg is None:
		logging.error("Failed to load configuration. Exiting.")
		return

	APPLY_GIRF 			= cfg['reconstruction']['apply_girf']
	gpu_device 			= cfg['reconstruction']['gpu_device']
	metafile_paths 		= cfg['metafile_paths']

	device = sp.Device(gpu_device)

	# load the .mat file containing the trajectory
	traj = reconutils.load_trajectory(metadata, metafile_paths)
	if traj is None:
		logging.error("Failed to load trajectory.")
		return

	# Filename for projections
	fn = get_ref_proj_filename(metadata)

	n_unique_angles = traj['param']['repetitions'][0,0][0,0]

	kx = traj['kx'][:,:]
	ky = traj['ky'][:,:]
	kz = traj['kz'][:,:]

	dt = traj['param']['dt'][0,0][0,0]
	msize_inplane = np.int16(traj['param']['fov'][0,0][0,0] / traj['param']['spatial_resolution'][0,0][0,0] * cfg['reconstruction']['fov_oversampling'])
	msize_slab    = np.int16(traj['param']['fov'][0,0][0,1] / traj['param']['spatial_resolution'][0,0][0,0] * cfg['reconstruction']['fov_oversampling'])
	print(f"Reconstruction matrix size: %d x %d x %d\n", msize_inplane, msize_slab)
	delta_r = traj['param']['spatial_resolution'][0,0][0,0] * 1e3 # [m] -> [mm]


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

	pre_discard = traj['param']['pre_discard'][0,0][0,0]
	w = traj['w']
	w = w[None, :, :]

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
			if np.any(np.isnan(arm.data)):
				print("NaN values in the data!")
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
	data = np.array(data) * 1e3 # TODO: Scaled manually
	data = np.transpose(data, axes=(1, 2, 0))
	coord = np.array(coord, dtype=np.float32)
	coord = np.transpose(coord, axes=(1, 0, 2))

	kdata = np.transpose(data, (2, 0, 1))           # [ndisks*n_arms, n_ch, n_samples]
	kdata = np.transpose(kdata, (2, 0, 1))       # [n_samples, ndisks*n_arms, n_ch]
	# ADDED
	kdata = np.transpose(kdata, (2, 1, 0)) 	 # [n_ch, ndisks*n_arms, n_samples]

	kloc = np.transpose(coord, (1, 0, 2)) 	# [ndisks*n_arms, n_samples, 2]
	kloc = kloc.transpose((2, 1, 0))		# [3, nr_samples, nr_arms]
	# ADDED
	kloc = np.transpose(kloc, (2, 1, 0)) 	# [nr_arms, nr_samples, 3]

	logging.info('kdata array shape: {}'.format(kdata.shape))
	logging.info('kloc array shape: {}'.format(kloc.shape))
	
	# Do NUFFT
	w = w / np.max(w)

	with device:
		coord_gpu = sp.to_device(kloc, device=device)
		adata = sp.to_device(kdata, device=device)
		w_gpu = sp.to_device(w, device=device)
		nchannel = kdata.shape[0]
	
	# 3D NUFFT
	with device:
		img = fourier.nufft_adjoint(
			adata*w_gpu.transpose(0,2,1),
			coord_gpu,
			(nchannel, msize_inplane, msize_inplane, msize_inplane))

	img = rssq(img, axis=0).get()
	maxVal = 2**12 - 1

	img *= maxVal/img.max()
	img = np.around(img)

	# Window Percentile
	# Take percentile for display
	WU = np.percentile(np.abs(img), cfg['reconstruction']['display_win_level'])
	WL = np.percentile(np.abs(img), 100 - cfg['reconstruction']['display_win_level'])
	WindowCenter = (WU + WL) / 2
	WindowWidth  = (WU - WL)

	# Reshape, process and send images.
	# TODO: This assumes the 3rd dimension is slice dimension. Any other dimension can be sliced. Investigate.
	# TODO: Edge slices that are not properly encoded can be skipped here.
	for ii in range(img.shape[2]):
		image = process_group(grp, img[None,:,:,ii], ii, 0, img.shape[2], delta_r, metadata, os = cfg['reconstruction']['fov_oversampling'], WindowCenter=WindowCenter, WindowWidth=WindowWidth)
		connection.send_image(image)

	# Save projection images somewhere
	with device:
		w_proj_gpu = sp.to_device(traj['w_proj'], device=device)
		P = get_projections(coord_gpu, adata, w_proj_gpu, msize_inplane, nr_interleaves=traj['param']['interleaves'][0,0][0,0]).get()
		savemat(fn, {'P': P, 'msize_inplane': msize_inplane, 'nr_interleaves': traj['param']['interleaves'][0,0][0,0]})

	connection.send_close()
	logging.info('Reconstruction is finished.')


def process_group(group, data, rep, image_series_idx, Nslc, resolution, metadata, os=1, WindowCenter=None, WindowWidth=None):

	data = np.abs(np.flip(data, axis=(1,)))
	data = np.transpose(data, (0, 2, 1))

	# Determine max value (12 or 16 bit)
	BitsStored = 12
	if (mrdhelper.get_userParameterLong_value(metadata, "BitsStored") is not None):
		BitsStored = mrdhelper.get_userParameterLong_value(metadata, "BitsStored")
	
	maxVal = 2**BitsStored - 1

	if WindowCenter is None or WindowWidth is None:
		WindowCenter = (maxVal+1)/2
		WindowWidth  = (maxVal+1)

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
	# TODO: Fix if given FOVs are not isotropic, i.e. FoV set are for "encoded", not "reconned" space. It can be set as max of FoVs.
	recon_fov = max(metadata.encoding[0].reconSpace.fieldOfView_mm.x, metadata.encoding[0].reconSpace.fieldOfView_mm.y)
	image.field_of_view = (ctypes.c_float(recon_fov  * os), 
							ctypes.c_float(recon_fov * os), 
							ctypes.c_float(resolution))

	# Set slice position
	# Center of the excited volume, in (left, posterior, superior) (LPS) coordinates relative to isocenter in millimeters. 
	# NB this is different than DICOM’s ImageOrientationPatient, which defines the center of the first (typically top-left) voxel.
	# TODO: This assumes slice direction is along "AP" axis. Can be generalized.
	# TODO: We step by resolution assuming during recon we kept it the same. Safer way may be dividing FoV by the number of slices for the step size.
	#image.position[1] = ctypes.c_float(image.position[1] + resolution * (rep-Nslc//2))
	#r_GCS2PCS = np.array([group.phase_dir, -np.array(group.read_dir), group.slice_dir])
	slice_shift = [0, 0, resolution * (rep-Nslc//2)]
	#del_img_pos = r_GCS2PCS @ slice_shift

	#
	r_GCS2RCS = np.array(  [[0,    1,   0],  # [PE]   [0 1 0] * [r]
										[1,    0,   0],  # [RO] = [1 0 0] * [c]
										[0,    0,   1]]) # [SL]   [0 0 1] * [s]
	r_GCS2PCS = np.array([group.phase_dir, -np.array(group.read_dir), group.slice_dir])
	r_PCS2DCS = GIRF.calculate_matrix_pcs_to_dcs(metadata.measurementInformation.patientPosition.value)
	r_GCS2DCS = r_PCS2DCS.dot(r_GCS2PCS)
	sR = r_GCS2DCS.dot(r_GCS2RCS)
	del_img_pos = r_GCS2PCS.T @ slice_shift

	for ii in range(3):
		image.position[ii] = ctypes.c_float(image.position[ii] + del_img_pos[ii])

	# Set ISMRMRD Meta Attributes
	meta = ismrmrd.Meta({'DataRole':               'Image',
						 'ImageProcessingHistory': ['FIRE', 'PYTHON'],
						 'WindowCenter':           str(WindowCenter), #str((maxVal+1)/2),
						 'WindowWidth':            str(WindowWidth)})  #str((maxVal+1))})

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

