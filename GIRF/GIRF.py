import numpy as np
from scipy.io import loadmat
import time
from scipy.signal.windows import hann
from scipy.fft import fft, fftshift, ifft, ifftshift
import pathlib

def apply_GIRF(gradients_nominal: np.ndarray, dt: float, R: np.ndarray, girf_file: str | None = None, tRR: float = 0):
    """
    GIRF correction. 
    
    Parameters:
    -----------
        gradients_nominal: Nominal gradient waveforms (samples, interleaves, axes)
        dt: Sampling time interval [s]
        R: Rotation matrix (3x3) or dict containing 'R' key
        girf_file: Path to GIRF calibration file (optional)
        tRR: Receiver-related delay [samples] (default: 0)
    
    Returns:
    -----------
        kPred: Predicted k-space trajectory
        GPred: Predicted gradient waveforms
    """
    # Handle "nasty" co-opting of R-variable to include field info.
    if isinstance(R, dict):
        R = R['R']

    # If no path is given assume our scanner for backwards compatibility.
    if girf_file is None:
        girf_file = str(pathlib.Path(__file__).parent / 'GIRF_20200221_Duyn_method_coil2.mat')

    try:
        girf_data = loadmat(girf_file)
        print(f'Using {girf_file}')
    except FileNotFoundError:
        print("Couldn't find the GIRF file, using ones..")
        girf_data = {'GIRF': np.ones((3800, 3))}

    GIRF = girf_data['GIRF']

    dtGIRF = 10e-6
    dtSim = dt 
    l_GIRF = GIRF.shape[0]
    samples, interleaves, gs = gradients_nominal.shape

    # If readout is real long, need to pad the GIRF measurement
    if samples * dt > dtGIRF * l_GIRF:
        print('readout length > 38ms, zero-padding GIRF')
        pad_factor = 1.5 * (samples * dt) / (dtGIRF * l_GIRF)
        new_GIRF = np.zeros((round(l_GIRF * pad_factor), 3))

        # Vectorized: Process all 3 axes at once
        fft_GIRF = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(GIRF, axes=0), axis=0), axes=0)
        zeropad = round(abs((l_GIRF - len(new_GIRF)) / 2))
        temp = np.zeros((len(new_GIRF), 3))
        
        # Smoothing of padding
        H_size = 200
        H = hann(H_size)
        fft_GIRF[:H_size//2, :] *= H[:H_size//2, np.newaxis]
        fft_GIRF[-(H_size//2 - 1):, :] *= H[H_size//2 + 1:, np.newaxis]
        
        temp[zeropad:zeropad + l_GIRF, :] = fft_GIRF
        new_GIRF = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(temp, axes=0), axis=0), axes=0)

        GIRF = new_GIRF

    if tRR is None:
        tRR = 0

    ADCshift = 0.85e-6 + 0.5 * dt + tRR * dt  # NCO Clock shift

    GPred = np.zeros((samples, 3, interleaves), dtype=float)

    start = time.perf_counter()

    N = gradients_nominal.shape[0]
    BW = 1 / dt
    L1 = round(dtGIRF * l_GIRF / dtSim)
    dw = 1 / (L1 * dtSim)
    L2 = round(BW / dw)

    hanning400 = hann(400)

    index_range1 = np.arange(-np.floor(N/2), np.ceil(N/2), dtype=int) + int(np.floor(L1/2)) + 1
    index_range2 = np.arange(-np.floor(l_GIRF/2), np.ceil(l_GIRF/2), dtype=int) + int(np.floor(L1/2)) + 1
    index_range3 = np.arange(-np.floor(samples/2), np.ceil(samples/2), dtype=int) + int(np.floor(L2/2)) + 1

    w = np.arange(-np.floor(L1/2), np.ceil(L1/2)) * dw
    
    # Vectorized approach: Prepare all gradients
    G0_all = gradients_nominal.copy()  # (samples, interleaves, gs)
    
    if G0_all.shape[-1] == 2:
        # Pad to 3 axes for all interleaves at once
        G0_all = np.concatenate((G0_all, np.zeros((gradients_nominal.shape[0], interleaves, 1))), axis=2)
    
    # Apply rotation to all interleaves at once using einsum for efficiency
    # G0_all is (samples, interleaves, 3), R is (3, 3)
    # Result should be (samples, interleaves, 3)
    G0_rotated = np.einsum('ij,klj->kli', R, G0_all)
    
    # Pre-compute GIRF1 for all axes (same for all interleaves)
    GIRF1_all = np.zeros((L1, 3), dtype=np.complex64)
    
    if dt > dtGIRF:
        GIRF1_all = GIRF[round(l_GIRF/2 - L1/2 + 1):round(l_GIRF/2 + L1/2), :]
        temp = hann(10)
        GIRF1_all[0, :] = 0
        GIRF1_all[-1, :] = 0
        GIRF1_all[1:round(len(temp)/2) + 1, :] *= temp[:round(len(temp)/2), np.newaxis]
        GIRF1_all[-round(len(temp)/2):-1, :] *= temp[round(len(temp)/2) + 1:, np.newaxis]
    else:
        GIRF1_all[index_range2, :] = GIRF
    
    # Precompute phase shift term
    phase_shift = np.exp(1j * ADCshift * 2 * np.pi * w)
    
    # ========================================================================
    # OPTIMIZED VECTORIZED: Process all 3 axes at once per interleave
    # ========================================================================
    # This approach vectorizes over the 3 axes (the tight inner loop)
    # but keeps the interleave loop to avoid large memory allocations
    # and 3D FFT overhead which can be slower than multiple 2D FFTs
    
    # Pre-allocate output to avoid repeated allocation
    GPred = np.zeros((samples, 3, interleaves), dtype=float)
    
    # Process each interleave with all 3 axes vectorized
    for interleave_idx in range(interleaves):
        # Get rotated gradients for this interleave: (samples, 3)
        G0 = G0_rotated[:, interleave_idx, :]
        
        # Prepare gradient array for all 3 axes: (L1, 3)
        G_all = np.zeros((L1, 3))
        G_all[index_range1, :] = G0
        
        # Make waveforms periodic for all axes simultaneously
        # Broadcasting: (3,) * (200,) -> (200, 3)
        H = G_all[index_range1[-1], :] * hanning400[:, np.newaxis]
        end_indices = index_range1[-1] + np.arange(1, len(hanning400)//2 + 1)
        G_all[end_indices, :] = H[len(hanning400)//2:, :]
        
        # FFT for all 3 axes at once: (L1, 3)
        I_all = fftshift(fft(ifftshift(G_all, axes=0), axis=0), axes=0)
        
        # Apply GIRF and phase shift to all axes: (L1, 3) * (L1, 3) * (L1, 1)
        P_all = I_all * GIRF1_all * phase_shift[:, np.newaxis]
        
        # Prepare for inverse FFT: (L2, 3)
        PredGrad_all = np.zeros((L2, 3), dtype=np.complex64)
        zeropad = round(abs((L1 - L2) / 2))
        PredGrad_all[zeropad:zeropad + L1, :] = P_all
        
        # Inverse FFT for all axes: (L2, 3)
        PredGrad_all = fftshift(ifft(ifftshift(PredGrad_all, axes=0), axis=0), axes=0)
        PredGrad_all = np.real(PredGrad_all)
        
        # Extract and rotate: (samples, 3)
        Predicted = PredGrad_all[index_range3, :]
        
        # Apply inverse rotation and store
        GPred[:, :, interleave_idx] = np.dot(Predicted, R)

    kPred = np.cumsum(GPred, axis=0)

    gamma = 2.67522212e8  # Gyromagnetic ratio for 1H [rad/sec/T]
    kPred = 0.01 * (gamma / (2 * np.pi)) * (kPred * 0.01) * dt

    kPred = np.transpose(kPred, (0, 2, 1))
    GPred = np.transpose(GPred, (0, 2, 1))

    end = time.perf_counter()
    print(f"Elapsed time during GIRF apply: {end-start} secs.")
    
    return kPred, GPred

def hanningt(windowLength):
    try:
        from scipy.signal.windows import hann
        return hann(windowLength)
    except ImportError:
        N = windowLength - 1
        num = np.linspace(0, N, windowLength)
        return 0.5 * (1 - np.cos(2 * np.pi * num / N))

# Example usage:
# kPred, GPred = apply_GIRF(gradients_nominal, dt, R)

def calculate_matrix_pcs_to_dcs(patient_position):
    """
    Calculate transformation matrix from Patient Coordinate System (PCS) to Device Coordinate System (DCS).

    Parameters:
    - patient_position (str): Patient position code (e.g., 'HFP', 'HFS', 'HFDR', 'HFDL', 'FFP', 'FFS', 'FFDR', 'FFDL').

    Returns:
    - R_pcs2dcs (numpy.ndarray): 3x3 transformation matrix.
    """
    # Initialize the transformation matrix
    R_pcs2dcs = np.eye(3)

    # Define transformation matrices for different patient positions
    switch_dict = {
        'HFP': np.array([[-1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]]),
        'HFS': np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]]),
        'HFDR': np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, -1]]),
        'HFDL': np.array([[0, -1, 0],
                          [-1, 0, 0],
                          [0, 0, -1]]),
        'FFP': np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]]),
        'FFS': np.array([[-1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]]),
        'FFDR': np.array([[0, -1, 0],
                          [1, 0, 0],
                          [0, 0, 1]]),
        'FFDL': np.array([[0, 1, 0],
                          [-1, 0, 0],
                          [0, 0, 1]])
    }

    # Retrieve the transformation matrix based on the patient position
    R_pcs2dcs = switch_dict.get(patient_position, R_pcs2dcs)

    return R_pcs2dcs


if __name__ == '__main__':
    # Example code to construct rot matrix
    # RO_sign = -1
    # phase_dir = raw_data.head.phase_dir[:,1])
    # read_dir  = raw_data.head.read_dir[:,1])*RO_sign # Because we flip PE(ky) of the trajectory.
    # slice_dir = raw_data.head.slice_dir[:,1])
    # rotMatrixGCSToPCS = [phase_dir read_dir slice_dir]
    # patient_position = 'HFS'
    # rotMatrixPCSToDCS = calculate_matrix_pcs_to_dcs(patient_position)
    # rotMatrixGCSToDCS = rotMatrixPCSToDCS.dot(rotMatrixGCSToPCS)
    # load example gradients_nominal, rotMatrixGCSToDCS, dt, and resulting GPred from Matlab to compare.
    COMPARE_MATLAB = True
    if COMPARE_MATLAB:
        girf_matlab = loadmat('girf_matlab')

        gpred_matlab = girf_matlab['g_predicted']
        kpred_matlab = girf_matlab['k_predicted']

        gradients_nominal = girf_matlab['g_tot']
        sR = {}
        sR['R'] = girf_matlab['sR']['R'][0,0]
        sR['T'] = 0.55
        tRR = -1.5
        dt = girf_matlab['dt'][0][0]
        kPred, GPred = apply_GIRF(gradients_nominal, dt, sR, tRR)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(GPred[:,:,0] - gpred_matlab[:,:,0])
        plt.figure()
        plt.plot(GPred[:,0,0])
        plt.plot(gpred_matlab[:,0,0])
        plt.show()

    PROFILE_GIRF = True
    if PROFILE_GIRF:
        prf_ex = np.load('girf_prof.npz')
        sR = {'R':prf_ex['R'], 'T': 0.55}
        kPred, GPred = apply_GIRF(prf_ex['g_nom'], prf_ex['dt'], sR, prf_ex['tRR'])

    TEST_ROT_MTX = False
    if TEST_ROT_MTX:
        import ismrmrd
        f = ismrmrd.Dataset('/server/home/btasdelen/MRI_DATA/bodycomp/vol0712_20230926/raw/h5/meas_MID00219_FID07835_fl3d_spiral_vibe_bh_TE0_39_TR5_2_20deg_Qfatsat.h5', '/dataset', False)
        head = f.read_acquisition(0).getHead()
        meta = ismrmrd.xsd.CreateFromDocument(f.read_xml_header())
        f.close()

        rotMatrixGCSToPCS = np.array([head.phase_dir, -np.array(head.read_dir), head.slice_dir])

        patient_position = meta.measurementInformation.patientPosition.value
        rotMatrixPCSToDCS = calculate_matrix_pcs_to_dcs(patient_position)
        rotMatrixGCSToDCS = rotMatrixPCSToDCS.dot(rotMatrixGCSToPCS)
        pass


