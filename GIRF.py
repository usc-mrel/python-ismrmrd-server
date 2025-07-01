import numpy as np
from scipy.io import loadmat
import time
from scipy.signal.windows import hann
from scipy.fft import fft, fftshift, ifft, ifftshift

def apply_GIRF(gradients_nominal, dt, R, tRR=0):
    # Handle "nasty" co-opting of R-variable to include field info.
    if isinstance(R, dict):
        field_T = R['T']
        R = R['R']
    else:
        field_T = 0.55

    # Load GIRF file based on field strength
    if field_T == 1.4940:
        # 1.5T Aera (NHLBI 2016)
        girf_file = 'GIRF_20160501.mat'
    elif field_T == 0.55:
        # 0.55T Aera (NHLBI 2018)
        girf_file = 'GIRF_20200221_Duyn_method_coil2.mat'
        #girf_file = 'GIRF_20201014_Brodsky_method_coil16.mat'

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

        for i in range(3):
            fft_GIRF = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(GIRF[:, i])))
            zeropad = round(abs((l_GIRF - len(new_GIRF)) / 2))
            temp = np.zeros(len(new_GIRF))
            
            # Smoothing of padding
            H_size = 200
            H = hann(H_size)
            fft_GIRF[:H_size//2] *= H[:H_size//2]
            fft_GIRF[-(H_size//2 - 1):] *= H[H_size//2 + 1:]
            
            temp[zeropad:zeropad + l_GIRF] = fft_GIRF
            new_GIRF[:, i] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(temp)))

        GIRF = new_GIRF

    if tRR is None:
        tRR = 0

    ADCshift = 0.85e-6 + 0.5 * dt + tRR * dt  # NCO Clock shift

    # Nominal = np.zeros((samples, 3), dtype=float)
    Predicted = np.zeros((samples, 3), dtype=float)
    # GNom = np.zeros((samples, 3, interleaves), dtype=float)
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
    for l in range(interleaves):
        G0 = gradients_nominal[:, l, :].copy()
        G0 = np.concatenate((G0, np.zeros((gradients_nominal.shape[0], 1))), axis=1)
        G0 = np.dot(R, G0.T).T

        for ax in range(3):

            G = np.zeros(L1)
            
            # SPIRAL OUT
            G[index_range1] = G0[:, ax]

            # Make a waveform periodic by returning to zero
            H = G[index_range1[-1]] * hanning400
            G[index_range1[-1] + np.arange(1, len(H)//2 + 1)] = H[len(H)//2:]

            # I = np.conj(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G))))  # I am missing something and the grads are flipped without conj.
            I = fftshift(fft(ifftshift(G))) # Mysery solved.

            GIRF1 = np.zeros(L1, dtype=np.complex64)
            
            if dt > dtGIRF:
                GIRF1 = GIRF[round(l_GIRF/2 - L1/2 + 1):round(l_GIRF/2 + L1/2), ax]
                temp = hann(10)
                GIRF1[0] = 0
                GIRF1[-1] = 0
                GIRF1[1:round(len(temp)/2) + 1] *= temp[:round(len(temp)/2)]
                GIRF1[-round(len(temp)/2):-1] *= temp[round(len(temp)/2) + 1:]
            else:
                GIRF1[index_range2] = GIRF[:, ax]

            P = I * GIRF1 * np.exp(1j * ADCshift * 2 * np.pi * w)

            PredGrad = np.zeros(L2, dtype=np.complex64)
            # NomGrad = np.zeros(L2, dtype=np.complex64)
            zeropad = round(abs((L1 - L2) / 2))

            PredGrad[zeropad:zeropad + len(P)] = P
            # NomGrad[zeropad:zeropad + len(I)] = I
            
            PredGrad = fftshift(ifft(ifftshift(PredGrad)))

            # NomGrad = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(NomGrad))) * L2

            # multiplier = np.where(np.real(PredGrad) > 0, 1, -1)
            # PredGrad = np.abs(PredGrad) * multiplier
            PredGrad = np.real(PredGrad)

            # multiplier = np.where(np.real(NomGrad) > 0, 1, -1)
            # NomGrad = np.abs(NomGrad) * multiplier

            # Nominal[:, ax] = NomGrad[index_range3]
            Predicted[:, ax] = PredGrad[index_range3]


        # GNom[:, :, l] = np.dot(R.T, Nominal.T).T
        # GPred[:, :, l] = np.dot(R.T, Predicted.T).T
        GPred[:, :, l] = np.dot(Predicted, R)

    # kNom = np.cumsum(GNom, axis=0)
    kPred = np.cumsum(GPred, axis=0)

    gamma = 2.67522212e8  # Gyromagnetic ratio for 1H [rad/sec/T]
    kPred = 0.01 * (gamma / (2 * np.pi)) * (kPred * 0.01) * dt
    # kNom = 0.01 * (gamma / (2 * np.pi)) * (kNom * 0.01) * dt

    kPred = np.transpose(kPred, (0, 2, 1))
    # kNom = np.transpose(kNom, (0, 2, 1))
    GPred = np.transpose(GPred, (0, 2, 1))
    # GNom = np.transpose(GNom, (0, 2, 1))

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


