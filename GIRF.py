import numpy as np
from scipy.io import loadmat

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
            H = hanningt(H_size)
            fft_GIRF[:H_size//2] *= H[:H_size//2]
            fft_GIRF[-(H_size//2 - 1):] *= H[H_size//2 + 1:]
            
            temp[zeropad:zeropad + l_GIRF] = fft_GIRF
            new_GIRF[:, i] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(temp)))

        GIRF = new_GIRF

    if tRR is None:
        tRR = 0

    ADCshift = 0.85e-6 + 0.5 * dt + tRR * dt  # NCO Clock shift

    Nominal = np.zeros((samples, 3), dtype=float)
    Predicted = np.zeros((samples, 3), dtype=float)
    GNom = np.zeros((samples, 3, interleaves), dtype=float)
    GPred = np.zeros((samples, 3, interleaves), dtype=float)
    kNom = np.zeros((samples, 3, interleaves), dtype=float)
    kPred = np.zeros((samples, 3, interleaves), dtype=float)

    for l in range(interleaves):
        G0 = gradients_nominal[:, l, :].copy()
        G0 = np.concatenate((G0, np.zeros((gradients_nominal.shape[0], 1))), axis=1)
        G0 = np.dot(R, G0.T).T

        for ax in range(3):
            L = round(dtGIRF * l_GIRF / dtSim)
            G = np.zeros(L)
            
            # SPIRAL OUT
            N = len(G0)
            index_range = np.arange(-np.floor(N/2), np.ceil(N/2), dtype=int) + int(np.floor(L/2)) + 1
            G[index_range] = G0[:, ax]

            # Make a waveform periodic by returning to zero
            H = G[index_range[-1]] * hanningt(400)
            G[index_range[-1] + np.arange(1, len(H)//2 + 1)] = H[len(H)//2:]

            dw = 1 / (L * dtSim)
            w = np.arange(-np.floor(L/2), np.ceil(L/2)) * dw
            I = np.conj(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(G))))  # I am missing something and the grads are flipped without conj.

            GIRF1 = np.zeros(L, dtype=np.complex64)
            
            if dt > dtGIRF:
                GIRF1 = GIRF[round(l_GIRF/2 - L/2 + 1):round(l_GIRF/2 + L/2), ax]
                temp = hanningt(10)
                GIRF1[0] = 0
                GIRF1[-1] = 0
                GIRF1[1:round(len(temp)/2) + 1] *= temp[:round(len(temp)/2)]
                GIRF1[-round(len(temp)/2):-1] *= temp[round(len(temp)/2) + 1:]
            else:
                index_range = np.arange(-np.floor(l_GIRF/2), np.ceil(l_GIRF/2)) + np.floor(L/2) + 1
                GIRF1[index_range.astype(int)] = GIRF[:, ax]

            P = I * GIRF1 * np.exp(1j * ADCshift * 2 * np.pi * w)

            BW = 1 / dt
            L = round(BW / dw)
            PredGrad = np.zeros(L, dtype=np.complex64)
            NomGrad = np.zeros(L, dtype=np.complex64)
            zeropad = round(abs((len(G) - L) / 2))

            PredGrad[zeropad:zeropad + len(P)] = P
            NomGrad[zeropad:zeropad + len(I)] = I
            
            PredGrad = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(PredGrad))) * L
            NomGrad = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(NomGrad))) * L

            multiplier = np.where(np.real(PredGrad) > 0, 1, -1)
            PredGrad = np.abs(PredGrad) * multiplier

            multiplier = np.where(np.real(NomGrad) > 0, 1, -1)
            NomGrad = np.abs(NomGrad) * multiplier

            index_range = np.arange(-np.floor(samples/2), np.ceil(samples/2)) + np.floor(L/2) + 1
            Nominal[:, ax] = NomGrad[index_range.astype(int)]
            Predicted[:, ax] = PredGrad[index_range.astype(int)]

        GNom[:, :, l] = np.dot(R.T, Nominal.T).T
        GPred[:, :, l] = np.dot(R.T, Predicted.T).T

        kNom[:, :, l] = np.cumsum(GNom[:, :, l], axis=0)
        kPred[:, :, l] = np.cumsum(GPred[:, :, l], axis=0)

    gamma = 2.67522212e8  # Gyromagnetic ratio for 1H [rad/sec/T]
    kPred = 0.01 * (gamma / (2 * np.pi)) * (kPred * 0.01) * dt
    kNom = 0.01 * (gamma / (2 * np.pi)) * (kNom * 0.01) * dt

    kPred = np.transpose(kPred, (0, 2, 1))
    kNom = np.transpose(kNom, (0, 2, 1))
    GPred = np.transpose(GPred, (0, 2, 1))
    GNom = np.transpose(GNom, (0, 2, 1))

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

if __name__ == '__main__':
    # load example gradients_nominal, rotMatrixGCSToDCS, dt, and resulting GPred from Matlab to compare.
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
    pass