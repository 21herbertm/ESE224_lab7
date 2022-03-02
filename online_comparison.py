"""
@author: Juan Cervino
This function computes the online prediction of spoken digits by comparing their spectra
against the average spectrum of the training set stored in "spoken_digits_DFTs.npy".
"""
import numpy as np
from idft import idft
import sounddevice as sd

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency
    time_slots=20 # time of online recording

    # loads (DFTs of) training set
    training_set_DFTs = np.load("spoken_digits_DFTs.npy")

    # Average spectra
    num_digits = len(training_set_DFTs)
    _, N = training_set_DFTs[0].shape
    average_spectra = np.zeros((num_digits, N), dtype=np.complex_)
    average_signal = np.zeros((num_digits, N), dtype=np.complex_)

    for i in range(num_digits):
        # Average of modulus of spectra
        average_spectra[i, :] = np.mean(np.absolute(training_set_DFTs[i]), axis=0)
        iDFT = idft(average_spectra[i, :], fs, N)
        y_demod, Treal = iDFT.solve_ifft()
        average_signal[i, :] = y_demod


    for t in range(time_slots):
        voicerecording = sd.rec(int(T * fs), fs, 1)
        sd.wait()  # Wait until recording is finished
        rec_i = voicerecording.astype(np.float32)
        rec_i=rec_i[:,0]

        # We can use the norm of the ith signal to normalize its DFT
        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i
        # Comparisons
        inner_prods = np.zeros(num_digits)

        for j in range(num_digits):
            inner_prods[j] = np.linalg.norm(np.convolve(rec_i , average_signal[j, :]))**2

        print('The number said is:', np.argmax(inner_prods) + 1)
