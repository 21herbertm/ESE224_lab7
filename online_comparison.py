"""
ONLINE COMPARISON FILE

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
    numberDigits = len(training_set_DFT_values)
    _, N = training_set_DFT_values[0].shape
    average_spectra = np.zeros((numberDigits, N), dtype=np.complex_)
    average_signal = np.zeros((numberDigits, N), dtype=np.complex_)

# LOOPS THROUGH ALL VALUES TO GET THE AVERAGE OF MODULUS OF SPECTRA
    for i in range(numberDigits):
        # Average of modulus of spectra
        average_spectra[i, :] = np.mean(np.absolute(training_set_DFTs[i]), axis=0)
        iDFT = idft(average_spectra[i, :], fs, N)
        y_demod, Treal = iDFT.solve_ifft()
        average_signal[i, :] = y_demod
"""
the score function is nothing more than the norm of a multiplication of DFTs. We can implement the score functions
"""

    for m in range(time_slots):
        voicerecording = sd.rec(int(T * fs), fs, 1)
        sd.wait()  # Wait until recording is finished
        rec_i = voicerecording.astype(np.float32)
        """
        Note that numpy.float is just an alias to Python's float type. It is not a numpy scalar type like numpy.float64. 
        The name is only exposed for backwards compatibility with a very early version of numpy that inappropriately
        exposed numpy.float64 as numpy.float, causing problems when people did from numpy import
        """
        rec_i=rec_i[:,0]

        # We can use the norm of the ith signal to normalize its DFT
        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i
        # Comparisons
        inner_product = np.zeros(numberDigits)

        for x in range(numberDigits):
            inner_product[j] = np.linalg.norm(np.convolve(rec_i , average_signal[j, :]))**2

# PRINTS OUT THE INNER PRODUCTS
        print('The number said is:', np.argmax(inner_product) + 1)
