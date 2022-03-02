"""
@author: Juan Cervino
This function predicts spoken digits stored in "test_set.npy" by comparing their spectra
against the average spectrum of the training set stored in "spoken_digits_DFTs.npy".
"""
import numpy as np
from idft import idft

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency

    # loads test set
    test_set = np.load("test_set.npy")

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

    num_recs, N = test_set.shape
    predicted_labels = np.zeros(num_recs)

    for i in range(num_recs):
        rec_i = test_set[i, :]
        # We can use the norm of the ith signal to normalize its DFT
        energy_rec_i = np.linalg.norm(rec_i)
        rec_i /= energy_rec_i

        # Comparisons
        inner_prods = np.zeros(num_digits)

        for j in range(num_digits):
            inner_prods[j] = np.linalg.norm(np.convolve(rec_i , average_signal[j, :],'same'))**2

        predicted_labels[i] = np.argmax(inner_prods) + 1

    print("Average spectrum comparison --- predicted labels: \n")

    # Storing predicted labels
    np.save("predicted_labels_avg.npy", predicted_labels)
    true_labels=np.array([1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2])
    print('The accuracy is:',(1-sum(abs(true_labels-predicted_labels))/len(true_labels))*100)
