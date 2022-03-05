"""
CONV_COMPARISON FILE:
computes the score function  using the norm square of the convolution.
"""
import numpy as np
from idft import idft

if __name__ == '__main__':
    T = 1  # recording time
    fs = 8000  # sampling frequency

    # loads test set
    test_set = np.load("test_set.npy")

    # loads (DFTs of) training set
    training_set_DFT_values = np.load("spoken_digits_DFTs.npy")
    # Average spectra
    numberDigits = len(training_set_DFT_values)
    _, N = training_set_DFT_values[0].shape
    average_spectra = np.zeros((numberDigits, N), dtype=np.complex_)
    average_signal = np.zeros((numberDigits, N), dtype=np.complex_)

    """
    In[22]: bane
Out[22]: array(['1.000027337501943-7.331085223659654E-6j',
       '1.0023086995640738-1.8228368353755985E-4j',
       '-0.017014515914781394-0.2820013864855318j'], 
       dtype='|S41')

In [23]: bane.astype(dtype=complex)
Out[23]: 
array([ 1.00002734 -7.33108522e-06j,  1.00230870 -1.82283684e-04j,
       -0.01701452 -2.82001386e-01j])
    
    """

    for i in range(numberDigits):
        # Average of modulus of spectra
        average_spectra[i, :] = np.mean(np.absolute(training_set_DFT_values[i]), axis=0)
        iDFT = idft(average_spectra[i, :], fs, N)
        y_demod, Treal = iDFT.solve_ifft()

        """
        def fm_demod(x, df=1.0, fc=0.0):
    ''' Perform FM demodulation of complex carrier.

    Args:
        x (array):  FM modulated complex carrier.
        df (float): Normalized frequency deviation [Hz/V].
        fc (float): Normalized carrier frequency.

    Returns:
        Array of real modulating signal.
    '''

    # Remove carrier.
    n = sp.arange(len(x))
    rx = x*sp.exp(-1j*2*sp.pi*fc*n)

    # Extract phase of carrier.
    phi = sp.arctan2(sp.imag(rx), sp.real(rx))

    # Calculate frequency from phase.
    y = sp.diff(sp.unwrap(phi)/(2*sp.pi*df))

    return y
        
        """
        average_signal[i, :] = y_demod

    numberRecordings, N = test_set.shape
    predicted_labels = np.zeros(numberRecordings)

    for i in range(numberRecordings):
        rec_i = test_set[i, :]
        # We can use the norm of the ith signal to normalize its DFT
        energy_rec_i = np.linalg.norm(rec_i)
        """
        In NumPy, the np.linalg.norm() function is used to calculate one of the eight different matrix norms or one of 
        the vector norms.
        x: This is an input array.
        
        ord: This stands for “order”.
        
        axis: If the axis is an integer, then the vector norm is computed for the axis of x.If the axis is a 2-tuple, 
        the matrix norms of specified matrices are computed. If the axis is None, then either a vector norm (when x is 1-D) 
        or a matrix norm (when x is 2-D) is returned.
        
        dims: It receives a boolean value. If its value is​ true, then the axes that are normed over are left in 
        the result as dimensions with size one. Otherwise, the axes which are normed over are kept in the result.
        
        """
        rec_i /= energy_rec_i

        # Comparisons
        inner_prods = np.zeros(numberDigits)

        for j in range(numberDigits):
            inner_prods[j] = np.linalg.norm(np.convolve(rec_i , average_signal[j, :],'same'))**2

        predicted_labels[i] = np.argmax(inner_prods) + 1

    print("Average spectrum comparison --- predicted labels: \n")

    # Storing predicted labels
    np.save("predicted_labels_avg.npy", predicted_labels)
    true_labels=np.array([1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2])
    print('The accuracy is:',(1-sum(abs(true_labels-predicted_labels))/len(true_labels))*100)
