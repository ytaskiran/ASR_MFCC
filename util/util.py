import numpy as np
import matplotlib.pyplot as plt


def pad(mfcc, max_col=70):
    if (mfcc.shape[1] < max_col):
        mfcc = np.pad(mfcc, ((0, 0), (0, max_col - mfcc.shape[1])), "mean")
    else:
        mfcc = mfcc[:,:max_col]

    return mfcc


def plot_cepstral_coeffs(audio_signal, sample_rate, cepstral_coefficients):    
    plt.figure(figsize=(15,5))
    plt.plot(np.linspace(0, len(audio_signal) / sample_rate, num=len(audio_signal)), audio_signal)
    plt.imshow(cepstral_coefficients, aspect='auto', origin='lower')
    plt.show()