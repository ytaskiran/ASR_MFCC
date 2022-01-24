from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

class AudioSignal:

    def __init__(self, path):
        self.sample_freq, self.signal = wavfile.read(path)
        self.duration = self.signal.shape[0] / float(self.sample_freq)

        if (self.signal.ndim == 2):
            self.signal = self.signal[:,0]

        #print(len(self.signal[self.signal < 0.01]) / len(self.signal))
        #self.signal = self.signal[self.signal < 0.01]

    @property
    def signal(self):
        return self._signal

    @property
    def sample_freq(self):
        return self._sample_freq

    @property
    def duration(self):
        return self._duration

    @signal.setter
    def signal(self, val):
        self._signal = val

    @sample_freq.setter
    def sample_freq(self, val):
        self._sample_freq = val

    @duration.setter
    def duration(self, val):
        self._duration = val

    def get_info(self):
        print('\nSignal Datatype:', self.signal.dtype)
        print("Sampling rate: ", self.sample_freq)
        print("Shape of the signal: ", self.signal.shape)
        print('Signal duration:', round(self.duration, 2), 'seconds\n')

    def plot_timedomain_waveform(self):
        time = np.linspace(0., self.duration, self.signal.shape[0])
        plt.plot(time, self.signal)
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    def normalize(self):
        self.signal = self.signal / np.max(np.abs(self.signal))
        
    def pre_emphasis(self):
        pre_emphasis = 0.97
        self.signal = np.append(self.signal[0], self.signal[1:] - pre_emphasis * self.signal[:-1])
