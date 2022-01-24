import warnings
import numpy as np
from scipy.signal import get_window
from scipy.fftpack import fft


class MFCCProcessor:

    def __init__(self, audio_signal):
        self.signal = audio_signal

    @property
    def signal(self):
        return self._signal

    @property
    def frame(self):
        return self._frame

    @property
    def signal_freq(self):
        return self._signal_freq

    @signal.setter
    def signal(self, val):
        self._signal = val

    @frame.setter
    def frame(self, val):
        self._frame = val

    @signal_freq.setter
    def signal_freq(self, val):
        self._signal_freq = val

    @signal.deleter
    def signal(self):
        del self._signal 


    def frame_audio(self, fft_size=2048, hop_size=10, sample_rate=44100):
        self.signal = np.pad(self.signal, fft_size // 2, mode="reflect")
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(self.signal) - fft_size) / frame_len) + 1
        self.frames = np.zeros((frame_num, fft_size))

        for i in range(frame_num):
            self.frames[i] = self.signal[i * frame_len : i * frame_len + fft_size]

    
    def convert_to_frequency(self, fft_size):
        window = get_window("hann", fft_size, fftbins=True)
        signal_win = self.frames * window
        signal_win_T = np.transpose(signal_win)
        signal_freq = np.empty((int(1 + fft_size // 2), signal_win_T.shape[1]), dtype=np.complex64, order='F')

        for i in range(signal_freq.shape[1]):
            signal_freq[:, i] = fft(signal_win_T[:, i], axis=0)[:signal_freq.shape[0]]

        self.signal_freq = np.transpose(signal_freq)

    
    def get_filter_points(self, freq_min, freq_max, mel_filter_num, 
                          fft_size, sample_freq=44100):
        mel_freq_min = self.freq_to_mel(freq_min)
        mel_freq_max = self.freq_to_mel(freq_max)

        mels = np.linspace(mel_freq_min, mel_freq_max, num = mel_filter_num+2)
        freqs = self.mel_to_freq(mels)

        return np.floor((fft_size + 1) / sample_freq * freqs).astype(int), freqs


    def get_filters(self, filter_points, fft_size):
        filters = np.zeros((len(filter_points) - 2, int(fft_size/2 + 1)))
        for i in range(len(filter_points)-2):
            filters[i, filter_points[i] : filter_points[i + 1]] = np.linspace(0, 1, filter_points[i + 1] - filter_points[i])
            filters[i, filter_points[i + 1] : filter_points[i + 2]] = np.linspace(1, 0, filter_points[i + 2] - filter_points[i + 1])

        return filters

    def normalize_filters(self, filters, mel_freqs, mel_filter_num):
        enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
        filters *= enorm[:, np.newaxis]

        return filters


    def filter_signal(self, filters, sig_power):
        warnings.filterwarnings("ignore")
        return (10.0 * np.log10(np.dot(filters, np.transpose(sig_power))))


    def get_dct_filters(self, filter_num, filter_len): # DCT-III
        dct_filters = np.empty((filter_num, filter_len))
        dct_filters[0, :] = 1.0 / np.sqrt(filter_len)

        samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

        for i in range(1, filter_num):
            dct_filters[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)

        return dct_filters

    def get_cepstral_coefficients(self, signal_filtered, dct_filters):
        return np.dot(dct_filters, signal_filtered)


    def calculate_power(self):
        return np.square(np.abs(self.signal_freq))

    def freq_to_mel(self, freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    def mel_to_freq(self, mels):
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

