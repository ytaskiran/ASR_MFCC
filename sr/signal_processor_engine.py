import numpy as np
from sklearn.preprocessing import normalize

from .mfcc_processor import MFCCProcessor

class SignalProcessorEngine:

    def __init__(self, fft_size=2048, frame_hop_size=20, mel_filter_num=10, dct_filter_num=40, norm="l2"):
        self.params = dict()
        self.params["fft_size"] = fft_size
        self.params["frame_hop_size"] = frame_hop_size
        self.params["mel_filter_num"] = mel_filter_num
        self.params["dct_filter_num"] = dct_filter_num
        self.params["norm"] = norm


    def process(self, audio_signal, sample_freq=44100):

        self.mfcc_processor = MFCCProcessor(audio_signal=audio_signal)

        self.mfcc_processor.frame_audio(self.params["fft_size"], self.params["frame_hop_size"], sample_freq)
        self.mfcc_processor.convert_to_frequency(self.params["fft_size"])
        signal_power = self.mfcc_processor.calculate_power()
        filter_points, mel_freqs = self.mfcc_processor.get_filter_points(freq_min=0, freq_max=sample_freq/2, 
                                                                         mel_filter_num=self.params["mel_filter_num"], 
                                                                         fft_size=self.params["fft_size"],
                                                                         sample_freq=sample_freq)
        filters = self.mfcc_processor.get_filters(filter_points=filter_points, fft_size=self.params["fft_size"])
        filters = self.mfcc_processor.normalize_filters(filters, mel_freqs=mel_freqs, mel_filter_num=self.params["mel_filter_num"])
        signal_filtered = self.mfcc_processor.filter_signal(filters=filters, sig_power=signal_power)
        dct_filters = self.mfcc_processor.get_dct_filters(filter_num=self.params["dct_filter_num"], filter_len=self.params["mel_filter_num"])
        self.cepstral_coefficients = self.mfcc_processor.get_cepstral_coefficients(signal_filtered=signal_filtered, dct_filters=dct_filters)


    def get_cepstral_coefficients(self, normalized=True, mfcc_num=40):

        self.cepstral_coefficients[np.isnan(self.cepstral_coefficients)] = 0
        self.cepstral_coefficients[np.isinf(self.cepstral_coefficients)] = 0
        if normalized:
            return normalize(self.cepstral_coefficients[:mfcc_num,:], axis=1, norm=self.params["norm"])
        else:
            return self.cepstral_coefficients[:mfcc_num,:]

