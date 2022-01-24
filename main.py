import os
import argparse
from scipy import signal
import pickle

from sr.audio_signal import AudioSignal
from sr.signal_processor_engine import SignalProcessorEngine
from util.util import *

word_list = ["Cappucino", "Coffee", "Espresso", "Hot_Chocolate", "Latte", "Macchiato", "Mocha", "Tea"]

def main(path):

    model_file = os.getcwd() + "/model/svm.sav"
    svm_model = pickle.load(open(model_file, "rb"))

    audio_signal = AudioSignal(path)
    audio_signal.get_info()
    audio_signal.normalize()
    #audio_signal.plot_timedomain_waveform()

    engine = SignalProcessorEngine(fft_size=2048 ,frame_hop_size=20, dct_filter_num=40)
    engine.process(audio_signal=audio_signal.signal, sample_freq=audio_signal.sample_freq)
    mfcc = engine.get_cepstral_coefficients(normalized=True, mfcc_num=10)
    mfcc_padded = pad(mfcc).reshape(1, -1)

    #plot_cepstral_coeffs(audio_signal.signal, audio_signal.sample_freq, mfcc)

    [res] = svm_model.predict(mfcc_padded)
    print(f"Prediction: {word_list[res]}")
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", "-f", type=str, required=True,
                        help="Path to the audio (.wav) file to be processed")

    args = parser.parse_args()

    filepath = os.getcwd() + args.audio_file
    main(filepath)