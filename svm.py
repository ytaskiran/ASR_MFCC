import os
import numpy as np
from sklearn import svm
import pickle

from sr.audio_signal import AudioSignal
from sr.signal_processor_engine import SignalProcessorEngine
from util.util import pad

word_list = ["Cappucino", "Coffee", "Espresso", "Hot_Chocolate", "Latte", "Macchiato", "Mocha", "Tea"]

def main():
    x_test = []
    y_test = []

    model = train()

    for n, word in enumerate(word_list):

        if (word == "Cappucino" or word == "Espresso"): end = 25
        else: end = 26
            
        for i in range(20, end):
            mfcc = get_coeffs(word, i)
            mfcc_padded = pad(mfcc).reshape(-1, )
            x_test.append(mfcc_padded)
            y_test.append(n)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    index = np.arange(len(x_test))
    np.random.shuffle(index)

    x_test = x_test[index]
    y_test = y_test[index]

    print(f"\n\nTest data number: {x_test.shape[0]}")
    print(f"Model prediction accuracy: {model.score(x_test, y_test)}")

    filename = os.getcwd() + "/model/svm.sav"
    pickle.dump(model, open(filename, "wb"))



def train():
    x_train = []
    y_train = []

    for n, word in enumerate(word_list):
        for i in range(1, 20):
            mfcc = get_coeffs(word, i)
            mfcc_padded = pad(mfcc).reshape(-1, )
            x_train.append(mfcc_padded)
            y_train.append(n)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    index = np.arange(len(x_train))
    np.random.shuffle(index)

    x_train = x_train[index]
    y_train = y_train[index]

    model = svm.SVC(kernel="poly", C=1, degree=2, tol=0.001, decision_function_shape="ovo")
    model.fit(x_train, y_train)

    print("-- Training is finished")

    return model


def get_coeffs(word, i):
    if (i < 10): audio_signal = AudioSignal(os.getcwd() + f"/data/{word}/00{i}.wav")
    else: audio_signal = AudioSignal(os.getcwd() + f"/data/{word}/0{i}.wav")
    audio_signal.normalize()

    engine = SignalProcessorEngine(fft_size=2048 ,frame_hop_size=20, dct_filter_num=40)
    engine.process(audio_signal.signal, sample_freq=audio_signal.sample_freq)
    mfcc = engine.get_cepstral_coefficients(normalized=True, mfcc_num=10)

    return mfcc



if __name__ == "__main__":
    main()