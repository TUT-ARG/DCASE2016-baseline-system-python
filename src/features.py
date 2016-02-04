import numpy
import librosa
import scipy

def feature_extraction(y=None, fs=None, statistics=True, include_mfcc0=True, include_delta=True, include_acceleration=True, mfcc_params=None, delta_params=None, acceleration_params=None):
    # Extract features, Mel Frequency Cepstral Coefficients
    eps = numpy.spacing(1)

    # Windowing function
    if mfcc_params['window'] == 'hamming_asymmetric':
        window = scipy.signal.hamming(mfcc_params['n_fft'], sym=False)
    elif mfcc_params['window'] == 'hamming_symmetric':
        window = scipy.signal.hamming(mfcc_params['n_fft'], sym=True)
    elif mfcc_params['window'] == 'hann_asymmetric':
        window = scipy.signal.hann(mfcc_params['n_fft'], sym=False)
    elif mfcc_params['window'] == 'hann_symmetric':
        window = scipy.signal.hann(mfcc_params['n_fft'], sym=True)
    else:
        window = None

    # Calculate Static Coefficients
    magnitude_spectrogram = numpy.abs(librosa.stft(y + eps, n_fft=mfcc_params['n_fft'], win_length=mfcc_params['win_length'], hop_length=mfcc_params['hop_length'], window=window))**2
    mel_basis = librosa.filters.mel(sr=fs, n_fft=mfcc_params['n_fft'], n_mels=mfcc_params['n_mels'], fmin=mfcc_params['fmin'], fmax=mfcc_params['fmax'], htk=mfcc_params['htk'])
    mel_spectrum = numpy.dot(mel_basis, magnitude_spectrogram)
    mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum))

    # Collect the feature matrix
    feature_matrix = mfcc
    if include_delta:
        # Delta coefficients
        mfcc_delta = librosa.feature.delta(mfcc, **delta_params)

        # Add Delta Coefficients to feature matrix
        feature_matrix = numpy.vstack((feature_matrix, mfcc_delta))

    if include_acceleration:
        # Acceleration coefficients (aka delta)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2, **acceleration_params)

        # Add Acceleration Coefficients to feature matrix
        feature_matrix = numpy.vstack((feature_matrix, mfcc_delta2))


    if not include_mfcc0:
        # Omit mfcc0
        feature_matrix = feature_matrix[1:, :]

    feature_matrix = feature_matrix.T

    # Collect into data structure
    if statistics:
        return {
            'feat': feature_matrix,
            'stat': {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }
        }
    else:
        return {
            'feat': feature_matrix}


class FeatureNormalizer(object):
    def __init__(self, feature_matrix=None):
        if feature_matrix is None:
            self.N = 0
            self.mean = 0
            self.S1 = 0
            self.S2 = 0
            self.std = 0
        else:
            self.mean = numpy.mean(feature_matrix, axis=0)
            self.std = numpy.std(feature_matrix, axis=0)
            self.N = feature_matrix.shape[0]
            self.S1 = numpy.sum(feature_matrix, axis=0)
            self.S2 = numpy.sum(feature_matrix ** 2, axis=0)
            self.finalize()

    def __enter__(self):
        self.N = 0
        self.mean = 0
        self.S1 = 0
        self.S2 = 0
        self.std = 0
        return self

    def __exit__(self, type, value, traceback):
        self.finalize()

    def accumulate(self, stat):
        self.N += stat['N']
        self.mean += stat['mean']
        self.S1 += stat['S1']
        self.S2 += stat['S2']

    def finalize(self):
        # Finalize statistics
        self.mean = self.S1 / self.N
        self.std = numpy.sqrt((self.N * self.S2 - (self.S1 * self.S1)) / (self.N * (self.N - 1)))

        # In case we have very brain-death material we get std = Nan => 0.0
        self.std = numpy.nan_to_num(self.std)

        self.mean = numpy.reshape(self.mean, [1, -1])
        self.std = numpy.reshape(self.std, [1, -1])

    def normalize(self, feature_matrix):
        return (feature_matrix - self.mean) / self.std
