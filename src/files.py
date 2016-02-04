import os
import wave
import numpy
import csv
import cPickle as pickle
import librosa
import yaml


def load_audio(filename, mono=True, fs=44100):
    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        audio_file = wave.open(filename)

        # Audio info
        sample_rate = audio_file.getframerate()
        sample_width = audio_file.getsampwidth()
        number_of_channels = audio_file.getnchannels()
        number_of_frames = audio_file.getnframes()

        # Read raw bytes
        data = audio_file.readframes(number_of_frames)
        audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = numpy.empty((num_samples, number_of_channels, 4), dtype=numpy.uint8)
            raw_bytes = numpy.fromstring(data, dtype=numpy.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            array = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = numpy.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            array = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            array = numpy.mean(array, axis=0)

        # Convert int values into float
        array = array / float(2 ** (sample_width * 8 - 1) + 1)

        if (fs != sample_rate):
            array = librosa.core.resample(array, sample_rate, fs)
            sample_rate = fs

        return array, sample_rate

    elif file_extension == '.flac':
        array, sample_rate = librosa.load(filename, sr=fs, mono=mono)

        return array, sample_rate

    return None, None


def load_event_list(file):
    data = []
    with open(file, 'rt') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) == 2:
                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1])
                    }
                )
            elif len(row) == 3:
                data.append(
                    {
                        'event_onset': float(row[0]),
                        'event_offset': float(row[1]),
                        'event_label': row[2]
                    }
                )
            elif len(row) == 5:
                data.append(
                    {
                        'file': row[0],
                        'scene_label': row[1],
                        'event_onset': float(row[2]),
                        'event_offset': float(row[3]),
                        'event_label': row[4]
                    }
                )
    return data


def save_data(file,data):
    pickle.dump(data, open(file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def load_data(file):
    return pickle.load(open(file, "rb"))


def load_parameters(file):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            return yaml.load(f)
    else:
        raise IOError("Parameter file not found [%s]" % file)


def save_text(file, text):
    with open(file, "w") as text_file:
        text_file.write(text)


def load_text(file):
    f = open(file, 'r')
    return f.readlines()

