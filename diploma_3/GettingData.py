from librosa import load, effects, feature, amplitude_to_db
import os
import numpy as np

# Current:
# amplitude_to_db: ref = np.max(training)
# normalization: minmax(training)

# Options
sr = 22050
step_multiplier = 30
dataset_path = "./dataset"
test_percent = 20

n_mels = 128
steps = 128
data_percent = 100


# Получить mel-спектрограмму одного wav файла
def get_mels(audio_file_path: str):
    global sr, step_multiplier, n_mels, steps

    y, _ = load(audio_file_path, sr=sr)
    y, _ = effects.trim(y)

    step_size = step_multiplier * sr
    hop_length = step_size // (steps - 1)
    n_fft = hop_length * 4

    return [feature.melspectrogram(y=y[step * step_size: (step + 1) * step_size], sr=sr, n_fft=n_fft,
                                   hop_length=hop_length, n_mels=n_mels) for step in range(len(y) // step_size)]


# Получить mel-спектрограммы всех wav файлов одного жанра
def get_mels_from_directory(directory: str):
    global data_percent

    files = os.listdir(directory)
    end = len(files) * data_percent // 100

    res = []
    for file in files[:end]:
        res += get_mels(f'{directory}/{file}')
    return res


# Привести все полученные mel-спектрограммы к децибелам
def dataset_amplitude_to_db(data, ref):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = amplitude_to_db(data[i][j], ref=ref)
    return data


# Нормализация полученных данных
def data_normalization(data, minimum, maximum):
    return (data - minimum) / (maximum - minimum)


# Получение всего датасета готового для входа в нейронную сеть
def get_dataset():
    global dataset_path, test_percent

    train = []
    test = []
    dataset = {'train_input': [], 'test_input': [], 'train_label': [], 'test_label': []}

    for label, directory in enumerate(os.listdir(dataset_path)):
        data = get_mels_from_directory(f'{dataset_path}/{directory}')
        train_len = len(data) - len(data) * test_percent // 100

        train += data[:train_len]
        test += data[train_len:]
        dataset['train_label'] += [label] * train_len
        dataset['test_label'] += [label] * (len(data) - train_len)

    # Вынести потом значение в константу, чтобы было
    amplitude_maximum = np.max(train)
    train = dataset_amplitude_to_db(train, amplitude_maximum)
    test = dataset_amplitude_to_db(test, amplitude_maximum)

    # Вынести потом значения в константы, чтобы было
    minimum = np.min(train)
    maximum = np.max(train)

    dataset['train_input'] = data_normalization(train, minimum, maximum)
    dataset['test_input'] = data_normalization(test, minimum, maximum)

    return dataset


if __name__ == '__main__':
    data_percent = 100
    dataset = get_dataset()
    print(np.min(dataset['train_input']), np.max(dataset['train_input']))
    print(np.min(dataset['test_input']), np.max(dataset['test_input']))
    exit(0)
