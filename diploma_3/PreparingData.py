import torch
import numpy as np


def prep_data_for_kan(dataset: dict):
    dataset['train_input'] = torch.flatten(torch.from_numpy(np.array(dataset['train_input'])), start_dim=1)
    dataset['test_input'] = torch.flatten(torch.from_numpy(np.array(dataset['test_input'])), start_dim=1)
    dataset['train_label'] = torch.LongTensor(dataset['train_label'])
    dataset['test_label'] = torch.LongTensor(dataset['test_label'])

    return dataset


def prep_data_for_cnn(dataset: dict):
    train = []
    test = []

    for i in range(len(dataset['train_input'])):
        train.append([dataset['train_input'][i]])

        j = dataset['train_label'][i]
        tmp = [0. for _ in range(j)] + [1.] + [0. for _ in range(9 - j)]
        dataset['train_label'][i] = tmp

    for i in range(len(dataset['test_input'])):
        test.append([dataset['test_input'][i]])

        j = dataset['test_label'][i]
        tmp = [0. for _ in range(j)] + [1.] + [0. for _ in range(9 - j)]
        dataset['test_label'][i] = tmp

    dataset['train_input'] = torch.from_numpy(np.array(train))
    dataset['test_input'] = torch.from_numpy(np.array(test))
    dataset['train_label'] = torch.from_numpy(np.array(dataset['train_label']))
    dataset['test_label'] = torch.from_numpy(np.array(dataset['test_label']))

    return dataset
