import torch
from kan import KAN
from tqdm import tqdm
from PreparingData import *
from GettingData import get_dataset
from CNNModel import CNN, Net

# 3 эпоха 'train_acc': [0.6364780068397522], 'test_acc': [0.36082473397254944]

# Options
start_epoch = 0
num_epoch = 30
batch_size = 128

isKAN = False

dataset = get_dataset()

if isKAN:
    model = KAN(width=[16384, 10], grid=5, k=3)

    if start_epoch != 0:
        model.load_ckpt(f'KAN_{start_epoch}.pth', 'results')

    dataset = prep_data_for_kan(dataset)

    def train_acc():
        mean = 0
        loader = torch.utils.data.DataLoader(list(zip(dataset['train_input'], dataset['train_label'])),
                                             batch_size=batch_size, shuffle=False)
        for input_batch, label_batch in loader:
            mean += torch.mean((torch.argmax(model(input_batch), dim=1) == label_batch).float()) * len(label_batch)
        return mean / len(dataset['train_label'])


    def test_acc():
        mean = 0
        loader = torch.utils.data.DataLoader(list(zip(dataset['test_input'], dataset['test_label'])),
                                             batch_size=batch_size, shuffle=False)
        for input_batch, label_batch in loader:
            mean += torch.mean((torch.argmax(model(input_batch), dim=1) == label_batch).float()) * len(label_batch)
        return mean / len(dataset['test_label'])


    for epoch in tqdm(range(start_epoch + 1, start_epoch + num_epoch + 1)):
        results = model.train(dataset, opt="LBFGS", steps=1, metrics=(train_acc, test_acc),
                              loss_fn=torch.nn.CrossEntropyLoss(), batch=batch_size)

        model.save_ckpt(f'KAN_{epoch}.pth', 'results')

        print(results)

else:
    model = CNN()

    if start_epoch != 0:
        model.load_state_dict(torch.load(f'results/CNN_{start_epoch}.pth'))

    dataset = prep_data_for_cnn(dataset)

    test_loader = torch.utils.data.DataLoader(list(zip(dataset['test_input'], dataset['test_label'])),
                                             batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(start_epoch + 1, start_epoch + num_epoch + 1):
        train_loader = torch.utils.data.DataLoader(list(zip(dataset['train_input'], dataset['train_label'])),
                                                   batch_size=batch_size, shuffle=True)
        train_loss = []
        train_acc = 0
        for input_batch, label_batch in train_loader:
            label_pred = model(input_batch)
            loss = criterion(label_pred, label_batch)
            train_loss.append(loss.item())
            train_acc += torch.mean(
                (torch.argmax(label_pred, dim=1) == torch.argmax(label_batch, dim=1)).float()) * len(label_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            test_loss = []
            test_acc = 0
            for input_batch, label_batch in test_loader:
                label_pred = model(input_batch)
                loss = criterion(label_pred, label_batch)
                test_loss.append(loss.item())
                test_acc += torch.mean(
                    (torch.argmax(label_pred, dim=1) == torch.argmax(label_batch, dim=1)).float()) * len(label_batch)

        print(f'train_loss: {np.mean(train_loss)}, test_loss: {np.mean(test_loss)}, '
              f'train_acc: {train_acc / len(dataset["train_label"])}, '
              f'test_acc: {test_acc / len(dataset["test_label"])}')

        torch.save(model.state_dict(), f'results/CNN_{epoch}.pth')
