# %%
import sys
from glob import iglob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from absl import flags
from absl import app
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

# %%
flags.DEFINE_integer('Batch_size', 64, 'The size of the batch from a round of training')
flags.DEFINE_integer('Epochs', 5, 'The number of rounds of training')
flags.DEFINE_float('Learn_rate', 0.001, 'The rate of learning by the optimizer')
flags.DEFINE_integer('Input_dim', 10, 'the input dimension, used from getting the train data')
flags.DEFINE_string('Current_dir', os.path.dirname(os.path.abspath(__file__)), 'the current directory')
FLAGS = flags.FLAGS
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
writer = SummaryWriter("B:/projects/GRA/FederatedLearningAnalysis/anomaly-detection/log/centralized")


# %%
def get_train_data(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('../fisher.csv')
    # y_train = []
    # with open("../data/labels.txt", 'r') as labels:
    #    for lines in labels:
    #        y_train.append(lines.rstrip())
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    # return df, y_train
    return df, top_n_features


# %%
def create_scalar(x_opt, x_test, x_train):
    scalar = StandardScaler()
    scalar.fit(x_train.append(x_opt))
    x_train = scalar.transform(x_train)
    x_opt = scalar.transform(x_opt)
    x_test = scalar.transform(x_test)
    return x_train, x_opt, x_test


# %%
def train(net, x_train, x_opt, batch_size, epochs, learn_rate):
    outputs = 0
    optimizer = optim.SGD(net.parameters(), lr=learn_rate)
    loss_function = nn.MSELoss()
    loss = 0
    batch_y = 0
    train_loss = 0
    for epoch in range(epochs):
        for i in tqdm(range(0, len(x_train), batch_size)):
            batch_y = x_train[i:i + batch_size]
            batch_y = batch_y.to(device)
            net.zero_grad()
            outputs = net(batch_y)
            train_loss = loss_function(outputs, batch_y)
            train_loss.backward()
            optimizer.step()  # Does the update

        print(f"Epoch: {epoch}. Train_Loss: {train_loss.item():.3f}.")

    return np.mean(np.power(batch_y.data.numpy() - outputs.data.numpy(), 2), axis=1)


# %%
def cal_threshold(mse, input_dim):
    # mse = np.mean(np.power(loss_val.real, 2), axis=1)
    print("mean is %.5f" % mse.mean())
    print("min is %.5f" % mse.min())
    print("max is %.5f" % mse.max())
    print("std is %.5f" % mse.std())

    tr = mse.mean() + mse.std()
    writer.add_scalar("threshold_over_learn_rate", tr, FLAGS.Learn_rate)
    with open(f'threshold_centralized/threshold_centralized_{input_dim}', 'w') as t:
        t.write(str(tr))
    print(f"Calculated threshold is {tr}")
    return tr


# %%
def test(net, x_test, tr):
    writer.add_graph(net, x_test)
    net.eval()
    x_test_predictions = net(x_test)
    print("Calculating MSE on test set...")
    mse_test = np.mean(np.power(x_test.data.numpy() - x_test_predictions.data.numpy(), 2), axis=1)
    over_tr = mse_test > tr
    false_positives = sum(over_tr)
    test_size = mse_test.shape[0]
    # writer.add_text("false_positives_over_learn_rate", false_positives, global_step=0.001)
    print(f"{false_positives:} false positives on dataset without attacks with size {test_size}")


'''
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(len(x_test))):
            output = net(x_test)
            test_loss += F.softmax(output).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(x_test.view_as(pred)).sum().item()
    test_loss /= len(x_test)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(x_test),
        100. * correct / len(x_test)))
    # writer.add_scalar("testing_accuracy__", round(correct / total, 3), global_step=i)
'''


# %%
class Net(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, int(0.75 * input_dim))
        self.fc2 = nn.Linear(int(0.75 * input_dim), int(0.5 * input_dim))
        self.fc3 = nn.Linear(int(0.5 * input_dim), int(0.33 * input_dim))
        self.fc4 = nn.Linear(int(0.33 * input_dim), int(0.25 * input_dim))
        self.fc5 = nn.Linear(int(0.25 * input_dim), int(0.33 * input_dim))
        self.fc6 = nn.Linear(int(0.33 * input_dim), int(0.5 * input_dim))
        self.fc7 = nn.Linear(int(0.5 * input_dim), int(0.75 * input_dim))
        self.fc8 = nn.Linear(int(0.75 * input_dim), input_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        x = torch.tanh(self.fc7(x))
        x = self.fc8(x)
        return x


def main(argv):
    if len(argv) > 2:
        raise app.UsageError('Expected one command-line argument(s), '
                             'got: {}'.format(argv))
    input_dim = FLAGS.Input_dim
    # %%
    net = Net(input_dim)
    net.to(device)
    # %%
    training_data, input_dim = get_train_data(input_dim)
    x_train, x_opt, x_test = np.split(training_data.sample(frac=1, random_state=1),
                                      [int(1 / 3 * len(training_data)),
                                       int(2 / 3 * len(training_data))])
    # %%
    x_train, x_opt, x_test = create_scalar(x_opt, x_test, x_train)
    # %%
    batch_size = FLAGS.Batch_size
    epochs = FLAGS.Epochs
    learn_rate = FLAGS.Learn_rate
    # %%
    mse = train(net,
                torch.from_numpy(x_train).float(),
                torch.from_numpy(x_opt).float(),
                batch_size,
                epochs,
                learn_rate=learn_rate)
    tr = cal_threshold(mse=mse, input_dim=input_dim)
    print(tr)
    # %%
    test(net,
         torch.from_numpy(x_test).float(),
         tr=tr)
    writer.close()
    PATH = FLAGS.Current_dir+"PyModels/centralizedModel"
    torch.save(net.state_dict(), os.path.join(PATH , f"centralized_base_{tr:.3f}."))
    os._exit(0)


if __name__ == '__main__':
    app.run(main)
