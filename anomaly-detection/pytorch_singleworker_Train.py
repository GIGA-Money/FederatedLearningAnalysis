# %%
import os
from glob import iglob

import numpy as np
import pandas as pd
import syft as sy
import torch
import torch.nn as nn
import torch.optim as optim
from absl import app
from absl import flags
from sklearn.preprocessing import StandardScaler
from syft.federated.floptimizer import Optims
from tqdm import tqdm

# %%
flags.DEFINE_integer("Batch_size", 64, "The size of the batch from a round of training")
flags.DEFINE_integer("Epochs", 5, "The number of rounds of training")
flags.DEFINE_float("Learn_rate", 0.001, "The rate of learning by the optimizer")
flags.DEFINE_integer("Input_dim", 115, "the input dimension, used from getting the train data")
flags.DEFINE_string("Current_dir", os.path.dirname(os.path.abspath(__file__)), "the current directory")
FLAGS = flags.FLAGS
hook = sy.TorchHook(torch)
v_hook = sy.VirtualWorker(hook=hook, id="v")
workers = ['v']


# %%
def get_train_data(top_n_features=10):
    print("Loading combined training data...")
    df = pd.concat((pd.read_csv(f) for f in iglob("../data/**/benign_traffic.csv", recursive=True)),
                   ignore_index=True)
    fisher = pd.read_csv("../fisher.csv")
    features = fisher.iloc[0:int(top_n_features)]["Feature"].values
    df = df[list(features)]
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
def train(net, x_train, batch_size, epochs, learn_rate):
    optimizer = optim.SGD(net.parameters(), lr=learn_rate)
    loss_function = nn.MSELoss()
    loss, batch_x, outputs = 0, 0, 0
    optims = Optims(workers, optim=optimizer)
    for epoch in range(epochs):
        for i in tqdm(range(0, len(x_train), batch_size)):
            batch_x = x_train[i:i + batch_size]
            batch_x = batch_x.send('v')
            net.send(batch_x.location)
            opt = optims.get_optim(batch_x.location.id)
            opt.zero_grad()
            outputs = net(batch_x)
            loss = loss_function(outputs, batch_x)
            loss.backward()
            opt.step()
            net.get()
        print(f"Epoch: {epoch}. Loss: {loss.get()}")
    return np.mean(np.power(batch_x.get().data.numpy() - outputs.get().data.numpy(), 2), axis=1)


# %%
def cal_threshold(mse, input_dim):
    print(f"mean is {mse.mean():.5f}")
    print(f"min is {mse.min():.5f}")
    print(f"max is {mse.max():.5f}")
    print(f"std is {mse.std():.5f}")
    tr = mse.mean() + mse.std()
    with open(f"threshold_singleworker/threshold_federated_{input_dim}_{FLAGS.Learn_rate}.txt", 'w') as t:
        t.write(str(tr))
    print(f"Calculated threshold is {tr:.5f}")
    return tr


# %%
def test(net, x_test, tr):
    x_test = x_test.send('v')
    net.send(x_test.location)
    x_test_predictions = net(x_test)
    net.get()
    print("Calculating MSE on test set...")
    mse_test = np.mean(np.power(x_test.get().data.numpy() - x_test_predictions.get().data.numpy(), 2), axis=1)
    over_tr = mse_test > tr
    false_positives = sum(over_tr)
    test_size = mse_test.shape[0]
    print(f"{false_positives} false positives on dataset without attacks with size {test_size}")


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
        return torch.softmax(x, dim=1)


# %%

def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")
    # %%
    input_dim = FLAGS.Input_dim
    net = Net(input_dim)
    # %%
    training_data, input_dim = get_train_data(input_dim)
    x_train, x_opt, x_test = np.split(
        training_data.sample(frac=1, random_state=1),
        [int(1 / 3 * len(training_data)),
         int(2 / 3 * len(training_data))])
    # %%
    x_train, x_opt, x_test = create_scalar(x_opt, x_test, x_train)
    # %%
    batch_size = FLAGS.Batch_size
    epochs = FLAGS.Epochs
    learn_rate = FLAGS.Learn_rate
    # %%
    mse = train(net=net,
                x_train=torch.from_numpy(x_train).float(),
                batch_size=batch_size,
                epochs=epochs,
                learn_rate=learn_rate)
    tr = cal_threshold(mse=mse, input_dim=input_dim)
    print(tr)
    # %%
    test(net,
         torch.from_numpy(x_test).float(), tr=tr)

    PATH = FLAGS.Current_dir + "PyModels/singleModel"
    torch.save(net.state_dict(), os.path.join(PATH, f"single_worker_{tr:.3f}.pt"))

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
