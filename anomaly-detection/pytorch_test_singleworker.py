import os
from glob import iglob

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import torch
import torch.nn as nn
from absl import app
from absl import flags
from sklearn.metrics import recall_score, accuracy_score, precision_score, \
    confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# %%
flags.DEFINE_integer("Input_dim", 115, "the input dimension, used from getting the train data")
flags.DEFINE_string("Current_dir", os.path.dirname(os.path.abspath(__file__)), "the current directory")
flags.DEFINE_float("Learn_rate", 0.001, "The rate of learning by the optimizer")
FLAGS = flags.FLAGS
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


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


# %%
def load_mal_data():
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob("../data/**/mirai_attacks/*.csv", recursive=True)),
                         ignore_index=True)
    df_gafgyt = pd.DataFrame()
    for f in iglob("../data/**/gafgyt_attacks/*.csv", recursive=True):
        #    if 'tcp.csv' in f or 'udp.csv' in f:
        #        continue
        df_gafgyt = df_gafgyt.append(pd.read_csv(f), ignore_index=True)
    return df_mirai.append(df_gafgyt)


# %%
def test_with_data(top_n_features, df_malicious, PATH):
    # %% loading data
    df, features, scaler, x_test, x_train = testing(top_n_features)
    # %% loading Model
    saved_model = load_model(PATH, top_n_features)
    # %% opening threshold
    with open(FLAGS.Current_dir +
              f"threshold_singleworker/threshold_federated_{top_n_features}_{FLAGS.Learn_rate}.txt") as t:
        tr = np.float64(t.read())
    print(f"Calculated threshold is {tr}")
    model = AnomalyModel(saved_model, tr, scaler)
    # %% pandas data grabbing
    df_benign = pd.DataFrame(x_test, columns=df.columns)
    df_benign["malicious"] = 0
    df_malicious = df_malicious.sample(n=df_benign.shape[0], random_state=17)[list(features)]
    df_malicious["malicious"] = 1
    df = df_benign.append(df_malicious)
    X_test = df.drop(columns=["malicious"]).values
    X_test_scaled = scaler.transform(X_test)
    Y_test = df["malicious"]
    Y_pred = model.predict(torch.from_numpy(X_test_scaled).float())
    # %% printing to console
    printing_press(Y_pred, Y_test)
    # %% writing to lime html files
    lime_writing(X_test, Y_test, df, model, x_train)


# %%
def lime_writing(X_test, Y_test, df, model, x_train):
    print("explaining with LIME\n---------------------------------")
    for j in range(5):
        i = np.random.randint(0, X_test.shape[0])
        print(f"Explaining for record nr {i}")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            x_train.values,
            feature_names=df.drop(columns=["malicious"]).columns.tolist(),
            discretize_continuous=True)
        exp = explainer.explain_instance(X_test[i], model.scale_predict_classes)
        exp.save_to_file(f"lime_singleworker/explanation{j}.html")
        print(exp.as_list())
        print("Actual class")
        print(Y_test.iloc[[i]])
    print("---------------------------------")


# %%
def printing_press(Y_pred, Y_test):
    print(f"Accuracy:\n {accuracy_score(Y_test, Y_pred)}.")
    print(f"Recall:\n {recall_score(Y_test, Y_pred)}.")
    print(f"Precision score:\n {precision_score(Y_test, Y_pred)}.")
    print(f"confusion matrix:\n {confusion_matrix(Y_test, Y_pred)}.")
    print(f"classification report:\n {classification_report(Y_test, Y_pred)}")
    skplt.metrics.plot_confusion_matrix(Y_test,
                                        Y_pred,
                                        title="single worker Test",
                                        text_fontsize="large")
    plt.show()


# %%
def load_model(PATH, top_n_features):
    print(f"Loading model")
    saved_model = Net(top_n_features)
    saved_model.load_state_dict(torch.load(PATH))
    saved_model.to(device)
    saved_model.eval()
    return saved_model


# %%
def testing(top_n_features):
    print("Testing")
    df = pd.concat((pd.read_csv(f) for f in iglob("../data/**/benign_traffic.csv",
                                                  recursive=True)), ignore_index=True)
    fisher = pd.read_csv("../fisher.csv")
    features = fisher.iloc[0:int(top_n_features)]["Feature"].values
    df = df[list(features)]
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1 / 3 * len(df)), int(2 / 3 * len(df))])
    scaler = StandardScaler()
    scaler.fit(x_train.append(x_opt))
    return df, features, scaler, x_test, x_train


# %% *UNUSED*
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


# %%
class AnomalyModel:
    def __init__(self, model, threshold, scaler):
        self.model = model
        self.threshold = threshold
        self.scaler = scaler

    def predict(self, x):
        x_pred = self.model(x)
        mse = np.mean(np.power(x.data.numpy() - x_pred.data.numpy(), 2), axis=1)
        y_pred = mse > self.threshold
        return y_pred.astype(int)

    def scale_predict_classes(self, x):
        x = self.scaler.transform(x)
        y_pred = self.predict(torch.from_numpy(x).float())
        classes_arr = []
        for e in y_pred:
            el = [0, 0]
            el[e] = 1
            classes_arr.append(el)

        return np.array(classes_arr)


# %%
def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}.")
    top_n_features = FLAGS.Input_dim
    learn_rate = FLAGS.Learn_rate
    tr = " "
    tr_file = FLAGS.Current_dir + f"threshold_singleworker/threshold_federated_{top_n_features}_{learn_rate}.txt"
    with open(tr_file) as file:
        tr = file.read()
    tr = tr[:5]
    PATH = FLAGS.Current_dir + f"PyModels/singleModel/single_worker_{tr}.pt"
    test_with_data(top_n_features, load_mal_data(), PATH)
    os._exit(0)


# %%
if __name__ == '__main__':
    app.run(main)