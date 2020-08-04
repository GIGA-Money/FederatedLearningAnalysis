import sys
import pandas as pd
import numpy as np
import random
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
from glob import iglob
from sklearn.metrics import recall_score, accuracy_score, precision_score, \
    confusion_matrix, classification_report, precision_recall_curve
import scikitplot as skplt
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import torch
import torch.nn as nn
import os
from absl import flags
from absl import app

flags.DEFINE_integer('Input_dim', 115, 'the input dimension, used from getting the train data')
flags.DEFINE_string('Model_dir', os.path.dirname(os.path.abspath(__file__)), 'the current directory')
FLAGS = flags.FLAGS


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


def load_mal_data():
    df_mirai = pd.concat((pd.read_csv(f) for f in iglob('../data/**/mirai_attacks/*.csv', recursive=True)),
                         ignore_index=True)
    df_gafgyt = pd.DataFrame()
    for f in iglob('../data/**/gafgyt_attacks/*.csv', recursive=True):
        #    if 'tcp.csv' in f or 'udp.csv' in f:
        #        continue
        df_gafgyt = df_gafgyt.append(pd.read_csv(f), ignore_index=True)
    return df_mirai.append(df_gafgyt)
    # return df_mirai


def test_with_data(top_n_features, df_malicious, PATH):
    print("Testing")
    df = pd.concat((pd.read_csv(f) for f in iglob('../data/**/benign_traffic.csv', recursive=True)), ignore_index=True)
    fisher = pd.read_csv('../fisher.csv')
    features = fisher.iloc[0:int(top_n_features)]['Feature'].values
    df = df[list(features)]
    x_train, x_opt, x_test = np.split(df.sample(frac=1, random_state=17), [int(1 / 3 * len(df)), int(2 / 3 * len(df))])
    scaler = StandardScaler()
    scaler.fit(x_train.append(x_opt))

    print(f"Loading model")
    saved_model = Net(top_n_features)
    saved_model.load_state_dict(torch.load(PATH))
    saved_model.eval()
    # load_model(f'models/model_{top_n_features}.h5')
    with open(FLAGS.Model_dir + f'threshold_centralized/threshold_centralized_{top_n_features}') as t:
        tr = np.float64(t.read())
    print(f"Calculated threshold is {tr}")
    model = AnomalyModel(saved_model, tr, scaler)

    df_benign = pd.DataFrame(x_test, columns=df.columns)
    df_benign['malicious'] = 0
    df_malicious = df_malicious.sample(n=df_benign.shape[0], random_state=17)[list(features)]
    df_malicious['malicious'] = 1
    df = df_benign.append(df_malicious)
    X_test = df.drop(columns=['malicious']).values
    X_test_scaled = scaler.transform(X_test)
    Y_test = df['malicious']
    Y_pred = model.predict(torch.from_numpy(X_test_scaled).float())
    print(f"Accuracy:\n {accuracy_score(Y_test, Y_pred)}.")
    print(f"Recall:\n {recall_score(Y_test, Y_pred)}.")
    print(f'Precision score:\n{precision_score(Y_test, Y_pred)}.')
    print(f"confusion matrix:\n {confusion_matrix(Y_test, Y_pred)}.")
    skplt.metrics.plot_confusion_matrix(Y_test, Y_pred, text_fontsize="large")
    print(f"classification report:\n {classification_report(Y_test, Y_pred)}")
    print("precision_recall_curve: ")
    pre, recall, thresholds = precision_recall_curve(Y_test, Y_pred)
    print(f"precision: {pre}. recall: {recall}. thrshold: {thresholds}.")
    plt.show()

    print('explaining with LIME')
    for j in range(5):
        i = np.random.randint(0, X_test.shape[0])
        print(f'Explaining for record nr {i}')
        explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values, feature_names=df.drop(
            columns=['malicious']).columns.tolist(), discretize_continuous=True)
        exp = explainer.explain_instance(X_test[i], model.scale_predict_classes)
        exp.save_to_file(f'limecentralized/explanation{j}.html')
        print(exp.as_list())
        print('Actual class')
        print(Y_test.iloc[[i]])
    print("---------------------------------")


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


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


def main(argv):
    if len(argv) > 2:
        raise app.UsageError('Expected one command-line argument(s), '
                             'got: {}'.format(argv))
    top_n_features = FLAGS.Input_dim
    tr = " "
    tr_file = FLAGS.Model_dir + f"threshold_centralized/threshold_centralized_{top_n_features}"
    with open(tr_file) as file:
        tr = file.read()
    tr = tr[:5]
    PATH = FLAGS.Model_dir + f"PyModels/centralizedModel/centralized_base_{tr}.pt"
    test_with_data(top_n_features, load_mal_data(), PATH)
    os._exit(0)


if __name__ == '__main__':
    app.run(main)
