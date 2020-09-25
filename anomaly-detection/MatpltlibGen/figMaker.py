# %%
import matplotlib.pyplot as plt
from absl import app


def accuracy_plt(plt):
    title = "Accuracy"
    acc = plt
    acc.title(f"Measure of {title} across Input Dimension")
    acc.xlabel("Input Dimensions")
    acc.ylabel(f"{title}")
    acc
    print("accuracy")


def precision_plt(plt):
    title = "Precision"
    precision = plt
    precision.title(f"Measure of {title} across Input Dimension")
    precision.xlabel("Epochs")
    precision.ylabel(f"{title}")
    print("precision")


def recall_plt(plt):
    title = "Recall"
    recall = plt
    recall.title(f"Measure of {title} across Input Dimension")
    recall.xlabel("Epochs")
    recall.ylabel(f"{title}")
    print("recall")


def f1_plt(plt):
    title = "F-Measure"
    f1 = plt
    f1.title(f"Measure of {title} across Input Dimension")
    f1.xlabel("Epochs")
    f1.ylabel(f"{title}")
    print("F-measure")

def get_input_dim(input_count):
    input_dim_list = []
    
    print("input dim count")
    return input_dim_list

def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}.")
    plt.style.use("ggplot")


if __name__ == '__main__':
    app.run(main)
