# %%
import matplotlib
import matplotlib.pyplot as plt
from absl import app

# %%
# def accuracy_plt_multi_worker_e50_lr01(plt):

'''plt.style.use("ggplot")
title = "Accuracy"
acc = plt
acc.grid()
acc.title(f"Autoencoder {title} for different number of features")
acc.xlabel("Input Dimensions")
acc.ylabel(f"{title}")
xlista = [60, 65, 70, 75, 80]
ylista = [0.9876892088035784, 0.9892354631357564, 0.9885126929115199, 0.9880298587043455, 0.9893213661542107]
acc.plot(xlista, ylista)
acc.savefig(f"mutli_worker_acc_e8_lr001.png")
print("accuracy")
'''
# %%
'''title = "Loss with 60 features, learn rate at 0.01"
acc = plt
# acc.title(f"Autoencoder {title} for different number of features")
acc.title(title)
acc.xlabel("Input Dimensions")
acc.ylabel(f"{title}")
xlist = [0, 1, 2, 3, 4, 5, 6, 7]
ylist = [0.586, 0.558, 0.549, 0.468, 0.427, 0.404, 0.374, 0.351]
acc.plot(xlist, ylist)
acc.savefig(f"mutli_worker_loss_e8_lr001.png")
print("loss")
'''  #


# %%
#def precision_plt(plt):
'''    title = "Precision"
    precision = plt
    precision.style.use("ggplot")
    precision.grid()
    precision.title(f"Autoencoder {title} for different number of features")
    precision.xlabel("Input Dimension")
    precision.ylabel(f"{title}")
    xlistp = [6, 7, 8, 9, 10]
    ylistp = [0.9449794818213479, 0.9340637137148373, 0.9257056406757539, 0.9512947918730662, 0.911211849126572]
    precision.plot(xlistp, ylistp)
    precision.savefig(f"multi_worker_precision_e50_lr01_bs64.png")
    print("precision")
'''
# %%
#def recall_plt(plt):
'''    title = "Recall"
    recall = plt
    recall.style.use("ggplot")
    recall.grid()
    recall.title(f"Autoencoder {title} for different number of features")
    recall.xlabel("Epochs")
    recall.ylabel(f"{title}")
    xlistr = [6, 7, 8, 9, 10]
    ylistr = [0.7394235611244409, 0.7394650315471429, 0.7395005776237448, 0.7395183506620456, 0.7395124263159454]
    recall.plot(xlistr, ylistr)
    recall.savefig(f"multi_worker_recall_e50_lr01_bs64.png")
    print("recall")
    '''
# %%

#def f1_plt(plt):
    title = "F-Measure"
    f1 = plt
    f1.style.use("ggplot")
    f1.grid()
    f1.title(f"Autoencoder {title} for different number of features")
    f1.xlabel("Epochs")
    f1.ylabel(f"{title}")
    xlistf = [6, 7, 8, 9, 10]
    ylistf = [0.8296590255689277, 0.8254503610824536, 0.8221922888185563, 0.832143940429248, 0.8164325144546477]
    f1.plot(xlistf, ylistf)
    f1.savefig(f"multi_worker_f1_e50_lr01_bs64.png")
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
    matplotlib.use("pdf")
    # accuracy_plt_multi_worker_e50_lr01(plt)


if __name__ == '__main__':
    app.run(main)
