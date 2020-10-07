# %%
import matplotlib
import matplotlib.pyplot as plt
import os
from absl import app

# %%

xList = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
centAcc = [0.81247, 0.83164, 0.85245, 0.85637, 0.86294, 0.86284, 0.86218,
           0.85949, 0.85865, 0.98686, 0.98608, 0.98717, 0.98051, 0.98498, 0.98819, 0.99007, 0.98936, 0.98528, 0.98555,
           0.98560, 0.98574, 0.98559]
centRecall = [0.73956, 0.73960, 0.73959, 0.73959, 0.73952, 0.73954,
              0.73953, 0.73954, 0.73954, 0.99982, 0.99983, 0.99983, 0.99983, 0.99983, 0.99980, 0.99977, 0.99981,
              0.99984, 0.99983, 0.99981, 0.99982, 0.99982]
centPrecision = [0.86582, 0.90645, 0.95519, 0.96497, 0.98188, 0.98159,
                 0.97990, 0.97295, 0.97079, 0.97457, 0.97307, 0.97514, 0.96263, 0.97100, 0.97711, 0.98073, 0.97935,
                 0.97156, 0.97207, 0.97218, 0.97244, 0.97216]
centF1 = [0.79773, 0.81457, 0.83368, 0.83738, 0.84364, 0.84355, 0.84291,
          0.84034, 0.83953, 0.98703, 0.98627, 0.98733, 0.98088, 0.98520, 0.98832, 0.99016, 0.98947, 0.98550, 0.98575,
          0.98580, 0.98594, 0.98579]

multiAcc = [0.82832, 0.84076, 0.85975, 0.86133, 0.863257, 0.863316,
            0.863337, 0.860937, 0.860209, 0.989158, 0.988287, 0.988945, 0.988672, 0.9881039, 0.99008, 0.991658,
            0.991214, 0.989202, 0.9891258, 0.9877247, 0.988237, 0.98768]
multiRecall = [0.73949, 0.73957, 0.73956, 0.73954, 0.73948, 0.739494,
               0.739506, 0.739512, 0.73951, 0.99979, 0.999804, 0.999798, 0.9997985, 0.9997807, 0.999774, 0.999727,
               0.999745, 0.99978, 0.9997689, 0.999763, 0.99977, 0.99978]
multiPrecision = [0.9, 0.92723, 0.97360, 0.97769, 0.982765, 0.982904,
                  0.9829436, .976706, 0.974822, 0.97897, 0.977293, 0.978557, 0.978035, 0.9769649, 0.9807633, 0.9838501,
                  0.9829736, 0.979067, 0.97893, 0.976258, 0.977225, 0.97615]
multiF1 = [0.82839, 0.82283, 0.840597, 0.842104, 0.843941, 0.8440002,
           0.844022, 0.841718, 0.841021, 0.9892725, 0.9884209, 0.9890638, 0.988797, 0.9882412, 0.990177, 0.9917252,
           0.991288, 0.989315, 0.98924037, 0.98787, 0.9883713, 0.987827]
# %%
'''
xList = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
centAcc = [0.84872, 0.82681, 0.82286, 0.82262, 0.83980, 0.81247, 0.83164, 0.85245, 0.85637, 0.86294, 0.86284, 0.86218,
           0.85949, 0.85865, 0.98686, 0.98608, 0.98717, 0.98051, 0.98498, 0.98819, 0.99007, 0.98936, 0.98528, 0.98555,
           0.98560, 0.98574, 0.98559]
centRecall = [0.73944, 0.73948, 0.73954, 0.73954, 0.73954, 0.73956, 0.73960, 0.73959, 0.73959, 0.73952, 0.73954,
              0.73953, 0.73954, 0.73954, 0.99982, 0.99983, 0.99983, 0.99983, 0.99983, 0.99980, 0.99977, 0.99981,
              0.99984, 0.99983, 0.99981, 0.99982, 0.99982]
centPrecision = [0.94626, 0.89597, 0.88741, 0.88691, 0.92503, 0.86582, 0.90645, 0.95519, 0.96497, 0.98188, 0.98159,
                 0.97990, 0.97295, 0.97079, 0.97457, 0.97307, 0.97514, 0.96263, 0.97100, 0.97711, 0.98073, 0.97935,
                 0.97156, 0.97207, 0.97218, 0.97244, 0.97216]
centF1 = [0.83016, 0.81024, 0.80676, 0.80655, 0.82195, 0.79773, 0.81457, 0.83368, 0.83738, 0.84364, 0.84355, 0.84291,
          0.84034, 0.83953, 0.98703, 0.98627, 0.98733, 0.98088, 0.98520, 0.98832, 0.99016, 0.98947, 0.98550, 0.98575,
          0.98580, 0.98594, 0.98579]

multiAcc = [0.85555, 0.85605, 0.84154, 0.84629, 0.84674, 0.82832, 0.84076, 0.85975, 0.86133, 0.863257, 0.863316,
            0.863337, 0.860937, 0.860209, 0.989158, 0.988287, 0.988945, 0.988672, 0.9881039, 0.99008, 0.991658,
            0.991214, 0.989202, 0.9891258, 0.9877247, 0.988237, 0.98768]
multiRecall = [0.73943, 0.73938, 0.73948, 0.73942, 0.73947, 0.73949, 0.73957, 0.73956, 0.73954, 0.73948, 0.739494,
               0.739506, 0.739512, 0.73951, 0.99979, 0.999804, 0.999798, 0.9997985, 0.9997807, 0.999774, 0.999727,
               0.999745, 0.99978, 0.9997689, 0.999763, 0.99977, 0.99978]
multiPrecision = [0.9631, 0.96441, 0.92914, 0.94042, 0.94144, 0.94158, 0.92723, 0.97360, 0.97769, 0.982765, 0.982904,
                  0.9829436, .976706, 0.974822, 0.97897, 0.977293, 0.978557, 0.978035, 0.9769649, 0.9807633, 0.9838501,
                  0.9829736, 0.979067, 0.97893, 0.976258, 0.977225, 0.97615]
multiF1 = [0.83657, 0.83704, 0.82353, 0.8279, .82832, 0.82839, 0.82283, 0.840597, 0.842104, 0.843941, 0.8440002,
           0.844022, 0.841718, 0.841021, 0.9892725, 0.9884209, 0.9890638, 0.988797, 0.9882412, 0.990177, 0.9917252,
           0.991288, 0.989315, 0.98924037, 0.98787, 0.9883713, 0.987827]'''


# %%
def precision_plt(plt):
    title = "Precision"
    precision = plt
    precision.style.use("ggplot")
    precision.grid()
    precision.title(f"Centralized {title} for different number of features")
    precision.xlabel("Input Dimension")
    precision.ylabel(f"{title}")
    xlistp = xList
    ylistp = centPrecision
    precision.plot(xlistp, ylistp)
    precision.savefig(f"cent_precision_e50_lr0001_bs128.png")
    print("cent precision")


# %%
def recall_plt(plt):
    title = "Recall"
    recall = plt
    recall.style.use("ggplot")
    recall.grid()
    recall.title(f"Centralized {title} for different number of features")
    recall.xlabel("Epochs")
    recall.ylabel(f"{title}")
    xlistr = xList
    ylistr = centRecall
    recall.plot(xlistr, ylistr)
    recall.savefig(f"cent_recall_e50_lr0001_bs128.png")
    print("cent recall")


# %%
def f1_plt(plt):
    title = "F-Measure"
    f1 = plt
    f1.style.use("ggplot")
    f1.grid()
    f1.title(f"Centralized {title} for different number of features")
    f1.xlabel("Epochs")
    f1.ylabel(f"{title}")
    xlistf = xList
    ylistf = centF1
    f1.plot(xList, multiF1)
    f1.savefig(f"cent_f1_e50_lr0001_bs128.png")
    print("cent F-measure")


# %%
def accuracy_plt_(plt):
    plt.style.use("ggplot")
    title = "Accuracy"
    acc = plt
    acc.grid()
    acc.title(f"Centralized {title} for different number of features")
    acc.xlabel("Input Dimensions")
    acc.ylabel(f"{title}")
    xlista = xList
    ylista = centAcc
    acc.plot(xlista, ylista)
    acc.savefig(f"cent_acc_e50_lr001_bs128.png")
    print("cent accuracy")


# %%
def precision_plt_multi(plt):
    title = "Precision"
    precision = plt
    precision.style.use("ggplot")
    precision.grid()
    precision.title(f"Multi worker {title} for different number of features")
    precision.xlabel("Input Dimension")
    precision.ylabel(f"{title}")
    xlistp = xList
    ylistp = multiPrecision
    precision.plot(xlistp, ylistp)
    precision.savefig(f"multi_worker_precision_e50_lr0001_bs128.png")
    print("multi precision")


# %%
def recall_plt_multi(plt):
    title = "Recall"
    recall = plt
    recall.style.use("ggplot")
    recall.grid()
    recall.title(f"Multi worker {title} for different number of features")
    recall.xlabel("Epochs")
    recall.ylabel(f"{title}")
    xlistr = xList
    ylistr = multiRecall
    recall.plot(xlistr, ylistr)
    recall.savefig(f"multi_worker_recall_e50_lr0001_bs128.png")
    print("Multi recall")


# %%
def f1_plt_multi(plt):
    title = "F-Measure"
    f1 = plt
    f1.style.use("ggplot")
    f1.grid()
    f1.title(f"Multi Worker {title} for different number of features")
    f1.xlabel("Epochs")
    f1.ylabel(f"{title}")
    xlistf = xList
    ylistf = multiF1
    f1.plot(xList, multiF1)
    f1.savefig(f"multi_worker_f1_e50_lr0001_bs128.png")
    print("Multi F-measure")


# %%
def accuracy_plt_multi_worker_multi(plt):
    plt.style.use("ggplot")
    title = "Accuracy"
    acc = plt
    acc.grid()
    acc.title(f"Multi Worker {title} for different number of features")
    acc.xlabel("Input Dimensions")
    acc.ylabel(f"{title}")
    xlista = xList
    ylista = multiAcc
    acc.plot(xlista, ylista)
    acc.savefig(f"mutli_worker_acc_e50_lr001_bs128.png")
    print("Multi accuracy")


# %%
def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}.")
    plt.style.use("ggplot")
    matplotlib.use("pdf")
    #accuracy_plt_multi_worker_multi(plt)
    #f1_plt_multi(plt)
    #recall_plt_multi(plt)
    precision_plt_multi(plt)


    #accuracy_plt_(plt)
    #f1_plt(plt)
    #precision_plt(plt)
    #recall_plt(plt)

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
