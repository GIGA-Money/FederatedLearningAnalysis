# %%
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from absl import app

# %%

xList = [10, 15, 20, 25, 30, 35, 40, 45, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 65, 70, 75, 80, 85, 90, 95, 100,
         105, 110, 115]
centAcc = [0.81247, 0.83164, 0.85245, 0.85637, 0.86294, 0.86284, 0.86218,
           0.85949, 0.85865, 0.85870, 0.85926, 0.98802, 0.98729,
           0.98686, 0.98686, 0.98693, 0.98688, 0.98664,
           0.98608, 0.98717, 0.98051, 0.98498, 0.98819, 0.99007, 0.98936, 0.98528, 0.98555,
           0.98560, 0.98574, 0.98559]
centRecall = [0.73956, 0.73960, 0.73959, 0.73959, 0.73952, 0.73954,
              0.73953, 0.73954, 0.73954, 0.73954, 0.73955, 0.99981, 0.99981,
              0.99982, 0.99982, 0.99983, 0.99985, 0.99985,
              0.99983, 0.99983, 0.99983, 0.99983, 0.99980, 0.99977, 0.99981,
              0.99984, 0.99983, 0.99981, 0.99982, 0.99982]

centPrecision = [0.86582, 0.90645, 0.95519, 0.96497, 0.98188, 0.98159,
                 0.97990, 0.97295, 0.97079, 0.97092, 0.97236, 0.97678, 0.97539,
                 0.97457, 0.97455, 0.97469, 0.97458, 0.97411,
                 0.97307, 0.97514, 0.96263, 0.97100, 0.97711, 0.98073, 0.97935,
                 0.97156, 0.97207, 0.97218, 0.97244, 0.97216]
centF1 = [0.79773, 0.81457, 0.83368, 0.83738, 0.84364, 0.84355, 0.84291,
          0.84034, 0.83953, 0.83958, 0.84013, 0.98816, 0.98745,
          0.98703, 0.98703, 0.98710, 0.98705, 0.98681,
          0.98627, 0.98733, 0.98088, 0.98520, 0.98832, 0.99016, 0.98947, 0.98550, 0.98575,
          0.98580, 0.98594, 0.98579]

multiAcc = [0.82832, 0.84076, 0.85975, 0.86133, 0.86325, 0.86331,
            0.86333, 0.86093, 0.86020, 0.86021, 0.86009, 0.98935, 0.98893,
            0.98915, 0.98906, 0.98926, 0.98901, 0.98911,
            0.98828, 0.98894, 0.98867, 0.98810, 0.99008, 0.99165,
            0.99121, 0.98920, 0.98912, 0.98772, 0.98823, 0.98768]
multiRecall = [0.73949, 0.73957, 0.73956, 0.73954, 0.73948, 0.73949,
               0.73950, 0.73951, 0.73951, 0.73951, 0.73951, 0.99973, 0.999798,
               0.99979, 0.99980, 0.99980, 0.99980, 0.99980,
               0.99980, 0.99979, 0.99979, 0.99978, 0.99977, 0.99972,
               0.99974, 0.99978, 0.99976, 0.99976, 0.99977, 0.99978]

multiPrecision = [0.94158, 0.92723, 0.97360, 0.97769, 0.98276, 0.98290,
                  0.98294, 0.97670, 0.97482, 0.97483, 0.97453, 0.97940, 0.97853,
                  0.97897, 0.97877, 0.97915, 0.97868, 0.97888,
                  0.97729, 0.97855, 0.97803, 0.97696, 0.98076, 0.98385,
                  0.98297, 0.97906, 0.97893, 0.97625, 0.97722, 0.97615]
multiF1 = [0.82839, 0.82283, 0.840597, 0.842104, 0.843941, 0.8440002,
           0.844022, 0.841718, 0.841021, 0.841024, 0.840914, 0.9894662, 0.9890522,
           0.9892725, 0.989179, 0.989374, 0.9891305, 0.9892320,
           0.9884209, 0.9890638, 0.988797, 0.9882412, 0.990177, 0.9917252,
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
    precision.legend(loc="lower right", framealpha=1.0, facecolor='white')
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
    recall.legend(loc="lower right", framealpha=1.0, facecolor='white')
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
    f1.legend(loc="lower right", framealpha=1.0, facecolor='white')
    f1.savefig(f"cent_f1_e50_lr0001_bs128.png")
    print("cent F-measure")


# %%
def accuracy_avg_bar(plt):
    cent_list = centAcc
    multi_list = multiAcc
    cent_avg = np.average(cent_list)
    multi_avg = np.average(multi_list)
    objects = ("Centralized", "Multi Worker")
    y_pos = np.arange(len(objects))
    avg_bar = plt
    avg_bar.style.use("default")
    avg_bar.ylim(90, 95)
    avg_bar.bar(y_pos, [cent_avg * 100, multi_avg * 100], align="center", color=("orange", "cornflowerblue"))
    avg_bar.xticks(y_pos, objects)
    avg_bar.ylabel("Average Accuracy")
    avg_bar.savefig(f"comparison_bar_avg_acc_e50_lr0001_bs128.png")
    print(f"AVG bar {cent_avg * 100, multi_avg * 100}")


# %%
def accuracy_plt_multi(plt):
    title = "Accuracy"
    acc = plt
    acc.style.use("default")
    acc.grid()
    acc.title("")  # (f"{title}")# for different number of features")
    acc.xlabel("Input Dimensions")
    acc.ylabel(f"{title}")
    half = len(xList) >> 1
    xlista = xList[half:]
    ylista = centAcc[half:]
    ylistmulti = multiAcc[half:]
    # acc.plot(xlista, ylista, label="Centralized")
    # acc.plot(xlista, ylistmulti, label="Multi Worker")
    acc.scatter(xlista, ylista, c="orange", label="Centralized")
    acc.scatter(xlista, ylistmulti, c="cornflowerblue", label="Multi Worker")
    acc.legend(loc="lower right", framealpha=1.0, facecolor='white')
    acc.savefig(f"comparison_scatter_acc_e50_lr001_bs128.png")
    print("cent accuracy")


# %%
def precision_plt_multi(plt):
    title = "Precision"
    precision = plt
    precision.style.use("default")
    precision.grid()
    precision.title("")  # (f"{title}")# for different number of features")
    precision.xlabel("Input Dimension")
    precision.ylabel(f"{title}")
    xlistp = xList
    ylistp = centPrecision
    ylistmulti = multiPrecision
    # precision.plot(xlistp,  ylistp, label="Centralized")
    # precision.plot(xlistp, ylistmulti, label="Multi Worker")
    precision.scatter(xlistp, ylistp, c="orange", label="Centralized")
    precision.scatter(xlistp, ylistmulti, c="cornflowerblue", label="Multi Worker")
    precision.legend(loc="lower right", framealpha=1.0, facecolor='white')
    precision.savefig(f"comparison_scatter_precision_e50_lr0001_bs128.png")
    print("multi precision")


# %%
def recall_plt_multi(plt):
    title = "Recall"
    recall = plt
    recall.style.use("default")
    recall.grid()
    recall.title("")  # (f"{title}")# for different number of features")
    recall.xlabel("Input Dimensions")
    recall.ylabel(f"{title}")
    half = len(xList) >> 1
    xlistr = xList[half:]
    ylistr = centRecall[half:]
    ylistmulti = multiRecall[half:]
    # recall.plot(xlistr, ylistr, label="Centralized")
    # recall.plot(xlistr, ylistmulti, label="Multi Worker")
    # this line will shift the plot away from the left side by 0.15
    recall.subplots_adjust(left=0.15)
    recall.scatter(xlistr, ylistr, c="orange", label="Centralized")
    recall.scatter(xlistr, ylistmulti, c="cornflowerblue", label="Multi Worker")
    recall.legend(loc="lower right", framealpha=1.0, facecolor='white')
    recall.savefig(f"comparison_scatter_recall_e50_lr0001_bs128.png")
    print("Multi recall")


# %%
def f1_plt_multi(plt):
    title = "F-Measure"
    f1 = plt
    f1.style.use("default")
    f1.grid()
    f1.title("")  # f"{title}")# per number of features")
    f1.xlabel("Input Dimnesions")
    f1.ylabel(f"{title}")
    half = len(xList) >> 1
    xlistf = xList[half:]
    ylistf = centF1[half:]
    ylistmulti = multiF1[half:]
    # f1.plot(xlistf, ylistf, label="Centralized")
    # f1.plot(xlistf, ylistmulti, label="Multi Worker")
    f1.scatter(xlistf, ylistf, c="orange", label="Centralized")
    f1.scatter(xlistf, ylistmulti, c="cornflowerblue", label="Multi Worker")
    f1.legend(loc="lower right", framealpha=1.0, facecolor='white')
    f1.savefig(f"comparison_scatter_f1_e50_lr0001_bs128.png")
    print("Multi F-measure")


# %%
'''def accuracy_plt_multi_worker_multi(plt):
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
    print("Multi accuracy")'''


# %%
def pandas_dataframe_print(plt):
    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    df = pd.DataFrame([multiAcc, centAcc])
    rowLabel = ("mutli worker Accuracy", "centralized Accuracy")
    ax.table(cellText=df.values, cellLoc='center', colLabels=xList, rowLabels=rowLabel, loc='center')
    fig.tight_layout()
    plt.savefig(f"data_acc_e50_lr001_bs128_table.pdf")


# %%
def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}.")
    plt.style.use("ggplot")

    matplotlib.use("pdf")
    accuracy_avg_bar(plt)
    # accuracy_plt_multi(plt)
    # f1_plt_multi(plt)
    # recall_plt_multi(plt)
    # precision_plt_multi(plt)
    # pandas_dataframe_print(plt)

    # f1_plt(plt)
    # precision_plt(plt)
    # recall_plt(plt)

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
