from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch


def line(y, x=None):
    y = y.detach().cpu().numpy()

    if x is not None:
        x = x.detach().cpu().numpy()
        plt.plot(x, y)
    else:
        plt.plot(y)
    plt.show()


def lines(ys, xs, labels, xlabel=None):
    fig = plt.figure()

    N = len(xs)
    for i in range(N):
        if isinstance(xs[i], torch.Tensor):
            xs[i] = xs[i].detach().cpu().numpy()
        if isinstance(ys[i], torch.Tensor):
            ys[i] = ys[i].detach().cpu().numpy()
        plt.plot(xs[i], ys[i], label=labels[i])

    plt.legend(loc="lower right")

    if xlabel is not None:
        plt.xlabel(xlabel)
    plt.show()


def metrics(metrics):

    legend = []

    map_key_2_label = {
        "se3_disp1_outlier_perc": "se3_disp1",
        "nn_disp1_outlier_perc": "nn_disp1",
        "se3_disp2_outlier_perc": "se3_disp2",
        "se3_oflow_outlier_perc": "se3_oflow",
        "nn_oflow_outlier_perc": "nn_oflow",
        "se3_sflow_outlier_perc": "se3_sflow",
        "se3_f_measure_outlier_perc": "se3_f_measure",
        "se3_objects_count": "se3_objects_count",
        # "se3_duration" : "se3_duration",
    }

    color_mapping = {
        "se3_disp1_outlier_perc": "green",
        "nn_disp1_outlier_perc": "lime",
        "se3_disp2_outlier_perc": "cyan",
        "se3_oflow_outlier_perc": "orange",
        "nn_oflow_outlier_perc": "wheat",
        "se3_sflow_outlier_perc": "purple",
        "se3_f_measure_outlier_perc": "lightcoral",
        "se3_objects_count": "grey",
        # "se3_duration" : "se3_duration",
    }
    fname = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    for key, val in metrics.items():
        if key in map_key_2_label.keys():
            val = val.detach().cpu().numpy()
            plt.plot(val, "--^", color=color_mapping[key])
            legend.append(map_key_2_label[key])
    plt.legend(legend, loc="upper left")

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(20, 5)
    # plt.show()
    plt.savefig("results/metrics/" + fname + ".svg")
