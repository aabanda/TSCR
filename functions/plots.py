import numpy as np, pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances
from metrics.kendall import kendalltau_partial_both

def plot_cor_by_dataset(accu, results):

    classifier_list = np.genfromtxt("../data/classifier_list.txt",dtype="str") 
    UCR_list = np.genfromtxt('../data/db_names.txt',dtype='str')
    classifier_list[classifier_list=="ShapeletTransformClassifier"]="ST"
    classifier_list[classifier_list=="FastShapelets"]="FT"

    fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(18, 20), constrained_layout=True)
    k = 0
    
    for ax in axs.flat:
        l1 = ax.plot(landmarkers[k, :], color="darkorange", label="Landmarkers")
        l2 = ax.plot(results_UCR[k, :], label="Original algorithms")
        ax.set_title(db_names[k], fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_xticks(np.arange(len(classifier_list)))
        ax.set_xticklabels(classifier_list, rotation=90, ha='center', fontsize=14)
        ax.set_ylim(0,1.05)
        ax.tick_params(axis='both', which='major', labelsize=12)
        k = k + 1

    lgd=fig.legend([l1, l2], labels=["Landmarkers", "Classifiers"], bbox_to_anchor=(0.58, 1.05), fontsize=18)
    fig.tight_layout()
    plt.tight_layout()

    fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(18, 20), constrained_layout=True)
    k = 28
   
    for ax in axs.flat:
        l1 = ax.plot(landmarkers[k, :], color="darkorange", label="Landmarkers")
        l2 = ax.plot(results_UCR[k, :], label="Original algorithms")
        ax.set_title(db_names[k], fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_xticks(np.arange(len(classifier_list)))
        ax.set_xticklabels(classifier_list, rotation=90, ha='center', fontsize=14)
        ax.set_ylim(0,1.05)
        ax.tick_params(axis='both', which='major', labelsize=12)
        k = k + 1

    lgd=fig.legend([l1, l2], labels=["Landmarkers", "Classifiers"], bbox_to_anchor=(0.58, 1.05), fontsize=18)
    fig.tight_layout()
    plt.tight_layout()

    fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(18, 20), constrained_layout=True)
    k = 56

    for ax in axs.flat:
        l1 = ax.plot(landmarkers[k, :], color="darkorange", label="Landmarkers")
        l2 = ax.plot(results_UCR[k, :], label="Original algorithms")
        ax.set_title(db_names[k], fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_xticks(np.arange(len(classifier_list)))
        ax.set_xticklabels(classifier_list, rotation=90, ha='center', fontsize=14)
        ax.set_ylim(0,1.05)
        ax.tick_params(axis='both', which='major', labelsize=12)
        k = k + 1

    lgd=fig.legend([l1, l2], labels=["Landmarkers", "Classifiers"], bbox_to_anchor=(0.58, 1.05), fontsize=18)
    fig.tight_layout()
    plt.tight_layout()

    fig, axs = plt.subplots(nrows=7, ncols=4, figsize=(18, 20), constrained_layout=True)
    k = 84

    for ax in axs.flat:
        l1 = ax.plot(landmarkers[k, :], color="darkorange", label="Landmarkers")
        l2 = ax.plot(results_UCR[k, :], label="Original algorithms")
        ax.set_title(db_names[k], fontsize=18)
        ax.set_ylabel("Accuracy", fontsize=18)
        ax.set_xticks(np.arange(len(classifier_list)))
        ax.set_xticklabels(classifier_list, rotation=90, ha='center', fontsize=14)
        ax.set_ylim(0,1.05)
        ax.tick_params(axis='both', which='major', labelsize=12)
        k = k + 1

    lgd=fig.legend([l1, l2], labels=["Landmarkers", "Classifiers"], bbox_to_anchor=(0.58, 1.05), fontsize=18)
    fig.tight_layout()
    plt.tight_layout()






def plot_cor_by_landmarker_and_time(accu, time, results):

    correlations = []
    font = 20
    for a in range(0,results.shape[1]):
        correlations.append(np.corrcoef(accu[:,a],results[:,a])[0,1])

    plt.subplot(1,2,1)
    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.hist(np.sum(np.around(time,4),axis=1), bins=85,color="darkblue")
    plt.xlabel('Time (minutes)',fontsize=font)
    plt.ylabel('Count',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.tick_params(axis="y", direction="in", pad=10)
    plt.tick_params(axis="x", direction="in", pad=10)
    plt.title("Time spent computing \n all the landmarkers",fontsize=font)

    plt.subplot(1,2,2)
    plt.figure(figsize=(6, 5), constrained_layout=True)
    plt.hist(correlations,bins=25)
    plt.xlabel('Correlation',fontsize=font)
    plt.ylabel('Count',fontsize=font)
    plt.xticks(np.arange(np.round(min(correlations),decimals=1),np.round(max(correlations),decimals=1), 0.1),fontsize=font)
    plt.yticks(fontsize=font)
    plt.tick_params(axis="y", direction="in", pad=10)
    plt.tick_params(axis="x", direction="in", pad=10)
    plt.title("Accuracy correlations between \n landmarkers and original algorithms",fontsize=font)

    plt.show()




def plot_ranking_separate(accu, results):

    X = accu
    Y = results


    error_far = []
    error_far_base = []
    error_mid = []
    error_mid_base = []
    error_mid2 = []
    error_mid2_base = []
    error_close = []
    error_close_base = []

    for r in range(0, 10):
        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = Y[train_index, :], Y[test_index, :]

            ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
            ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
            mean_ranking_train = pd.DataFrame(y_train).mean(axis=0).rank(ascending=False, method="min")

            dis_train_mean = []
            for d in range(0, len(train_index)):
                dis_train_mean.append(kendalltau_partial_both(mean_ranking_train, ranking_train.iloc[d, :], 0))

            p25 = np.percentile(dis_train_mean, 25)
            p50 = np.percentile(dis_train_mean, 50)
            p75 = np.percentile(dis_train_mean, 75)

            for test_indx_nn in range(0, ranking_test.shape[0]):
                dis = euclidean_distances(X_test[test_indx_nn, :].reshape(1, -1),
                                          X_train)
                neigh = np.stack((ranking_train.iloc[np.argsort(dis)[0][0], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis)[0][1], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis)[0][2], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis)[0][3], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis)[0][4], :].to_numpy()))

                pred = np.argsort(np.argsort(neigh.sum(axis=0)))

                if np.max(pred) ==  X_train.shape[1]-1:
                    pred = pred + 1

                dis_test = kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0)

                if dis_test < p25:
                    error_far.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], 0))
                    error_far_base.append(
                        kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0))
                elif dis_test < p50:
                    error_mid.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], 0))
                    error_mid_base.append(
                        kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0))
                elif dis_test < p75:
                    error_mid2.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], 0))
                    error_mid2_base.append(
                        kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0))
                else:
                    error_close.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], 0))
                    error_close_base.append(
                        kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0))


    met = [np.mean(error_far), np.mean(error_mid), np.mean(error_mid2), np.mean(error_close)]
    met_sd = [np.std(error_far), np.std(error_mid), np.std(error_mid2), np.std(error_close)]
    bas = [np.mean(error_far_base), np.mean(error_mid_base), np.mean(error_mid2_base), np.mean(error_close_base)]
    bas_sd = [np.std(error_far_base), np.std(error_mid_base), np.std(error_mid2_base), np.std(error_close_base)]

    plt.plot(met, label="SM")
    plt.fill_between(range(4), np.asarray(met) - np.asarray(met_sd), np.asarray(met) + np.asarray(met_sd), alpha=.1)
    plt.plot(bas, label="Baseline")
    plt.fill_between(range(4), np.asarray(bas) - np.asarray(bas_sd), np.asarray(bas) + np.asarray(bas_sd), alpha=.1)

    total = len(error_far) + len(error_mid) + len(error_mid2) + len(error_close)
    freq1 = int(np.round(len(error_far) * 100 / total, 0))
    freq2 = int(np.round(len(error_mid) * 100 / total, 0))
    freq3 = int(np.round(len(error_mid2) * 100 / total, 0))
    freq4 = int(np.round(len(error_close) * 100 / total, 0))

    labx = ["S$_{1}$ \n (%d%%)" % freq1, "S$_{2}$ \n(%d%%)" % freq2,
            "S$_{3}$ \n(%d%%)" % freq3, "S$_{4}$ \n(%d%%)" % freq4]
    plt.xticks(np.arange(len(labx)), labx, fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel("Set of test instances \n"
               " (percentage of instances in each set)", fontsize=20, labelpad=20)
    plt.ylabel("Performance", fontsize=20, labelpad=20)
    plt.legend(loc="lower right", fontsize=20)

    plt.tight_layout()


def plot_topm_rankings_separate(accu,results):

    X = accu
    Y = results

    error_far = []
    error_far_base = []
    error_mid = []
    error_mid_base = []
    error_mid2 = []
    error_mid2_base = []
    error_close = []
    error_close_base = []
    j = 0

    for k in [3, 5, 10]:

        for r in range(0, 1):
            kf = KFold(n_splits=5, shuffle=True)

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = Y[train_index, :], Y[test_index, :]

                ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
                ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
                mean_ranking_train = pd.DataFrame(y_train).mean(axis=0).rank(ascending=False, method="min")

                dis_train_mean = []
                for d in range(0, len(train_index)):
                    dis_train_mean.append(kendalltau_partial_both(mean_ranking_train, ranking_train.iloc[d, :], 0))

                p25 = np.percentile(dis_train_mean, 25)
                p50 = np.percentile(dis_train_mean, 50)
                p75 = np.percentile(dis_train_mean, 75)

                for test_indx_nn in range(0, ranking_test.shape[0]):
                    dis = euclidean_distances(X_test[test_indx_nn, :].reshape(1, -1),
                                              X_train)
                    neigh = np.stack((ranking_train.iloc[np.argsort(dis)[0][0], :].to_numpy(),
                                      ranking_train.iloc[np.argsort(dis)[0][1], :].to_numpy(),
                                      ranking_train.iloc[np.argsort(dis)[0][2], :].to_numpy(),
                                      ranking_train.iloc[np.argsort(dis)[0][3], :].to_numpy(),
                                      ranking_train.iloc[np.argsort(dis)[0][4], :].to_numpy()))
                    pred = np.argsort(np.argsort(neigh.sum(axis=0)))

                    if np.max(pred) ==  X_train.shape[1]-1:
                        pred = pred + 1

                    dis_test = kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0)

                    if dis_test < p25:
                        error_far.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], k))
                        error_far_base.append(
                            kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], k))
                    elif dis_test < p50:
                        error_mid.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], k))
                        error_mid_base.append(
                            kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], k))
                    elif dis_test < p75:
                        error_mid2.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], k))
                        error_mid2_base.append(
                            kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], k))
                    else:
                        error_close.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], k))
                        error_close_base.append(
                            kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], k))


        met = [np.mean(error_far), np.mean(error_mid), np.mean(error_mid2), np.mean(error_close)]
        met_sd = [np.std(error_far), np.std(error_mid), np.std(error_mid2), np.std(error_close)]
        bas = [np.mean(error_far_base), np.mean(error_mid_base), np.mean(error_mid2_base), np.mean(error_close_base)]
        bas_sd = [np.std(error_far_base), np.std(error_mid_base), np.std(error_mid2_base), np.std(error_close_base)]

        plt.figure(j)
        plt.plot(met)
        plt.fill_between(range(4), np.asarray(met) - np.asarray(met_sd), np.asarray(met) + np.asarray(met_sd), alpha=.1)
        plt.plot(bas)
        plt.fill_between(range(4), np.asarray(bas) - np.asarray(bas_sd), np.asarray(bas) + np.asarray(bas_sd), alpha=.1)

        total = len(error_far) + len(error_mid) + len(error_mid2) + len(error_close)
        freq1 = np.round(len(error_far) / total, 2)
        freq2 = np.round(len(error_mid) / total, 2)
        freq3 = np.round(len(error_mid2) / total, 2)
        freq4 = np.round(len(error_close) / total, 2)

        labx = ["P1 (%0.2f%%)" % freq1, "P2 (%0.2f%%)" % freq2, "P3 (%0.2f%%)" % freq3, "P4 (%0.2f%%)" % freq4]
        plt.title("K = %d " % k)
        plt.xticks(np.arange(len(labx)), labx)
        j = j+1
