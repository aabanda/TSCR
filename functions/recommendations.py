import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from functions.metalearners import margin_based_labels

def TSCR(features, metatarget):
    features = np.array(features)
    classifier_list = np.genfromtxt("data/classifier_list.txt",dtype="str")
    classifier_list = np.append(classifier_list, ["Resnet", "InceptionTime"])

    # Read landmarkers of UCR  (accuracies and times)
    landmarkers_UCR = np.loadtxt("data/landmarkers_UCR.txt")

    # Read results from UCR
    results_UCR = np.loadtxt("data/results_UCR.txt")

    if metatarget=="classifier_accuracies":

        clf = LinearRegression()
        clf.fit(landmarkers_UCR, results_UCR)
        pred = clf.predict(features.reshape(1, -1))
        pred[pred>1]=1
        pred = pd.DataFrame(np.column_stack((classifier_list, pred[0])))
        pred.values[:, 1] = np.round(pred.values[:, 1].astype(float),decimals=3)
        pred.columns = ["Classifier", "Accuracy"]
        pred = pred.sort_values(by="Accuracy",ascending=False)
        print(pred)
        return

    elif metatarget=="complete_ranking" :

        ranking_train = pd.DataFrame(results_UCR).rank(axis=1, ascending=False, method="min")

        dis = euclidean_distances(features.reshape(1, -1), landmarkers_UCR)
        neigh = np.stack((ranking_train.iloc[np.argsort(dis[0,:])[0], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[0,:])[1], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[0,:])[2], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[0,:])[3], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[0,:])[4], :].to_numpy()))

        pred = np.argsort(np.argsort(neigh.sum(axis=0)))
        pred = pd.DataFrame(np.column_stack((classifier_list, pred)))
        pred.values[:,1] = pred.values[:,1].astype(int)
        pred.columns = ["Classifier","Ranking"]
        pred = pred.sort_values(by="Ranking")
        print(pred)
        return

    elif metatarget == "topm_rankings":

        clf = LinearRegression()
        clf.fit(landmarkers_UCR, results_UCR)
        pred = clf.predict(features.reshape(1, -1))
        pred[pred > 1] = 1
        ranking = np.array(pd.DataFrame(pred).rank(axis=1, ascending=False, method="min"))
        ranking = np.argsort(ranking)
        r = pd.DataFrame(np.stack((list(range(1,4)), classifier_list[ranking[0][0:3]])))
        r.index = ['Position in ranking', 'Classifier']
        print("Top-3:")
        print(r)

        r = pd.DataFrame(np.stack(( list(range(1,6)),classifier_list[ranking[0][0:5]])))
        r.index = ['Position in ranking', 'Classifier']
        print("Top-5:")
        print(r)

        ranking_train = pd.DataFrame(results_UCR).rank(axis=1, ascending=False, method="min")

        dis = euclidean_distances(features.reshape(1, -1), landmarkers_UCR)
        neigh = np.stack((ranking_train.iloc[np.argsort(dis[0, :])[0], :].to_numpy(),
                          ranking_train.iloc[np.argsort(dis[0, :])[1], :].to_numpy(),
                          ranking_train.iloc[np.argsort(dis[0, :])[2], :].to_numpy(),
                          ranking_train.iloc[np.argsort(dis[0, :])[3], :].to_numpy(),
                          ranking_train.iloc[np.argsort(dis[0, :])[4], :].to_numpy()))

        pred = np.argsort(np.argsort(neigh.sum(axis=0)))


        r = pd.DataFrame(np.stack((list(range(1, 11)), classifier_list[np.argsort(pred)][0:10])))
        r.index = ['Position in ranking', 'Classifier']
        print("Top-10:")
        print(r)
        return

    elif metatarget == "best_set":


        w = np.array([0.05, 0.1, 0.2])

        clf = LinearRegression()
        clf.fit(landmarkers_UCR, results_UCR)
        pred = clf.predict(features.reshape(1, -1))
        pred[pred > 1] = 1
        margin = margin_based_labels(pred, 0.05)
        print("w = 0.05")
        print(classifier_list[np.where(margin[0])])

        clf = KNeighborsClassifier(n_neighbors=5)

        for ws in range(1, len(w)):

            y_train_labels = margin_based_labels(results_UCR, w[ws])
            clf.fit(landmarkers_UCR, y_train_labels)
            pred = clf.predict(features.reshape(1, -1))
            print("w = %0.2f" % w[ws])
            print(classifier_list[np.where(pred[0])])

        return


    elif metatarget == "best":

        labels_train = np.argmax(results_UCR, axis=1)
        clf = KNeighborsClassifier(n_neighbors=1)

        clf.fit(landmarkers_UCR, labels_train)
        pred = clf.predict(features.reshape(1, -1))
        print(classifier_list[pred])
        return

    else:
        print("Invalid meta-target. Valid meta-targets are: classifier_accuracies, complete_ranking, topm_rankings, best_set and best. ")
