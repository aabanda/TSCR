import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import warnings
from metrics.kendall import kendalltau_partial_both
from metrics.hamming import hamming_score



def model_metatarget(X_train, X_test, y_train, y_test, metatarget,inference,m=None,w=None):

    if metatarget=="classifier_accuracies":

        clf = LinearRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred[y_pred > 1] = 1
        sml=metrics.mean_absolute_error(y_test, y_pred)
        y_pred_bs = np.mean(y_train, axis=0)
        y_pred_bs = np.tile(y_pred_bs, (y_test.shape[0], 1))
        bs=metrics.mean_absolute_error(y_test, y_pred_bs, multioutput='uniform_average')

        if inference:

            complete_ranking, topm_rankings = inference_metatarger(y_test, y_pred, "rankings", m)

            # Best set
            best_set = [[] for i in range(len(w))]
            for ws in range(0, len(w)):
                labels_test_set = margin_based_labels(y_test, w[ws])
                labels_pred_set = margin_based_labels(y_pred, w[ws])
                best_set[ws]=hamming_score(labels_test_set, labels_pred_set)

            # Best
            labels_test_best = np.argmax(y_test, axis=1)
            labels_pred_best = np.argmax(y_pred, axis=1)
            best = f1_score(labels_test_best, labels_pred_best, average='weighted')

            return sml, bs, complete_ranking, topm_rankings, best_set, best

        else:

            return sml, bs


    if metatarget == "complete_ranking":

        sml = []
        bs = []
        ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
        ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
        mean_ranking_train = pd.DataFrame(y_train).mean(axis=0).rank(ascending=False, method="min")

        topm = [[] for i in range(len(m))]
        labels_pred_best = []
        dis = euclidean_distances(X_test,X_train)

        for test_indx_nn in range(0, X_test.shape[0]):
            neigh = np.stack((ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[0], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[1], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[2], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[3], :].to_numpy(),
                              ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[4], :].to_numpy()))

            pred = np.argsort(np.argsort(neigh.sum(axis=0)))
            if np.max(pred) ==  X_train.shape[1]-1:
                pred = pred + 1
            sml.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], 0))
            bs.append(kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :], 0))

            if inference:
                for ms in range(0, len(m)):
                    topm[ms].append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :].to_numpy(), m[ms]))
                    topm[ms].append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :].to_numpy(), m[ms]))
                    topm[ms].append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :].to_numpy(), m[ms]))

                labels_pred_best.append(np.argmin(pred))

        if inference:
            # Best
            labels_test_best = np.argmax(y_test, axis=1)
            best = f1_score(labels_test_best, labels_pred_best, average='weighted')

            return np.mean(sml), np.mean(bs), np.mean(topm,axis=1), best

        else:

            return np.mean(sml), np.mean(bs)


    if metatarget == "topm_rankings":

        if np.size(m) > 1:
            sml = [[] for i in range(len(m))]
            bs = [[] for i in range(len(m))]
            labels_pred_best = []
            ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
            ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
            mean_ranking_train = pd.DataFrame(y_train).mean(axis=0).rank(ascending=False, method="min")

            dis = euclidean_distances(X_test, X_train)
            for test_indx_nn in range(0, ranking_test.shape[0]):
                neigh = np.stack((ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[0], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[1], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[2], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[3], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[4], :].to_numpy()))

                pred = np.argsort(np.argsort(neigh.sum(axis=0)))
                if np.max(pred) == X_train.shape[1]-1:
                    pred = pred + 1

                for ms in range(0, len(m)):
                    sml[ms].append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :].to_numpy(), m[ms]))
                    bs[ms].append(kendalltau_partial_both(mean_ranking_train, ranking_test.iloc[test_indx_nn, :].to_numpy(), m[ms]))

                if inference:
                    labels_pred_best.append(np.argmin(pred))

            if inference:
                # Best
                labels_test_best = np.argmax(y_test, axis=1)
                best = f1_score(labels_test_best, labels_pred_best, average='weighted')

                return np.mean(sml, axis=1), np.mean(bs, axis=1), best

            else:
                return np.mean(sml,axis=1), np.mean(bs,axis=1)

        elif np.size(m) == 1:

            sml = []
            labels_pred_best = []
            ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
            ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
            mean_ranking_train = pd.DataFrame(y_train).mean(axis=0).rank(ascending=False, method="min")

            dis = euclidean_distances(X_test, X_train)
            for test_indx_nn in range(0, ranking_test.shape[0]):
                neigh = np.stack((ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[0], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[1], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[2], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[3], :].to_numpy(),
                                  ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[4], :].to_numpy()))

                pred = np.argsort(np.argsort(neigh.sum(axis=0)))
                if np.max(pred) == X_train.shape[1] - 1:
                    pred = pred + 1
                sml.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :].to_numpy(), m))


            return np.mean(sml)




        else:
            print("algo mal")

    if metatarget == "best_set":

        if np.size(w) > 1:

            sml = np.zeros(len(w))
            bs = np.zeros(len(w))
            mean_performances = np.mean(y_train, axis=0)

            clf = KNeighborsClassifier(n_neighbors=5)

            for ws in range(0, len(w)):
                y_test_labels = margin_based_labels(y_test, w[ws])
                y_train_labels = margin_based_labels(y_train, w[ws])
                labels_bs = np.tile(margin_based_labels(mean_performances, w[ws]), (y_test.shape[0], 1))
                clf.fit(X_train, y_train_labels)
                pred = clf.predict(X_test)
                sml[ws]= hamming_score(y_test_labels, pred)
                bs[ws] = hamming_score(y_test_labels, labels_bs)

            return sml, bs

        elif np.size(w) == 1:
            sml = []
            mean_performances = np.mean(y_train, axis=0)

            clf = KNeighborsClassifier(n_neighbors=5)
            y_test_labels = margin_based_labels(y_test, w)
            y_train_labels = margin_based_labels(y_train, w)
            labels_bs = np.tile(margin_based_labels(mean_performances, w), (y_test.shape[0], 1))
            clf.fit(X_train, y_train_labels)
            pred = clf.predict(X_test)
            sml.append(hamming_score(y_test_labels, pred))

            return np.mean(sml)


    if metatarget == "best":

        labels_test = np.argmax(y_test, axis=1)
        labels_train = np.argmax(y_train, axis=1)
        labels_bs = np.repeat(np.argmax(np.mean(y_train, axis=0)), len(labels_test))

        clf = KNeighborsClassifier(n_neighbors=1)

        clf.fit(X_train, labels_train)
        pred = clf.predict(X_test)

        sml = f1_score(labels_test, pred, average='weighted')
        bs = f1_score(labels_test, labels_bs, average='weighted')

        return np.mean(sml),np.mean(bs)





def inference_metatarger(y_test,y_pred,metatarget,m):

     if metatarget == "rankings":

         ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
         ranking_pred = pd.DataFrame(y_pred).rank(axis=1, ascending=False, method="min")

         complete_ranking = []
         topm = [[] for i in range(len(m))]

         for test_indx_nn in range(0, y_test.shape[0]):
             complete_ranking.append(kendalltau_partial_both(ranking_pred.iloc[test_indx_nn, :], ranking_test.iloc[test_indx_nn, :], 0))

             for ms in range(0,len(m)):
                 topm[ms].append(kendalltau_partial_both(ranking_pred.iloc[test_indx_nn, :].to_numpy(),
                                                                ranking_test.iloc[test_indx_nn, :].to_numpy(), m[ms]))

         return np.mean(complete_ranking), np.mean(topm,axis=1)






def margin_based_labels(classifier_accuracies,w):

    if len(classifier_accuracies.shape)==1:
        labels = np.zeros(len(classifier_accuracies))
        max = np.max(classifier_accuracies)
        min = np.min(classifier_accuracies)
        min_margin = max - w * (max - min)
        labels[np.where(min_margin <= classifier_accuracies)[0]] = 1
    else:
        labels = np.zeros((classifier_accuracies.shape[0], classifier_accuracies.shape[1]))
        for db in range(0, classifier_accuracies.shape[0]):
            max = np.max(classifier_accuracies[db, :])
            min = np.min(classifier_accuracies[db, :])
            min_margin = max - w * (max - min)
            labels[db, np.where(min_margin <= classifier_accuracies[db, :])[0]] = 1
    return labels



def select_landmarkers_train(X_train,y_train,metatarget,m=None,w=None):

    if metatarget == "classifier_accuracies":

        best_starting_point = []
        best_subset = []
        delta_intern = []
        selected_intern = []

        kf2 = KFold(n_splits=3, shuffle=True)

        for starting_point in range(0, X_train.shape[1]):

            actual_landmarkers = starting_point
            best_error = [1]
            fold = 0
            delta_algo = np.zeros((X_train.shape[1], 3))

            for train_index2, test_index2 in kf2.split(X_train):

                X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                    algorithm_sum = np.append(actual_landmarkers, algorithm)

                    X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                    y_train3, y_test3 = y_train2, y_test2

                    delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget,False,m=m,w=w)[0]
                    delta_algo[actual_landmarkers, fold] = 1

                fold = fold + 1

            delta = np.mean(delta_algo, axis=1)

            while np.min(delta) < best_error[-1]:

                best_error.append(np.min(delta))
                best_to_add = np.argmin(delta)
                actual_landmarkers = np.append(actual_landmarkers, best_to_add)
                fold = 0
                delta_algo = np.zeros((X_train.shape[1], 3))

                for train_index2, test_index2 in kf2.split(X_train):

                    X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                    y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                    for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                        algorithm_sum = np.append(actual_landmarkers, algorithm)

                        X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                        y_train3, y_test3 = y_train2, y_test2

                        delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget,False, m=m,w=w)[0]
                        delta_algo[actual_landmarkers, fold] = 1

                    fold = fold + 1

                delta = np.mean(delta_algo, axis=1)

            delta_intern.append(best_error[-1])
            selected_intern.append(actual_landmarkers)

        best_starting_point.append(np.min(delta_intern))
        best_subset.append(selected_intern[np.argmin(delta_intern)])
        selection_train = best_subset[np.argmin(best_starting_point)]

    elif metatarget == "complete_ranking" or metatarget == "topm_rankings":

        best_starting_point = []
        best_subset = []
        delta_intern = []
        selected_intern = []

        kf2 = KFold(n_splits=3, shuffle=True)

        for starting_point in range(0, X_train.shape[1]):

            actual_landmarkers = starting_point
            best_error = [-1]
            fold = 0
            delta_algo = np.zeros((X_train.shape[1], 3))

            for train_index2, test_index2 in kf2.split(X_train):

                X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                    algorithm_sum = np.append(actual_landmarkers, algorithm)

                    X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                    y_train3, y_test3 = y_train2, y_test2

                    if metatarget == "complete_ranking":
                        delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget,False,m=m,w=w)[0]
                    else:
                        delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget, False, m=m, w=w)

                    delta_algo[actual_landmarkers, fold] = -1

                fold = fold + 1

            delta = np.mean(delta_algo, axis=1)

            while np.max(delta) > best_error[-1]:

                best_error.append(np.max(delta))
                best_to_add = np.argmax(delta)
                actual_landmarkers = np.append(actual_landmarkers, best_to_add)
                fold = 0
                delta_algo = np.zeros((X_train.shape[1], 3))

                for train_index2, test_index2 in kf2.split(X_train):

                    X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                    y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                    for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                        algorithm_sum = np.append(actual_landmarkers, algorithm)

                        X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                        y_train3, y_test3 = y_train2, y_test2

                        if metatarget == "complete_ranking":
                            delta_algo[algorithm, fold] = \
                            model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget, False, m=m, w=w)[0]
                        else:
                            delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3,
                                                                           metatarget, False, m=m, w=w)

                        delta_algo[actual_landmarkers, fold] = -1

                    fold = fold + 1

                delta = np.mean(delta_algo, axis=1)

            delta_intern.append(best_error[-1])
            selected_intern.append(actual_landmarkers)
        best_starting_point.append(np.max(delta_intern))
        best_subset.append(selected_intern[np.argmax(delta_intern)])
        selection_train = best_subset[np.argmax(best_starting_point)]

    elif metatarget == "best_set":

        best_starting_point = []
        best_subset = []
        delta_intern = []
        selected_intern = []

        kf2 = KFold(n_splits=3, shuffle=True)

        for starting_point in range(0, X_train.shape[1]):

            actual_landmarkers = starting_point
            best_error = [-1]
            fold = 0
            delta_algo = np.zeros((X_train.shape[1], 3))

            for train_index2, test_index2 in kf2.split(X_train):

                X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                    algorithm_sum = np.append(actual_landmarkers, algorithm)

                    X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                    y_train3, y_test3 = y_train2, y_test2

                    delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget,False,m=m,w=w)

                    delta_algo[actual_landmarkers, fold] = -1

                fold = fold + 1

            delta = np.mean(delta_algo, axis=1)

            while np.max(delta) > best_error[-1]:

                best_error.append(np.max(delta))
                best_to_add = np.argmax(delta)
                actual_landmarkers = np.append(actual_landmarkers, best_to_add)
                fold = 0
                delta_algo = np.zeros((X_train.shape[1], 3))

                for train_index2, test_index2 in kf2.split(X_train):

                    X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                    y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                    for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                        algorithm_sum = np.append(actual_landmarkers, algorithm)

                        X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                        y_train3, y_test3 = y_train2, y_test2


                        delta_algo[algorithm, fold] = model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget, False, m=m, w=w)
                        delta_algo[actual_landmarkers, fold] = -1

                    fold = fold + 1

                delta = np.mean(delta_algo, axis=1)

            delta_intern.append(best_error[-1])
            selected_intern.append(actual_landmarkers)
        best_starting_point.append(np.max(delta_intern))
        best_subset.append(selected_intern[np.argmax(delta_intern)])
        selection_train = best_subset[np.argmax(best_starting_point)]

    elif metatarget == "best":

        best_starting_point = []
        best_subset = []
        delta_intern = []
        selected_intern = []

        kf2 = KFold(n_splits=3, shuffle=True)

        for starting_point in range(0, X_train.shape[1]):

            actual_landmarkers = starting_point
            best_error = [-1]
            fold = 0
            delta_algo = np.zeros((X_train.shape[1], 3))

            for train_index2, test_index2 in kf2.split(X_train):

                X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                    algorithm_sum = np.append(actual_landmarkers, algorithm)

                    X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                    y_train3, y_test3 = y_train2, y_test2

                    delta_algo[algorithm, fold] = \
                    model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget, False, m=m, w=w)[0]
                    delta_algo[actual_landmarkers, fold] = -1

                fold = fold + 1

            delta = np.mean(delta_algo, axis=1)

            while np.max(delta) > best_error[-1]:

                best_error.append(np.max(delta))
                best_to_add = np.argmax(delta)
                actual_landmarkers = np.append(actual_landmarkers, best_to_add)
                fold = 0
                delta_algo = np.zeros((X_train.shape[1], 3))

                for train_index2, test_index2 in kf2.split(X_train):

                    X_train2, X_test2 = X_train[train_index2, :], X_train[test_index2, :]
                    y_train2, y_test2 = y_train[train_index2, :], y_train[test_index2, :]

                    for algorithm in list(np.setdiff1d(range(0, X_train.shape[1]), actual_landmarkers)):
                        algorithm_sum = np.append(actual_landmarkers, algorithm)

                        X_train3, X_test3 = X_train2[:, algorithm_sum], X_test2[:, algorithm_sum]
                        y_train3, y_test3 = y_train2, y_test2

                        delta_algo[algorithm, fold] = \
                        model_metatarget(X_train3, X_test3, y_train3, y_test3, metatarget, False, m=m, w=w)[0]
                        delta_algo[actual_landmarkers, fold] = -1

                    fold = fold + 1

                delta = np.mean(delta_algo, axis=1)

            delta_intern.append(best_error[-1])
            selected_intern.append(actual_landmarkers)
        best_starting_point.append(np.max(delta_intern))
        best_subset.append(selected_intern[np.argmax(delta_intern)])
        selection_train = best_subset[np.argmax(best_starting_point)]


    return selection_train





















def evaluate_metatarget(X, Y, metatarget, FLS, inference):
    warnings.filterwarnings("ignore")

    #Default values from m and w
    m = np.array([3,5,10])
    w = np.array([0.05,0.1,0.2])

    if FLS == False:

        if metatarget == "classifier_accuracies":
            eval_sml = []
            eval_bs = []
            eval_complete_ranking = []
            eval_topm = [[] for i in range(len(m))]
            eval_bestset = [[] for i in range(len(w))]
            eval_best = []
        elif metatarget == "complete_ranking":
            eval_sml = []
            eval_bs = []
            eval_topm = [[] for i in range(len(m))]
            eval_best = []
        elif metatarget == "topm_rankings":
            eval_sml = [[] for i in range(len(m))]
            eval_bs = [[] for i in range(len(m))]
            eval_best = []
        elif metatarget == "best_set":
            eval_sml = [[] for i in range(len(w))]
            eval_bs = [[] for i in range(len(w))]
            if inference:
                raise TypeError("This meta-target does not allow inference")
        elif metatarget == "best":
            eval_sml = []
            eval_bs = []
            if inference:
                raise TypeError("This meta-target does not allow inference")
        else:
            print("Error: unknown meta-target")
            print("Available meta-targets:")
            raise TypeError("Error: unknown meta-target \n Available meta-targets: \n classifier_accuracies, complete_ranking, topm_rankings, best_set, best")

        for r in range(0, 50):
            kf = KFold(n_splits=5, shuffle=True)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = Y[train_index, :], Y[test_index, :]

                if metatarget == "classifier_accuracies":
                    if inference:
                        sml, bs, complete_ranking, topm_rankings, best_set, best = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m, w=w)
                        eval_sml.append(sml)
                        eval_bs.append(bs)
                        eval_complete_ranking.append(complete_ranking)
                        for ms in range(0, len(m)):
                            eval_topm[ms].append(topm_rankings[ms])
                        for ws in range(0, len(w)):
                            eval_bestset[ws].append(best_set[ws])
                        eval_best.append(best)
                    else:
                        sml, bs = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m, w=w)
                        eval_sml.append(sml)
                        eval_bs.append(bs)
                elif metatarget == "complete_ranking":
                    if inference:
                        sml, bs, topm, best = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m, w=w)
                        eval_sml.append(sml)
                        eval_bs.append(bs)
                        for ms in range(0, len(m)):
                            eval_topm[ms].append(topm[ms])
                        eval_best.append(best)
                    else:
                        sml, bs = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m, w=w)
                        eval_sml.append(sml)
                        eval_bs.append(bs)
                elif metatarget == "topm_rankings":
                    if inference:
                        sml, bs , best = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m, w=w)
                        for ms in range(0, len(m)):
                            eval_sml[ms].append(sml[ms])
                            eval_bs[ms].append(bs[ms])
                        eval_best.append(best)
                    else:
                        sml, bs = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m, w=w)
                        for ms in range(0, len(m)):
                            eval_sml[ms].append(sml[ms])
                            eval_bs[ms].append(bs[ms])

                elif metatarget == "best_set":
                    sml, bs = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m,w=w)
                    for ws in range(0,len(w)):
                        eval_sml[ws].append(sml[ws])
                        eval_bs[ws].append(bs[ws])
                elif metatarget == "best":

                    sml, bs = model_metatarget(X_train, X_test, y_train, y_test, metatarget, inference, m=m,w=w)
                    eval_sml.append(sml)
                    eval_bs.append(bs)



        print("Evaluation without forward landmarker selection:")

        if metatarget == "classifier_accuracies":
            print("Specific meta-learner (SML) : %0.3f" % np.mean(eval_sml))
            print("Baseline (BS) : %0.3f" % np.mean(eval_bs))
            if inference:
                print("Inferred meta-targets")
                print("Complete ranking: %0.3f" % np.mean(eval_complete_ranking))
                print("Top-M rankings:")
                for ms in range(0, len(m)):
                    print("M = %d : : %0.3f" % (m[ms], np.mean(eval_topm[ms])))
                print("Best set:")
                for ws in range(0, len(w)):
                    print("w = %0.2f : : %0.3f" % (w[ws], np.mean(eval_bestset[ws])))
                print("Best: %0.3f" % np.mean(eval_best))
        elif metatarget == "complete_ranking":
            print("Specific meta-learner (SML) : %0.3f" % np.mean(eval_sml))
            print("Baseline (BS) : %0.3f" % np.mean(eval_bs))
            if inference:
                print("Inferred meta-targets")
                print("Top-M rankings:")
                for ms in range(0, len(m)):
                    print("M = %d : : %0.3f" % (m[ms], np.mean(eval_topm[ms])))
                print("Best: %0.3f" % np.mean(eval_best))
        elif metatarget == "topm_rankings":
            for ms in range(0, len(m)):
                print("M = %d" % m[ms])
                print("Specific meta-learner (SML): %0.3f" % np.mean(eval_sml[ms]))
                print("Baseline (BS): %0.3f" % np.mean(bs[ms]))
            if inference:
                print("Inferred meta-targets")
                print("Best: %0.3f" % np.mean(eval_best))
        elif metatarget == "best_set":
            for ws in range(0, len(w)):
                print("M = %d" % w[ws])
                print("Specific meta-learner (SML): %0.3f" % np.mean(eval_sml[ws]))
                print("Baseline (BS): %0.3f" % np.mean(bs[ws]))
        else:
            print("Specific meta-learner (SML) : %0.3f" % np.mean(eval_sml))
            print("Baseline (BS) : %0.3f" % np.mean(eval_bs))


    else:

        if metatarget == "classifier_accuracies":

            sele_cv = []
            sml_fls = []
            complete_ranking = []
            topm_rankings = []
            best_set = [[] for i in range(len(w))]
            best = []

            for repe_cv_extern in range(0, 10):

                kf = KFold(n_splits=5, shuffle=True)

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index, :], X[test_index, :]
                    y_train, y_test = Y[train_index, :], Y[test_index, :]

                    selection_train=select_landmarkers_train(X_train, y_train, metatarget,m=m,w=w)


                    if inference:
                        # Regression
                        clf = LinearRegression()
                        clf.fit(X_train[:, selection_train], y_train)
                        y_pred = clf.predict(X_test[:, selection_train])
                        y_pred[y_pred > 1] = 1
                        sml_fls.append(np.mean(metrics.mean_absolute_error(y_test, y_pred)))
                        sele_cv.append(selection_train)

                        # Ranking / Top K ranking
                        complete, topm = inference_metatarger(y_test, y_pred, "rankings", m)
                        complete_ranking.append(complete)
                        topm_rankings.append(topm)

                        # Best set
                        for ws in range(0,len(w)):
                            labels_test_set = margin_based_labels(y_test,  w[ws])
                            labels_pred_set = margin_based_labels(y_pred,  w[ws])
                            best_set[ws].append(hamming_score(labels_test_set, labels_pred_set))


                        # Best
                        labels_test_best = np.argmax(y_test,axis=1)
                        labels_pred_best = np.argmax(y_pred,axis=1)
                        best.append(f1_score(labels_test_best, labels_pred_best, average='weighted'))



            print("Evaluation with forward landmarker selection:")
            print("Classifier Accuracies:")
            print("Metric: MAE")
            print("Specific meta-learner with FLS (SML_FLS) : %0.3f" % np.mean(sml_fls))

            if inference:
                print("Inferred meta-targets")
                print("Complete ranking (Kendall): %0.3f" % np.mean(complete_ranking))
                print("Top-M ranking (Kendall):")
                for ms in range(0, len(m)):
                    print("M = %d: %0.3f " % (m[ms],np.mean(topm_rankings[ms])))
                print("Best set:")
                for ws in range(0, len(w)):
                    print("w = %0.2f: %0.3f " % (w[ws], np.mean(best_set[ws])))
                print("Best (weighted F1): %0.3f" % np.mean(best))



        elif metatarget == "complete_ranking":

            sele_cv = []
            sml_fls = []
            topm_rankings = [[] for i in range(len(m))]
            best = []

            for repe_cv_extern in range(0, 10):

                kf = KFold(n_splits=5, shuffle=True)

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index, :], X[test_index, :]
                    y_train, y_test = Y[train_index, :], Y[test_index, :]

                    selection_train = select_landmarkers_train(X_train, y_train, metatarget, m=m, w=w)
                    print(selection_train)
                    sele_cv.append(selection_train)

                    ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
                    ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
                    labels_test_best = []
                    labels_pred_best = []

                    dis = euclidean_distances(X_test[:, selection_train], X_train[:, selection_train])
                    for test_indx_nn in range(0, y_test.shape[0]):
                        neigh = np.stack((ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[0], :].to_numpy(),
                                          ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[1], :].to_numpy(),
                                          ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[2], :].to_numpy(),
                                          ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[3], :].to_numpy(),
                                          ranking_train.iloc[np.argsort(dis[test_indx_nn,:])[4], :].to_numpy()))

                        pred = np.argsort(np.argsort(neigh.sum(axis=0)))
                        if np.max(pred) == X_train.shape[1]-1:
                            pred = pred + 1

                        sml_fls.append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], 0))

                        if inference:

                            for ms in range(0, len(m)):
                                topm_rankings[ms].append(kendalltau_partial_both(pred,ranking_test.iloc[test_indx_nn, :].to_numpy(),
                                                                        m[ms]))

                            # Best
                            labels_test_best.append(np.argmin(ranking_test.iloc[test_indx_nn, :].to_numpy()))
                            labels_pred_best.append(np.argmin(pred))

                    if inference:
                        best.append(f1_score(labels_test_best, labels_pred_best, average='weighted'))

            print("Evaluation with forward landmarker selection:")
            print("Complete ranking:")
            print("Metric: Kendall")
            print("Specific meta-learner with FLS (SML_FLS) : %0.3f" % np.mean(sml_fls))

            if inference:
                print("Inferred meta-targets")
                print("Top-M ranking (Kendall):")
                for ms in range(0, len(m)):
                    print("M = %d: %0.3f " % (m[ms], np.mean(topm_rankings[ms])))
                print("Best (weighted F1): %0.3f" % np.mean(best))


        elif metatarget == "topm_rankings":

            sele_cv = []
            sml_fls = [[] for i in range(len(m))]
            best = [[] for i in range(len(m))]

            for ms in range(0,len(m)):

                for repe_cv_extern in range(0, 10):

                    kf = KFold(n_splits=5, shuffle=True)

                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index, :], X[test_index, :]
                        y_train, y_test = Y[train_index, :], Y[test_index, :]

                        selection_train = select_landmarkers_train(X_train, y_train, metatarget, m=m[ms], w=w)
                        print(selection_train)
                        sele_cv.append(selection_train)

                        ranking_test = pd.DataFrame(y_test).rank(axis=1, ascending=False, method="min")
                        ranking_train = pd.DataFrame(y_train).rank(axis=1, ascending=False, method="min")
                        labels_test_best = []
                        labels_pred_best = []

                        dis = euclidean_distances(X_test[:, selection_train], X_train[:, selection_train])
                        for test_indx_nn in range(0, y_test.shape[0]):
                            neigh = np.stack((ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[0], :].to_numpy(),
                                              ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[1], :].to_numpy(),
                                              ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[2], :].to_numpy(),
                                              ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[3], :].to_numpy(),
                                              ranking_train.iloc[np.argsort(dis[test_indx_nn, :])[4], :].to_numpy()))

                            pred = np.argsort(np.argsort(neigh.sum(axis=0)))
                            if np.max(pred) == X_train.shape[1]-1:
                                pred = pred + 1

                            sml_fls[ms].append(kendalltau_partial_both(pred, ranking_test.iloc[test_indx_nn, :], m[ms]))

                            if inference:
                                # Best
                                labels_test_best.append(np.argmin(ranking_test.iloc[test_indx_nn, :].to_numpy()))
                                labels_pred_best.append(np.argmin(pred))

                        if inference:
                            best[ms].append(f1_score(labels_test_best, labels_pred_best, average='weighted'))

                print("Evaluation with forward landmarker selection:")
                print("Top-M rankings:")
                print("Metric: Kendall")
                print("M = %d : " % m[ms])
                print("Specific meta-learner with FLS (SML_FLS) : %0.3f" % np.mean(sml_fls[ms]))

                if inference:
                    print("Inferred meta-targets")
                    print("Best (weighted F1): %0.3f" % np.mean(best[ms]))


        elif metatarget == "best_set":

            sele_cv = []
            sml_fls = [[] for i in range(len(w))]

            for ws in range(0, len(w)):

                for repe_cv_extern in range(0, 10):

                    kf = KFold(n_splits=5, shuffle=True)

                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index, :], X[test_index, :]
                        y_train, y_test = Y[train_index, :], Y[test_index, :]

                        selection_train = select_landmarkers_train(X_train, y_train, metatarget, m=m, w=w[ws])
                        print("selection_train")
                        print(selection_train)
                        sele_cv.append(selection_train)

                        y_test_labels = margin_based_labels(y_test, w[ws])
                        y_train_labels = margin_based_labels(y_train, w[ws])

                        clf = KNeighborsClassifier(n_neighbors=5)

                        clf.fit(X_train[:,selection_train], y_train_labels)
                        pred = clf.predict(X_test[:,selection_train])
                        sml_fls[ws].append(hamming_score(y_test_labels, pred))



                print("Evaluation with forward landmarker selection:")
                print("Best set:")
                print("W = %d : " % w[ws])
                print("Specific meta-learner with FLS (SML_FLS) : %0.3f" % np.mean(sml_fls[ws]))



        elif metatarget == "best":

            sele_cv = []
            sml_fls = []


            for repe_cv_extern in range(0, 10):

                kf = KFold(n_splits=5, shuffle=True)

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index, :], X[test_index, :]
                    y_train, y_test = Y[train_index, :], Y[test_index, :]

                    selection_train = select_landmarkers_train(X_train, y_train, metatarget, m=m, w=w)
                    sele_cv.append(selection_train)

                    labels_test = np.argmax(y_test, axis=1)
                    labels_train = np.argmax(y_train, axis=1)

                    clf = KNeighborsClassifier(n_neighbors=1)

                    clf.fit(X_train[:,selection_train], labels_train)
                    pred = clf.predict(X_test[:,selection_train])

                    sml_fls.append(f1_score(labels_test, pred, average='weighted'))

                    print(sml_fls[-1])

            print("Evaluation with forward landmarker selection:")
            print("Best set:")
            print("Specific meta-learner with FLS (SML_FLS) : %0.3f" % np.mean(sml_fls))