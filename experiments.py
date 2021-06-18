import numpy as np
from functions.landmarkers import read_landmarkes, read_results_UCR
from functions.subsamples import create_subsample
from functions.plots import plot_cor_by_dataset, plot_cor_by_landmarker_and_time, plot_ranking_separate, plot_topm_rankings_separate
from functions.metalearners import evaluate_metatarget
from pymfe.mfe import MFE


classifier_list = np.genfromtxt("data/classifier_list.txt",dtype="str")
UCR_list = np.genfromtxt('data/db_names.txt',dtype='str')

#Read results from UCR
results_UCR = np.loadtxt("data/results_UCR.txt")


#Two options for the landmarkers: 
#a) compute the 24 landmarkers for the 112 datasets from the UCR repository:

        # Create subsampled datasets of the datasets in UCR_list (orignal datasets need to be in .ts format, x_TRAIN.ts, x_TEST.ts):
        # create_subsample("data/Univariate_ts", UCR_list, "data/subsamples_UCR/")


        # Compute landmarkers:
        # for db_name in UCR_list:
        #   compute_landmarkers_TSCR(db_name,  "data/subsamples_UCR/",  "data/landmarkers_UCR/")

        # Load landmarkers:
        # landmarkers_UCR = np.zeros((len(UCR_list), len(classifier_list)))
        # landmarkers_time = np.zeros((len(UCR_list), len(classifier_list)))
        #
        # for i, db_name in UCR_list:
        #   land, time = read_landmarkes_TSCR(db_name, landmarker_output_dir)
        #   landmarkers_UCR[i,:]= land
        #   landmarkers_time[i,:]= time




#b) load the precomputed 24 landmarkers for the 112 datasets from the UCR repository:

landmarkers_UCR = np.loadtxt("data/landmarkers_UCR.txt")
landmarkers_time = np.loadtxt("data/landmarkers_time.txt")






#Two options for the standard metafeatures: 
#a) compute the 73 standard metafeatures for the 112 datasets from the UCR repository:

        # mfe = MFE(groups=["general", "statistical", "info-theory"])

        # metafeatures = np.zeros((len(UCR_list),73))
        # metafeatures_time = np.zeros((len(UCR_list),73))

        # for db_name in UCR_list:
        #     train_x, train_y = load_from_tsfile_to_dataframe(
        #         "%s/%s/%s_TRAIN.ts" % ("data/Univariate_ts", db_name, db_name))
        #     test_x, test_y = load_from_tsfile_to_dataframe(
        #         "%s/%s/%s_TEST.ts" % ("data/Univariate_ts", db_name, db_name))
        #     data = np.zeros((len(train_y) + len(test_y), len(train_x.iloc[1, 0])))

        #     for i in range(0, len(train_y)):
        #         data[i, :] = train_x.iloc[i, :][0].values

        #     k = 0
        #     for i in range(len(train_y), len(train_y) + len(test_y)):
        #         data[i, :] = test_x.iloc[k, :][0].values
        #         k = k + 1

        #     classes = np.concatenate((train_y, test_y))
        #     classes = classes.astype(int)

        #     mfe.fit(data, classes)
        #     ft = mfe.extract()
        #     np.savetxt("data/mf/mf_"+db_name, ft[1])






#b) load the precomputed metafeatures for the 112 datasets from the UCR repository:

metafeatures = np.zeros((len(UCR_list),73))
for i, db in db_names:
    metafeatures[i,:]=np.loadtxt("data/mf/mf_"+db+'.txt')
metafeatures = np.nan_to_num(metafeatures, nan=0)




# EVALUATION OF TSCR FOR EACH META-TARGET


evaluate_metatarget(landmarkers_UCR, results_UCR, "classifier_accuracies", FLS=False, inference=True)
evaluate_metatarget(metafeatures, results_UCR, "classifier_accuracies", FLS=False, inference=True)

evaluate_metatarget(landmarkers_UCR, results_UCR, "complete_ranking", FLS=False, inference=True)
evaluate_metatarget(metafeatures, results_UCR, "complete_ranking", FLS=False, inference=True)

evaluate_metatarget(landmarkers_UCR, results_UCR, "topm_rankings", FLS=True, inference=True)
evaluate_metatarget(metafeatures, results_UCR, "topm_rankings", FLS=True, inference=True)

evaluate_metatarget(landmarkers_UCR, results_UCR, "best_set", FLS=True, inference=False)
evaluate_metatarget(metafeatures, results_UCR, "best_set", FLS=True, inference=False)

evaluate_metatarget(landmarkers_UCR, results_UCR, "best", FLS=False, inference=False)
evaluate_metatarget(metafeatures, results_UCR, "best", FLS=False, inference=False)


#Reproduce figure 7
plot_ranking_separate(landmarkers_UCR,results_UCR)
#Reproduce figure 7 for topm rankings
plot_topm_rankings_separate(landmarkers_UCR,results_UCR)





#Reproduce figure 5 of paper
plot_cor_by_landmarker_and_time(landmarkers_UCR,landmarkers_time,results_UCR)


#Reproduce figure 6 of paper
plot_cor_by_dataset(landmarkers_UCR,results_UCR)



