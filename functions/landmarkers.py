import numpy as np
import pandas as pd
import time
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold
from sktime_dl.deeplearning import ResNetClassifier
from sktime_dl.deeplearning import InceptionTimeClassifier
from subprocess import PIPE, Popen, STDOUT

def create_landmarkers_deep(db_name, subsample_output_dir,landmarker_output_dir):

    data = arff.loadarff("%s/%s/%s.arff" % (subsample_output_dir, db_name, db_name))
    df = pd.DataFrame(data[0])
    df['target']= df['target'].str.decode("utf-8")

   
    data = df.values[:,:-1]
    data = np.asarray(data).astype(np.float32)
    classes = df.values[:,-1]
    skf = StratifiedKFold(n_splits=2)
    skf.get_n_splits(data, classes)


    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(data, classes):
         #print("TRAIN:", train_index, "TEST:", test_index)
        print("splitting")

    train_x, test_x = data[train_index,:], data[test_index,:]
    train_y, test_y = classes[train_index], classes[test_index]

    X = pd.DataFrame()
    X["dim_0"] = [pd.Series(train_x[x, :]) for x in range(len(train_x))]
    train_x = X

    X = pd.DataFrame()
    X["dim_0"] = [pd.Series(test_x[x, :]) for x in range(len(test_x))]
    test_x = X

    start_time = time.time()
    network = ResNetClassifier(nb_epochs=200)

    network.fit(train_x, train_y)

    accu = network.score(test_x, test_y)
    print("--- %s seconds ---" % (time.time() - start_time))
    t = time.time() - start_time


    np.savetxt("%s/%s" % (landmarker_output_dir,db_name)+"_Resnet.txt", np.array([accu,t/60 ]))


    

    start_time = time.time()
    network = InceptionTimeClassifier(nb_epochs=200)

    network.fit(train_x, train_y)

    accu = network.score(test_x, test_y)
    print("--- %s seconds ---" % (time.time() - start_time))
    t = time.time() - start_time


    np.savetxt("%s/%s" % (landmarker_output_dir,db_name)+"_Inception.txt", np.array([accu,t/60 ]))


def compute_landmarkers_TSCR(db_name, subsample_output_dir, landmarker_output_dir):

    args = ['tsml/tsml_reduced.jar', '-dp=%s/' % subsample_output_dir, '-ds={}'.format(db_name), '-cn=C45,NB,BayesNet,SVML,SVMQ,RotF,RandF,MLP,NN,DTW,WDTW,TWE,MSM,NN_CID,ERP,DD_DTW,DTD_C,TSF,FastShapelets,ShapeletTransformClassifier,BOP,BOSS']

    def run_command():
        p = Popen(['java', '-jar'] + list(args), stdout=PIPE, stderr=STDOUT)
        return iter(p.stdout.readline, b'')

    for output_line in run_command():
        print(output_line)


    create_landmarkers_deep(db_name, subsample_output_dir, landmarker_output_dir)





    
def read_landmarkes():

    landmarkers_UCR = np.loadtxt("../data/landmarkers_UCR.txt")
    landmarkers_time =np.loadtxt("../data/landmarkers_time.txt")

    return landmarkers_UCR, landmarkers_time


def read_landmarkes_TSCR(db_name, landmarker_output_dir):

    summary = pd.read_csv("%s/Sum_%s /Sum_%s _BIGglobalSummary.csv" % (landmarker_output_dir, db_name, db_name),header=None)

    df = pd.DataFrame(summary.values[1, 1:].reshape(-1, len(summary.values[1, 1:])), columns=summary.values[0, 1:])
    df = df.reindex(sorted(df.columns), axis=1)

    landmarkers= df.values.astype(float)
    df = pd.DataFrame(summary.values[25, 1:].reshape(-1, len(summary.values[1, 1:])), columns=summary.values[24, 1:])
    df = df.reindex(sorted(df.columns), axis=1)

    test_time = df.values.astype(float)
    df = pd.DataFrame(summary.values[31, 1:].reshape(-1, len(summary.values[1, 1:])), columns=summary.values[30, 1:])
    df = df.reindex(sorted(df.columns), axis=1)

    df = df.values.astype(float)
    df[df < 0] = 0
    train_time = df
    land_time = train_time + test_time
    land_time = land_time / 60000

    #Read deep landmarkers and append:

    inception_land = np.loadtxt("%s/%s"% (landmarker_output_dir,db_name)+"_Inception.txt")[0]
    inception_time = np.loadtxt("%s/%s"% (landmarker_output_dir,db_name)+"_Inception.txt")[1]

    resnet_land = np.loadtxt("%s/%s"% (landmarker_output_dir,db_name)+"_Resnet.txt")[0]
    resnet_time = np.loadtxt("%s/%s"% (landmarker_output_dir,db_name)+"_Resnet.txt")[1]
   
    landmarkers= np.column_stack((np.column_stack((landmarkers, resnet_land)), inception_land))
    land_time = np.column_stack((np.column_stack((land_time, resnet_time)), inception_time))

    return landmarkers[0], land_time[0]




def read_results_UCR():

    results_UCR = np.loadtxt("../data/results_UCR.txt")

    return results_UCR




