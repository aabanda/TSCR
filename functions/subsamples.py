import numpy as np
import arff
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sktime.utils.data_io import load_from_tsfile_to_dataframe

def create_subsample(input_dir, UCR_list, output_dir):

    for db_name_ite in UCR_list.values:

        db_name = db_name_ite[0]

        train_x, train_y = load_from_tsfile_to_dataframe(
            "%s/%s/%s_TRAIN.ts" % (input_dir, db_name, db_name))
        test_x, test_y = load_from_tsfile_to_dataframe(
            "%s/%s/%s_TEST.ts" % (input_dir, db_name, db_name))
        data = np.zeros((len(train_y) + len(test_y), len(train_x.iloc[1, 0])))

        for i in range(0, len(train_y)):
            data[i, :] = train_x.iloc[i, :][0]

        k = 0
        for i in range(len(train_y), len(train_y) + len(test_y)):
            data[i, :] = test_x.iloc[k, :][0]
            k = k + 1

        classes = np.concatenate((train_y, test_y))
        classes = classes.astype(int)

        l = data.shape[0]

        if l < 100:
            subratio = 0.8
        elif l < 300:
            subratio = 0.6
        elif l < 800:
            subratio = 0.4
        elif l < 1500:
            subratio = 0.2
        elif l < 5000:
            subratio = 0.1
        else:
            subratio = 0.05

        while l*subratio/len(np.unique(classes))<10:
            subratio = subratio + 0.1

        if subratio > 0.8:
            subratio = 0.8

        s = StratifiedShuffleSplit(test_size=subratio / 2, train_size=subratio / 2)
        train_index, test_index = next(s.split(data, classes))

        data_df = np.concatenate((data[train_index, :], data[test_index, :]))
        classes_df = np.concatenate((classes[train_index], classes[test_index]))
        data_df = np.column_stack((data_df, classes_df))

        df = pd.DataFrame(data_df)
        attributes = [(c.astype(str), 'NUMERIC') for c in df.columns.values[:-1]]
        t = df.columns[-1]
        attributes += [('target', df[t].unique().astype(str).tolist())]

        data = [df.loc[i].values[:-1].tolist() + [df[t].loc[i]] for i in range(df.shape[0])]

        arff_dic = {
            'attributes': attributes,
            'data': data,
            'relation': db_name,
            'description': ''
        }
        if not os.path.exists("%s/%s" % (output_dir,db_name)):
            os.makedirs("%s/%s" % (output_dir,db_name))
        with open("%s/%s/%s.arff" % (output_dir, db_name, db_name), "w", encoding="utf8") as f:
            arff.dump(arff_dic, f)
        print("%s created" % db_name)

    print("Subsample finished!")

    return



def create_subsample_TSCR(input_dir, db_name, output_dir):

        data = pd.read_csv('%s/%s.txt' %(input_dir,db_name), header=None, sep="  ")
        classes = data.values[:, 0]
        data = data.values[:, 1:]


        l = data.shape[0]

        if l < 100:
            subratio = 0.8
        elif l < 300:
            subratio = 0.6
        elif l < 800:
            subratio = 0.4
        elif l < 1500:
            subratio = 0.2
        elif l < 5000:
            subratio = 0.1
        else:
            subratio = 0.05

        while l*subratio/len(np.unique(classes))<10:
            subratio = subratio + 0.1

        if subratio > 0.8:
            subratio = 0.8

        s = StratifiedShuffleSplit(test_size=subratio / 2, train_size=subratio / 2)
        train_index, test_index = next(s.split(data, classes))

        data_df = np.concatenate((data[train_index, :], data[test_index, :]))
        classes_df = np.concatenate((classes[train_index], classes[test_index]))
        data_df = np.column_stack((data_df, classes_df))

        df = pd.DataFrame(data_df)
        attributes = [(c.astype(str), 'NUMERIC') for c in df.columns.values[:-1]]
        t = df.columns[-1]
        attributes += [('target', df[t].unique().astype(str).tolist())]

        data = [df.loc[i].values[:-1].tolist() + [df[t].loc[i]] for i in range(df.shape[0])]

        arff_dic = {
            'attributes': attributes,
            'data': data,
            'relation': db_name,
            'description': ''
        }
        if not os.path.exists("%s/%s" % (output_dir,db_name)):
            os.makedirs("%s/%s" % (output_dir,db_name))
        with open("%s/%s/%s.arff" % (output_dir, db_name, db_name), "w", encoding="utf8") as f:
            arff.dump(arff_dic, f)
        print("%s created" % db_name)

        print("Subsample finished!")

        return
