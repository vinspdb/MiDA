import datetime
from datetime import datetime
import time
import pandas as pd
from keras.layers import Embedding, Dense, Reshape, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer, LabelEncoder, OneHotEncoder, \
    KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from keras.layers import concatenate, Input, LSTM, Flatten
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed

set_random_seed(seed)
import random

random.seed(seed)  # for reproducibility
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def get_time(prova,max_trace):
    i = 0
    s = (max_trace)
    list_seq = []
    datetimeFormat = '%Y/%m/%d %H:%M:%S.%f'
    while i < len(prova):
        list_temp = []
        seq = np.zeros(s)
        j = 0
        while j < (len(prova.iat[i, 0]) - 1):
            t = time.strptime(prova.iat[i, 0][0 + j], datetimeFormat)
            list_temp.append(datetime.fromtimestamp(time.mktime(t)))
            new_seq = np.append(seq, list_temp)
            cut = len(list_temp)
            new_seq = new_seq[cut:]
            list_seq.append(new_seq)
            j = j + 1
        i = i + 1
    return list_seq

def get_sequence(prova, max_trace, mean_trace):
    i = 0
    s = (max_trace)
    list_seq = []
    list_label = []
    while i < len(prova):
        list_temp = []
        seq = np.zeros(s)
        j = 0
        # prova.iat[i, 0].insert(0, 0)
        while j < (len(prova.iat[i, 0]) - 1):
            list_temp.append(prova.iat[i, 0][0 + j])
            new_seq = np.append(seq, list_temp)
            cut = len(list_temp)
            new_seq = new_seq[cut:]
            list_seq.append(new_seq[-mean_trace:])
            list_label.append(prova.iat[i, 0][j + 1])
            j = j + 1
        i = i + 1
    return list_seq, list_label


def get_resource(prova, max_trace, mean_trace):
    i = 0
    s = (max_trace)
    list_seq = []
    while i < len(prova):
        list_temp = []
        seq = np.zeros(s)
        j = 0
        # prova.iat[i, 0].insert(0, 0)
        while j < (len(prova.iat[i, 0]) - 1):
            list_temp.append(prova.iat[i, 0][0 + j])
            new_seq = np.append(seq, list_temp)
            cut = len(list_temp)
            new_seq = new_seq[cut:]
            list_seq.append(new_seq[-mean_trace:])
            j = j + 1
        i = i + 1
    return list_seq


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def difference_feature(time_prefix_new,list_sequence_prefix):
    list_start_from_case = []
    i = 0
    while i < len(time_prefix_new):
        time_feature2 = []
        if len(list_sequence_prefix[i]) == 1:
            time_feature2.append([0])
        else:
            diff_start = [x - time_prefix_new[i][0] for x in time_prefix_new[i]]
            diff_start_sec = [((86400 * item.days + item.seconds) / 86400) for item in diff_start]
            time_feature2.append(diff_start_sec)
        list_start_from_case.append(time_feature2)
        i = i + 1
    return  list_start_from_case


if __name__ == "__main__":
    n_folds = 3
    benchmark = "bpi13_incidents"

    for fold in range(n_folds):
        info_dataset = []
        train_data_name = "fold/bpi13_incidents/" + str(benchmark) + "_kfoldcv_" + str(fold) + "_train.csv"
        test_data_name = "fold/bpi13_incidents/" + str(benchmark) + "_kfoldcv_" + str(fold) + "_test.csv"

        df_train = pd.read_csv(train_data_name, sep=',', encoding='windows-1252', header=None)
        df_test = pd.read_csv(test_data_name, sep=',', encoding='windows-1252', header=None)

        df_train.columns = ['Case ID', 'Activity', 'Resource', 'Timestamp', 'Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7']
        df_test.columns = ['Case ID', 'Activity', 'Resource', 'Timestamp', 'Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7']

        df_train['Resource'] = df_train['Resource'].fillna('-1')
        df_test['Resource'] = df_test['Resource'].fillna('-1')
        df_train['Att1'] = df_train['Att1'].fillna('-1')
        df_test['Att1'] = df_test['Att1'].fillna('-1')
        df_train['Att2'] = df_train['Att2'].fillna('-1')
        df_test['Att2'] = df_test['Att2'].fillna('-1')
        df_train['Att3'] = df_train['Att3'].fillna('-1')
        df_test['Att3'] = df_test['Att3'].fillna('-1')
        df_train['Att4'] = df_train['Att4'].fillna('-1')
        df_test['Att4'] = df_test['Att4'].fillna('-1')
        df_train['Att5'] = df_train['Att5'].fillna('-1')
        df_test['Att5'] = df_test['Att5'].fillna('-1')
        df_train['Att6'] = df_train['Att6'].fillna('-1')
        df_test['Att6'] = df_test['Att6'].fillna('-1')
        df_train['Att7'] = df_train['Att7'].fillna('-1')
        df_test['Att7'] = df_test['Att7'].fillna('-1')

        list_act = Union(df_train.Activity.unique(), df_test.Activity.unique())
        mapping = dict(zip(set(list_act), range(1, len(list_act) + 1)))

        list_res = Union(df_train.Resource.unique(), df_test.Resource.unique())
        mapping2 = dict(zip(set(list_res), range(1, len(list_res) + 1)))

        list_att1 = Union(df_train.Att1.unique(), df_test.Att1.unique())
        mapping_att1 = dict(zip(set(list_att1), range(1, len(list_att1) + 1)))

        list_att2 = Union(df_train.Att2.unique(), df_test.Att2.unique())
        mapping_att2 = dict(zip(set(list_att2), range(1, len(list_att2) + 1)))

        list_att3 = Union(df_train.Att3.unique(), df_test.Att3.unique())
        mapping_att3 = dict(zip(set(list_att3), range(1, len(list_att3) + 1)))

        list_att4 = Union(df_train.Att4.unique(), df_test.Att4.unique())
        mapping_att4 = dict(zip(set(list_att4), range(1, len(list_att4) + 1)))

        list_att5 = Union(df_train.Att5.unique(), df_test.Att5.unique())
        mapping_att5 = dict(zip(set(list_att5), range(1, len(list_att5) + 1)))

        list_att6 = Union(df_train.Att6.unique(), df_test.Att6.unique())
        mapping_att6 = dict(zip(set(list_att6), range(1, len(list_att6) + 1)))

        list_att7 = Union(df_train.Att7.unique(), df_test.Att7.unique())
        mapping_att7 = dict(zip(set(list_att7), range(1, len(list_att7) + 1)))


        tot = df_train.append(df_test)
        cont_trace = tot['Case ID'].value_counts(dropna=False)

        max_trace = max(cont_trace)
        print(df_train.head(5))
        mean_trace = int(round(np.mean(cont_trace)))
        print(mean_trace)

        df_train.Activity = [mapping[item] for item in df_train.Activity]
        df_test.Activity = [mapping[item] for item in df_test.Activity]

        df_train.Resource = [mapping2[item] for item in df_train.Resource]
        df_test.Resource = [mapping2[item] for item in df_test.Resource]

        df_train.Att1 = [mapping_att1[item] for item in df_train.Att1]
        df_test.Att1 = [mapping_att1[item] for item in df_test.Att1]

        df_train.Att2 = [mapping_att2[item] for item in df_train.Att2]
        df_test.Att2 = [mapping_att2[item] for item in df_test.Att2]

        df_train.Att3 = [mapping_att3[item] for item in df_train.Att3]
        df_test.Att3 = [mapping_att3[item] for item in df_test.Att3]

        df_train.Att4 = [mapping_att4[item] for item in df_train.Att4]
        df_test.Att4 = [mapping_att4[item] for item in df_test.Att4]

        df_train.Att5 = [mapping_att5[item] for item in df_train.Att5]
        df_test.Att5 = [mapping_att5[item] for item in df_test.Att5]

        df_train.Att6 = [mapping_att6[item] for item in df_train.Att6]
        df_test.Att6 = [mapping_att6[item] for item in df_test.Att6]

        df_train.Att7 = [mapping_att7[item] for item in df_train.Att7]
        df_test.Att7 = [mapping_att7[item] for item in df_test.Att7]

        act_train = df_train.groupby('Case ID', sort=False).agg({'Activity': lambda x: list(x)})
        res_train = df_train.groupby('Case ID', sort=False).agg({'Resource': lambda x: list(x)})
        temp_train = df_train.groupby('Case ID', sort=False).agg({'Timestamp': lambda x: list(x)})
        att1_train = df_train.groupby('Case ID', sort=False).agg({'Att1': lambda x: list(x)})
        att2_train = df_train.groupby('Case ID', sort=False).agg({'Att2': lambda x: list(x)})
        att3_train = df_train.groupby('Case ID', sort=False).agg({'Att3': lambda x: list(x)})
        att4_train = df_train.groupby('Case ID', sort=False).agg({'Att4': lambda x: list(x)})
        att5_train = df_train.groupby('Case ID', sort=False).agg({'Att5': lambda x: list(x)})
        att6_train = df_train.groupby('Case ID', sort=False).agg({'Att6': lambda x: list(x)})
        att7_train = df_train.groupby('Case ID', sort=False).agg({'Att7': lambda x: list(x)})


        act_test = df_test.groupby('Case ID', sort=False).agg({'Activity': lambda x: list(x)})
        res_test = df_test.groupby('Case ID', sort=False).agg({'Resource': lambda x: list(x)})
        temp_test = df_test.groupby('Case ID', sort=False).agg({'Timestamp': lambda x: list(x)})
        att1_test = df_test.groupby('Case ID', sort=False).agg({'Att1': lambda x: list(x)})
        att2_test = df_test.groupby('Case ID', sort=False).agg({'Att2': lambda x: list(x)})
        att3_test = df_test.groupby('Case ID', sort=False).agg({'Att3': lambda x: list(x)})
        att4_test = df_test.groupby('Case ID', sort=False).agg({'Att4': lambda x: list(x)})
        att5_test = df_test.groupby('Case ID', sort=False).agg({'Att5': lambda x: list(x)})
        att6_test = df_test.groupby('Case ID', sort=False).agg({'Att6': lambda x: list(x)})
        att7_test = df_test.groupby('Case ID', sort=False).agg({'Att7': lambda x: list(x)})


        # extract prefix from different view
        sequence_train, y_train = get_sequence(act_train, max_trace, mean_trace)
        sequence_test, y_test = get_sequence(act_test, max_trace, mean_trace)

        resource_train = get_resource(res_train, max_trace, mean_trace)
        resource_test = get_resource(res_test, max_trace, mean_trace)

        list_att1_train = get_resource(att1_train, max_trace, mean_trace)
        list_att1_test = get_resource(att1_test, max_trace, mean_trace)

        list_att2_train = get_resource(att2_train, max_trace, mean_trace)
        list_att2_test = get_resource(att2_test, max_trace, mean_trace)

        list_att3_train = get_resource(att3_train, max_trace, mean_trace)
        list_att3_test = get_resource(att3_test, max_trace, mean_trace)

        list_att4_train = get_resource(att4_train, max_trace, mean_trace)
        list_att4_test = get_resource(att4_test, max_trace, mean_trace)

        list_att5_train = get_resource(att5_train, max_trace, mean_trace)
        list_att5_test = get_resource(att5_test, max_trace, mean_trace)

        list_att6_train = get_resource(att6_train, max_trace, mean_trace)
        list_att6_test = get_resource(att6_test, max_trace, mean_trace)

        list_att7_train = get_resource(att7_train, max_trace, mean_trace)
        list_att7_test = get_resource(att7_test, max_trace, mean_trace)

        timestamp_train = get_time(temp_train, max_trace)
        timestamp_test = get_time(temp_test, max_trace)


        i = 0
        time_prefix_new_train = []
        list_sequence_prefix_train = []
        while i < len(timestamp_train):
            time_val = [x for x in timestamp_train[i] if x != 0.0]
            act_val = [x for x in sequence_train[i] if x != 0.0]
            time_prefix_new_train.append(time_val)
            list_sequence_prefix_train.append(act_val)
            i = i + 1

        i = 0
        time_prefix_new_test = []
        list_sequence_prefix_test = []
        while i < len(timestamp_test):
            time_val = [x for x in timestamp_test[i] if x != 0.0]
            act_val = [x for x in sequence_test[i] if x != 0.0]
            time_prefix_new_test.append(time_val)
            list_sequence_prefix_test.append(act_val)
            i = i + 1

        diff2_train = difference_feature(time_prefix_new_train, list_sequence_prefix_train)
        diff2_test = difference_feature(time_prefix_new_test, list_sequence_prefix_test)

        diff2_train = np.asarray(diff2_train)
        diff2_test = np.asarray(diff2_test)

        diff2_train = diff2_train.reshape(len(diff2_train), )
        diff2_test = diff2_test.reshape(len(diff2_test), )

        diff2_train = pad_sequences(diff2_train, maxlen=max_trace, padding='pre', dtype='float64')
        diff2_test = pad_sequences(diff2_test, maxlen=max_trace, padding='pre', dtype='float64')

        diff2_train = [x[-mean_trace:] for x in diff2_train]
        diff2_test = [x[-mean_trace:] for x in diff2_test]
        
        scaler = MinMaxScaler()

        diff2_train = scaler.fit_transform(diff2_train)
        diff2_test = scaler.transform(diff2_test)

        sequence_train = np.asarray(sequence_train)
        sequence_test = np.asarray(sequence_test)

        resource_train = np.asarray(resource_train)
        resource_test = np.asarray(resource_test)

        list_att1_train = np.asarray(list_att1_train)
        list_att1_test = np.asarray(list_att1_test)

        list_att2_train = np.asarray(list_att2_train)
        list_att2_test = np.asarray(list_att2_test)

        list_att3_train = np.asarray(list_att3_train)
        list_att3_test = np.asarray(list_att3_test)

        list_att4_train = np.asarray(list_att4_train)
        list_att4_test = np.asarray(list_att4_test)

        list_att5_train = np.asarray(list_att5_train)
        list_att5_test = np.asarray(list_att5_test)

        list_att6_train = np.asarray(list_att6_train)
        list_att6_test = np.asarray(list_att6_test)

        list_att7_train = np.asarray(list_att7_train)
        list_att7_test = np.asarray(list_att7_test)

        print("max lengh", max_trace)
        print("num unique act", len(mapping))
        print("num unique res", len(mapping2))
        info_dataset.append(mean_trace)
        info_dataset.append(len(mapping))
        info_dataset.append(len(mapping2))
        info_dataset.append(len(mapping_att1))
        info_dataset.append(len(mapping_att2))
        info_dataset.append(len(mapping_att3))
        info_dataset.append(len(mapping_att4))
        info_dataset.append(len(mapping_att5))
        info_dataset.append(len(mapping_att6))
        info_dataset.append(len(mapping_att7))

        print("INFO DATASET")
        print(info_dataset)

        np.save("fold/bpi13_incidents/" + str(benchmark) + "_act_" + str(fold) + "_train.npy", sequence_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_res_" + str(fold) + "_train.npy", resource_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_diff2_" + str(fold) + "_train.npy", diff2_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att1_" + str(fold) + "_train.npy", list_att1_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att2_" + str(fold) + "_train.npy", list_att2_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att3_" + str(fold) + "_train.npy", list_att3_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att4_" + str(fold) + "_train.npy", list_att4_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att5_" + str(fold) + "_train.npy", list_att5_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att6_" + str(fold) + "_train.npy", list_att6_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att7_" + str(fold) + "_train.npy", list_att7_train)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_y_" + str(fold) + "_train.npy", y_train)

        np.save("fold/bpi13_incidents/" + str(benchmark) + "_act_" + str(fold) + "_test.npy", sequence_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_res_" + str(fold) + "_test.npy", resource_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_diff2_" + str(fold) + "_test.npy", diff2_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att1_" + str(fold) + "_test.npy", list_att1_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att2_" + str(fold) + "_test.npy", list_att2_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att3_" + str(fold) + "_test.npy", list_att3_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att4_" + str(fold) + "_test.npy", list_att4_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att5_" + str(fold) + "_test.npy", list_att5_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att6_" + str(fold) + "_test.npy", list_att6_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_att7_" + str(fold) + "_test.npy", list_att7_test)
        np.save("fold/bpi13_incidents/" + str(benchmark) + "_y_" + str(fold) + "_test.npy", y_test)

    info_dataset = np.asarray(info_dataset)
    np.save("fold/bpi13_incidents/" + str(benchmark) + "_info_dataset.npy", info_dataset)
