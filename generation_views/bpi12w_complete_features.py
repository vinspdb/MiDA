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

seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed

set_random_seed(seed)
import random

random.seed(seed)  # for reproducibility
import os
from keras.preprocessing.sequence import pad_sequences
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
    benchmark = "bpi12w_complete"

    for fold in range(n_folds):
        info_dataset = []
        train_data_name = "fold/bpi12w_complete/" + str(benchmark) + "_kfoldcv_" + str(fold) + "_train.csv"
        test_data_name = "fold/bpi12w_complete/" + str(benchmark) + "_kfoldcv_" + str(fold) + "_test.csv"

        df_train = pd.read_csv(train_data_name, sep=',', encoding='utf-8', header=None)
        df_test = pd.read_csv(test_data_name, sep=',', encoding='utf-8', header=None)

        df_train.columns = ['Case ID', 'Activity', 'Resource', 'Timestamp', 'Amount']
        df_test.columns = ['Case ID', 'Activity', 'Resource', 'Timestamp', 'Amount']

        df_train['Resource'] = df_train['Resource'].fillna('-1')
        df_test['Resource'] = df_test['Resource'].fillna('-1')

        scaler = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy='quantile')

        df_train[['Amount']] = scaler.fit_transform(df_train[['Amount']])
        df_test[['Amount']] = scaler.transform(df_test[['Amount']])

        list_act = Union(df_train.Activity.unique(), df_test.Activity.unique())
        mapping = dict(zip(set(list_act), range(1, len(list_act) + 1)))

        list_res = Union(df_train.Resource.unique(), df_test.Resource.unique())
        mapping2 = dict(zip(set(list_res), range(1, len(list_res) + 1)))

        list_amount = Union(df_train.Amount.unique(), df_test.Amount.unique())
        mapping_amount = dict(zip(set(list_amount), range(1, len(list_amount) + 1)))
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

        df_train.Amount = [mapping_amount[item] for item in df_train.Amount]
        df_test.Amount = [mapping_amount[item] for item in df_test.Amount]

        act_train = df_train.groupby('Case ID', sort=False).agg({'Activity': lambda x: list(x)})
        res_train = df_train.groupby('Case ID', sort=False).agg({'Resource': lambda x: list(x)})
        temp_train = df_train.groupby('Case ID', sort=False).agg({'Timestamp': lambda x: list(x)})
        amount_train = df_train.groupby('Case ID').agg({'Amount': lambda x: list(x)})

        act_test = df_test.groupby('Case ID', sort=False).agg({'Activity': lambda x: list(x)})
        res_test = df_test.groupby('Case ID', sort=False).agg({'Resource': lambda x: list(x)})
        temp_test = df_test.groupby('Case ID', sort=False).agg({'Timestamp': lambda x: list(x)})
        amount_test = df_test.groupby('Case ID').agg({'Amount': lambda x: list(x)})

        # extract prefix from different view
        sequence_train, y_train = get_sequence(act_train, max_trace, mean_trace)
        sequence_test, y_test = get_sequence(act_test, max_trace, mean_trace)

        resource_train = get_resource(res_train, max_trace, mean_trace)
        resource_test = get_resource(res_test, max_trace, mean_trace)

        qt_train = get_resource(amount_train, max_trace, mean_trace)
        qt_test = get_resource(amount_test, max_trace, mean_trace)

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

        qt_train = np.asarray(qt_train)
        qt_test = np.asarray(qt_test)
        print("max lengh", max_trace)
        print("num unique act", len(mapping))
        print("num unique res", len(mapping2))
        info_dataset.append(mean_trace)
        info_dataset.append(len(mapping))
        info_dataset.append(len(mapping2))
        info_dataset.append(len(mapping_amount))


        np.save("fold/bpi12w_complete/" + str(benchmark) + "_act_" + str(fold) + "_train.npy", sequence_train)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_res_" + str(fold) + "_train.npy", resource_train)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_diff2_" + str(fold) + "_train.npy", diff2_train)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_amount_" + str(fold) + "_train.npy", qt_train)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_y_" + str(fold) + "_train.npy", y_train)

        np.save("fold/bpi12w_complete/" + str(benchmark) + "_act_" + str(fold) + "_test.npy", sequence_test)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_res_" + str(fold) + "_test.npy", resource_test)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_diff2_" + str(fold) + "_test.npy", diff2_test)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_amount_" + str(fold) + "_test.npy", qt_test)
        np.save("fold/bpi12w_complete/" + str(benchmark) + "_y_" + str(fold) + "_test.npy", y_test)
       

    info_dataset = np.asarray(info_dataset)
    np.save("fold/bpi12w_complete/" + str(benchmark) + "_info_dataset.npy", info_dataset)
