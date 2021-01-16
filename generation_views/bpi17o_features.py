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
    benchmark = "bpic2017_o"

    for fold in range(n_folds):
        info_dataset = []
        train_data_name = "fold/bpic2017_o/" + str(benchmark) + "_kfoldcv_" + str(fold) + "_train.csv"
        test_data_name = "fold/bpic2017_o/" + str(benchmark) + "_kfoldcv_" + str(fold) + "_test.csv"

        df_train = pd.read_csv(train_data_name, sep=',', encoding='utf-8', header=None)
        df_test = pd.read_csv(test_data_name, sep=',', encoding='utf-8', header=None)

        df_train.columns = ['Case ID', 'Activity','Timestamp', 'MonthlyCost', 'CreditScore', 'FirstWithdrawalAmount', 'OfferedAmount', 'NumberOfTerms', 'Action', 'Resource','null']
        df_test.columns = ['Case ID', 'Activity','Timestamp', 'MonthlyCost', 'CreditScore', 'FirstWithdrawalAmount', 'OfferedAmount', 'NumberOfTerms', 'Action', 'Resource','null']

        df_train['Resource'] = df_train['Resource'].fillna('-1')
        df_test['Resource'] = df_test['Resource'].fillna('-1')

        scaler = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy='quantile')

        df_train[['FirstWithdrawalAmount']] = scaler.fit_transform(df_train[['FirstWithdrawalAmount']])
        df_test[['FirstWithdrawalAmount']] = scaler.transform(df_test[['FirstWithdrawalAmount']])

        df_train[['OfferedAmount']] = scaler.fit_transform(df_train[['OfferedAmount']])
        df_test[['OfferedAmount']] = scaler.transform(df_test[['OfferedAmount']])

        list_act = Union(df_train.Activity.unique(), df_test.Activity.unique())
        mapping = dict(zip(set(list_act), range(1, len(list_act) + 1)))

        list_res = Union(df_train.Resource.unique(), df_test.Resource.unique())
        mapping2 = dict(zip(set(list_res), range(1, len(list_res) + 1)))

        list_action = Union(df_train.Action.unique(), df_test.Action.unique())
        mapping_action = dict(zip(set(list_action), range(1, len(list_action) + 1)))

        list_FirstWithdrawalAmount = Union(df_train.FirstWithdrawalAmount.unique(), df_test.FirstWithdrawalAmount.unique())
        mapping_FirstWithdrawalAmount = dict(zip(set(list_FirstWithdrawalAmount), range(1, len(list_FirstWithdrawalAmount) + 1)))

        list_OfferedAmount = Union(df_train.OfferedAmount.unique(), df_test.OfferedAmount.unique())
        mapping_OfferedAmount = dict(zip(set(list_OfferedAmount), range(1, len(list_OfferedAmount) + 1)))

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

        df_train.Action = [mapping_action[item] for item in df_train.Action]
        df_test.Action = [mapping_action[item] for item in df_test.Action]

        df_train.FirstWithdrawalAmount = [mapping_FirstWithdrawalAmount[item] for item in df_train.FirstWithdrawalAmount]
        df_test.FirstWithdrawalAmount = [mapping_FirstWithdrawalAmount[item] for item in df_test.FirstWithdrawalAmount]

        df_train.OfferedAmount = [mapping_OfferedAmount[item] for item in df_train.OfferedAmount]
        df_test.OfferedAmount = [mapping_OfferedAmount[item] for item in df_test.OfferedAmount]

        act_train = df_train.groupby('Case ID', sort=False).agg({'Activity': lambda x: list(x)})
        res_train = df_train.groupby('Case ID', sort=False).agg({'Resource': lambda x: list(x)})
        temp_train = df_train.groupby('Case ID', sort=False).agg({'Timestamp': lambda x: list(x)})
        action_train = df_train.groupby('Case ID').agg({'Action': lambda x: list(x)})
        OfferedAmount_train = df_train.groupby('Case ID').agg({'OfferedAmount': lambda x: list(x)})
        FirstWithdrawalAmount_train = df_train.groupby('Case ID').agg({'FirstWithdrawalAmount': lambda x: list(x)})
        MonthlyCost_train = df_train.groupby('Case ID', sort=False).agg({'MonthlyCost': lambda x: list(x)})
        CreditScore_train = df_train.groupby('Case ID', sort=False).agg({'CreditScore': lambda x: list(x)})
        NumberOfTerms_train = df_train.groupby('Case ID', sort=False).agg({'NumberOfTerms': lambda x: list(x)})

        act_test = df_test.groupby('Case ID', sort=False).agg({'Activity': lambda x: list(x)})
        res_test = df_test.groupby('Case ID', sort=False).agg({'Resource': lambda x: list(x)})
        temp_test = df_test.groupby('Case ID', sort=False).agg({'Timestamp': lambda x: list(x)})
        action_test = df_test.groupby('Case ID').agg({'Action': lambda x: list(x)})
        OfferedAmount_test = df_test.groupby('Case ID').agg({'OfferedAmount': lambda x: list(x)})
        FirstWithdrawalAmount_test = df_test.groupby('Case ID').agg({'FirstWithdrawalAmount': lambda x: list(x)})
        MonthlyCost_test = df_test.groupby('Case ID', sort=False).agg({'MonthlyCost': lambda x: list(x)})
        CreditScore_test = df_test.groupby('Case ID', sort=False).agg({'CreditScore': lambda x: list(x)})
        NumberOfTerms_test = df_test.groupby('Case ID', sort=False).agg({'NumberOfTerms': lambda x: list(x)})

        # extract prefix from different view
        sequence_train, y_train = get_sequence(act_train, max_trace, mean_trace)
        sequence_test, y_test = get_sequence(act_test, max_trace, mean_trace)

        resource_train = get_resource(res_train, max_trace, mean_trace)
        resource_test = get_resource(res_test, max_trace, mean_trace)

        FirstWithdrawalAmount_train = get_resource(FirstWithdrawalAmount_train, max_trace, mean_trace)
        FirstWithdrawalAmount_test = get_resource(FirstWithdrawalAmount_test, max_trace, mean_trace)

        OfferedAmount_train = get_resource(OfferedAmount_train, max_trace, mean_trace)
        OfferedAmount_test = get_resource(OfferedAmount_test, max_trace, mean_trace)

        NumberOfTerms_train = get_resource(NumberOfTerms_train, max_trace, mean_trace)
        NumberOfTerms_test = get_resource(NumberOfTerms_test, max_trace, mean_trace)

        Action_train = get_resource(action_train, max_trace, mean_trace)
        Action_test = get_resource(action_test, max_trace, mean_trace)

        CreditScore_train = get_resource(CreditScore_train, max_trace, mean_trace)
        CreditScore_test = get_resource(CreditScore_test, max_trace, mean_trace)

        MonthlyCost_train = get_resource(MonthlyCost_train, max_trace, mean_trace)
        MonthlyCost_test = get_resource(MonthlyCost_test, max_trace, mean_trace)

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

        scaler2 = MinMaxScaler()
        MonthlyCost_train = scaler2.fit_transform(MonthlyCost_train)
        MonthlyCost_test = scaler2.transform(MonthlyCost_test)

        scaler3 = MinMaxScaler()
        CreditScore_train = scaler3.fit_transform(CreditScore_train)
        CreditScore_test = scaler3.transform(CreditScore_test)

        scaler4 = MinMaxScaler()
        NumberOfTerms_train = scaler4.fit_transform(CreditScore_train)
        NumberOfTerms_test = scaler4.transform(CreditScore_test)

        sequence_train = np.asarray(sequence_train)
        sequence_test = np.asarray(sequence_test)

        resource_train = np.asarray(resource_train)
        resource_test = np.asarray(resource_test)

        print("max lengh", max_trace)
        print("num unique act", len(mapping))
        print("num unique res", len(mapping2))
        info_dataset.append(mean_trace)
        info_dataset.append(len(mapping))
        info_dataset.append(len(mapping2))
        info_dataset.append(len(mapping_action))
        info_dataset.append(len(mapping_FirstWithdrawalAmount))
        info_dataset.append(len(mapping_OfferedAmount))

        np.save("fold/bpic2017_o/" + str(benchmark) + "_act_" + str(fold) + "_train.npy", sequence_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_res_" + str(fold) + "_train.npy", resource_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_diff2_" + str(fold) + "_train.npy", diff2_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_MonthlyCost_" + str(fold) + "_train.npy", MonthlyCost_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_CreditScore_" + str(fold) + "_train.npy", CreditScore_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_FirstWithdrawalAmount_" + str(fold) + "_train.npy", FirstWithdrawalAmount_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_OfferedAmount_" + str(fold) + "_train.npy", OfferedAmount_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_NumberOfTerms_" + str(fold) + "_train.npy", NumberOfTerms_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_Action_" + str(fold) + "_train.npy", Action_train)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_y_" + str(fold) + "_train.npy", y_train)

        np.save("fold/bpic2017_o/" + str(benchmark) + "_act_" + str(fold) + "_test.npy", sequence_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_res_" + str(fold) + "_test.npy", resource_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_diff2_" + str(fold) + "_test.npy", diff2_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_MonthlyCost_" + str(fold) + "_test.npy", MonthlyCost_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_CreditScore_" + str(fold) + "_test.npy", CreditScore_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_FirstWithdrawalAmount_" + str(fold) + "_test.npy", FirstWithdrawalAmount_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_OfferedAmount_" + str(fold) + "_test.npy", OfferedAmount_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_NumberOfTerms_" + str(fold) + "_test.npy", NumberOfTerms_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_Action_" + str(fold) + "_test.npy", Action_test)
        np.save("fold/bpic2017_o/" + str(benchmark) + "_y_" + str(fold) + "_test.npy", y_test)

    info_dataset = np.asarray(info_dataset)
    np.save("fold/bpic2017_o/" + str(benchmark) + "_info_dataset.npy", info_dataset)
