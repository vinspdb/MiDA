import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pickle


class ReadLog:
    def __init__(self, eventlog):
        self._eventlog = eventlog
        self._list_cat_cols = []
        self._list_num_cols = []

    @staticmethod
    def Union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    @staticmethod
    def to_sec(c):
        sec = 86400 * c.days + c.seconds + c.microseconds / 1000000
        return sec / 86400

    def time_format(self, time_stamp):
        '''
        :param time_stamp: oggetto timestamp
        :return: converte l'oggetto timestamp utile in fase di calcolo dei tempi
        '''
        try:
            date_format_str = '%Y/%m/%d %H:%M:%S.%f%z'
            conversion = datetime.strptime(time_stamp, date_format_str)
        except:
            date_format_str = '%Y/%m/%d %H:%M:%S.%f'
            conversion = datetime.strptime(time_stamp, date_format_str)
        return conversion

    def get_time(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(self.to_sec(self.time_format(sequence.iat[i, 0][j])-self.time_format(sequence.iat[i, 0][0])))
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])
                j = j + 1
            i = i + 1

        list_seq = np.array(list_seq)
        return list_seq

    def get_seq_view(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])
                j = j + 1
            i = i + 1
        return list_seq

    def get_sequence(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])
                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq, list_label

    def mapping(self, df_train, df_test, col):
        list_word = self.Union(df_train[col].unique(), df_test[col].unique())
        mapping = dict(zip(set(list_word), range(1, len(list_word) + 1)))
        return mapping

    def mapping_cat(self, col, df_train, df_test, max_trace, mean_trace, fold):

            if col == 'timestamp':
                scaler = MinMaxScaler()
                view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_train = self.get_time(view_train, max_trace, mean_trace)
                view_test = self.get_time(view_test, max_trace, mean_trace)
                view_train = scaler.fit_transform(view_train)
                view_test = scaler.transform(view_test)
                np.save("fold/"+self._eventlog+"/" + self._eventlog + '_' +  col + '_'+  str(fold) + "_train.npy", view_train)
                np.save("fold/"+self._eventlog+"/" + self._eventlog + '_' + col + '_'+  str(fold) + "_test.npy", view_test)
                self._list_num_cols.append(col)
            elif col == 'case':
                view_train = None
                view_test = None
            else:
                if is_numeric_dtype(df_train[col]):
                    scaler = MinMaxScaler()
                    view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                    view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})
                    view_train = self.get_seq_view(view_train, max_trace, mean_trace)
                    view_test = self.get_seq_view(view_test, max_trace, mean_trace)
                    view_train = scaler.fit_transform(view_train)
                    view_test = scaler.transform(view_test)
                    np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' +  col + '_'+  str(fold) + "_train.npy",view_train)
                    np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' +  col + '_'+  str(fold) + "_test.npy", view_test)
                    self._list_num_cols.append(col)
                else:
                    mapping = self.mapping(df_train, df_test, col)
                    df_train[col] = [mapping[item] for item in df_train[col]]
                    df_test[col] = [mapping[item] for item in df_test[col]]
                    view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                    view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})
                    if col == 'activity':
                        view_train, label_train = self.get_sequence(view_train, max_trace, mean_trace)
                        view_test, label_test = self.get_sequence(view_test, max_trace, mean_trace)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_train.npy", view_train)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_test.npy", view_test)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_info.npy", len(mapping))
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_y_' + str(fold) + "_train.npy", label_train)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_y_' + str(fold) + "_test.npy", label_test)
                        self._list_cat_cols.append(col)
                    else:
                        view_train = self.get_seq_view(view_train, max_trace, mean_trace)
                        view_test = self.get_seq_view(view_test, max_trace, mean_trace)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_train.npy",view_train)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_test.npy", view_test)
                        np.save("fold/" + self._eventlog + "/" + self._eventlog + '_' + col + '_' + str(fold) + "_info.npy", len(mapping))
                        self._list_cat_cols.append(col)

    def readView(self):
        for fold in range(3):
            self._list_cat_cols = []
            self._list_num_cols = []
            df_train = pd.read_csv(
                "fold/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(fold) + "_train.csv", sep=',')
            df_test = pd.read_csv(
                "fold/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(fold) + "_test.csv", sep=',')
            if self._eventlog == 'bpi12w_complete' or self._eventlog == 'bpi12_all_complete' or self._eventlog == 'bpi12_work_all':
                df_train['resource'] = 'Res' + df_train['resource'].astype(str)
                df_test['resource'] = 'Res' + df_test['resource'].astype(str)
            full_df = df_train.append(df_test)
            cont_trace = full_df['case'].value_counts(dropna=False)
            max_trace = max(cont_trace)
            mean_trace = int(round(np.mean(cont_trace)))
            for col in df_train.columns:
                self.mapping_cat(col, df_train, df_test, max_trace, mean_trace, fold)

        with open("fold/" + self._eventlog + "/" + self._eventlog + '_seq_length.pickle', 'wb') as handle:
            pickle.dump(mean_trace, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("fold/" + self._eventlog + "/" + self._eventlog + '_cat_cols.pickle', 'wb') as handle:
            pickle.dump(self._list_cat_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("fold/" + self._eventlog + "/" + self._eventlog + '_num_cols.pickle', 'wb') as handle:
            pickle.dump(self._list_num_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
