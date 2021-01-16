import numpy as np

seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed

set_random_seed(seed)
import numpy as np
from sklearn import preprocessing
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
benchmark = 'bpi12_work_all'

info_dataset = np.load("fold/bpi12_work_all/" + str(benchmark) + "_info_dataset.npy")

lenght_seq = info_dataset[0]
num_act = info_dataset[1]
num_res = info_dataset[2]
num_amount = info_dataset[3]


acc_list = []
f1_list = []
rec_list = []
pre_list = []


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)

for f in range(3):
    outfile2 = open("bpi12_work_allmodel_verifica.txt", 'a')

    print("FOLD numero----------------------->", f)
    print("<--------------------------------->")
    sequence_train = np.load("fold/bpi12_work_all/" + str(benchmark) + "_act_" + str(f) + "_train.npy")
    resource_train = np.load("fold/bpi12_work_all/" + str(benchmark) + "_res_" + str(f) + "_train.npy")
    diff2_train = np.load("fold/bpi12_work_all/" + str(benchmark) + "_diff2_" + str(f) + "_train.npy")
    qt_train = np.load("fold/bpi12_work_all/" + str(benchmark) + "_amount_" + str(f) + "_train.npy")
    y_train = np.load("fold/bpi12_work_all/" + str(benchmark) + "_y_" + str(f) + "_train.npy")

    # test views
    sequence_test = np.load("fold/bpi12_work_all/" + str(benchmark) + "_act_" + str(f) + "_test.npy")
    resource_test = np.load("fold/bpi12_work_all/" + str(benchmark) + "_res_" + str(f) + "_test.npy")
    diff2_test = np.load("fold/bpi12_work_all/" + str(benchmark) + "_diff2_" + str(f) + "_test.npy")
    qt_test = np.load("fold/bpi12_work_all/" + str(benchmark) + "_amount_" + str(f) + "_test.npy")
    y_test = np.load("fold/bpi12_work_all/" + str(benchmark) + "_y_" + str(f) + "_test.npy")
    df_labels = np.unique(list(y_train) + list(y_test))

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    onehot_encoder.fit(integer_encoded)
    onehot_encoded = onehot_encoder.transform(integer_encoded)

    train_integer_encoded = label_encoder.transform(y_train).reshape(-1, 1)
    train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
    Y_train = np.asarray(train_onehot_encoded)

    test_integer_encoded = label_encoder.transform(y_test).reshape(-1, 1)
    test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
    Y_test = np.asarray(test_onehot_encoded)
    Y_test_int = np.asarray(test_integer_encoded)

    n_classes = len(df_labels)
    print(n_classes)

    from keras.models import load_model
    model = load_model("models/bpi12_work_allmodel_smac_"+str(f)+"_batch.h5")
    model.summary()

    preds_a = model.predict([diff2_test, sequence_test, resource_test, qt_test])

    y_a_test = np.argmax(Y_test, axis=1)
    preds_a = np.argmax(preds_a, axis=1)

    precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                   pos_label=None)
    auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
    prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")


    print(classification_report(Y_test_int, preds_a, digits=3))
    outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
    outfile2.write('\nAUC: ' + str(auc_score_macro))
    outfile2.write('\nPRAUC: '+ str(prauc_score_macro))
    outfile2.write('\n')


    outfile2.flush()

outfile2.close()
