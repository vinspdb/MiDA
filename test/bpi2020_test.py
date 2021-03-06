import numpy as np

seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed

set_random_seed(seed)
from keras.layers import Embedding, Dense, Reshape, BatchNormalization, SpatialDropout1D, Dropout
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from keras.layers import concatenate, Input, LSTM
import numpy as np
from sklearn import preprocessing
import os
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

benchmark = "bpic2020"
info_dataset = np.load("fold/bpic2020/" + str(benchmark) + "_info_dataset.npy")

lenght_seq = info_dataset[0]
num_act = info_dataset[1]
num_res = info_dataset[2]
num_org= info_dataset[3]
num_project = info_dataset[4]
num_task = info_dataset[5]
num_role = info_dataset[6]

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
    # train views
    outfile2 = open("bpic2020model_verifica.txt", 'w')

    print("FOLD numero----------------------->", f)
    print("<--------------------------------->")
    sequence_train = np.load("fold/bpic2020/" + str(benchmark) + "_act_" + str(f) + "_train.npy")
    resource_train = np.load("fold/bpic2020/" + str(benchmark) + "_res_" + str(f) + "_train.npy")
    diff2_train = np.load("fold/bpic2020/" + str(benchmark) + "_diff2_" + str(f) + "_train.npy")
    org_train = np.load("fold/bpic2020/" + str(benchmark) + "_org_" + str(f) + "_train.npy")
    project_train = np.load("fold/bpic2020/" + str(benchmark) + "_Project_" + str(f) + "_train.npy")
    task_train = np.load("fold/bpic2020/" + str(benchmark) + "_Task_" + str(f) + "_train.npy")
    role_train = np.load("fold/bpic2020/" + str(benchmark) + "_Role_" + str(f) + "_train.npy")
    y_train = np.load("fold/bpic2020/" + str(benchmark) + "_y_" + str(f) + "_train.npy")

    # test views
    sequence_test = np.load("fold/bpic2020/" + str(benchmark) + "_act_" + str(f) + "_test.npy")
    resource_test = np.load("fold/bpic2020/" + str(benchmark) + "_res_" + str(f) + "_test.npy")
    diff2_test = np.load("fold/bpic2020/" + str(benchmark) + "_diff2_" + str(f) + "_test.npy")
    org_test = np.load("fold/bpic2020/" + str(benchmark) + "_org_" + str(f) + "_test.npy")
    project_test = np.load("fold/bpic2020/" + str(benchmark) + "_Project_" + str(f) + "_test.npy")
    task_test = np.load("fold/bpic2020/" + str(benchmark) + "_Task_" + str(f) + "_test.npy")
    role_test = np.load("fold/bpic2020/" + str(benchmark) + "_Role_" + str(f) + "_test.npy")
    y_test = np.load("fold/bpic2020/" + str(benchmark) + "_y_" + str(f) + "_test.npy")
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
    model = load_model("models/pretrainedbpic2020model_smac_"+str(f)+".h5")
    preds_a = model.predict([sequence_test, resource_test, diff2_test, org_test, project_test, task_test, role_test])

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
