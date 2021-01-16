import logging

import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from sklearn import preprocessing

from keras.layers.core import Dense
from keras.optimizers import Nadam
from keras.layers import Input, concatenate, BatchNormalization, LSTM, Reshape, Embedding, SpatialDropout1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from time import perf_counter
from smac.facade.hyperband_facade import HB4AC


def get_model(cfg):
    input_diff2 = Input(shape=(lenght_seq,), dtype='float32', name='input_diff2')
    diff2 = Reshape((lenght_seq, 1))(input_diff2)

    size_act = (num_act + 1) // 2
    print(size_act)
    input_act = Input(shape=(lenght_seq,), dtype='int32', name='input_act')
    x_act = Embedding(output_dim=size_act, input_dim=num_act+1, input_length=lenght_seq)(input_act)

    size_res = (num_res + 1) // 2
    print(size_res)
    input_res = Input(shape=(lenght_seq,), dtype='int32', name='input_res')
    x_res = Embedding(output_dim=size_res, input_dim=num_res+1, input_length=lenght_seq)(input_res)

    size_att1 = (num_att1 + 1) // 2
    input_att1 = Input(shape=(lenght_seq,), dtype='int32', name='input_att1')
    x_att1 = Embedding(output_dim=size_att1, input_dim=num_att1+1, input_length=lenght_seq)(input_att1)

    size_att2 = (num_att2 + 1) // 2
    input_att2 = Input(shape=(lenght_seq,), dtype='int32', name='input_att2')
    x_att2 = Embedding(output_dim=size_att2, input_dim=num_att2+1, input_length=lenght_seq)(input_att2)

    size_att3 = (num_att3 + 1) // 2
    input_att3 = Input(shape=(lenght_seq,), dtype='int32', name='input_att3')
    x_att3 = Embedding(output_dim=size_att3, input_dim=num_att3+1, input_length=lenght_seq)(input_att3)

    size_att4 = (num_att4 + 1) // 2
    input_att4 = Input(shape=(lenght_seq,), dtype='int32', name='input_att4')
    x_att4 = Embedding(output_dim=size_att4, input_dim=num_att4+1, input_length=lenght_seq)(input_att4)

    size_att5 = (num_att5 + 1) // 2
    input_att5 = Input(shape=(lenght_seq,), dtype='int32', name='input_att5')
    x_att5 = Embedding(output_dim=size_att5, input_dim=num_att5+1, input_length=lenght_seq)(input_att5)

    size_att6 = (num_att6 + 1) // 2
    input_att6 = Input(shape=(lenght_seq,), dtype='int32', name='input_att6')
    x_att6 = Embedding(output_dim=size_att6, input_dim=num_att6+1, input_length=lenght_seq)(input_att6)

    layer_in = concatenate(([diff2, x_act, x_res, x_att1, x_att2, x_att3, x_att4, x_att5, x_att6]))

    layer_l = LSTM(units=int(cfg["lstmsize1"]), implementation=2, kernel_initializer='glorot_uniform',
                   return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(layer_in)
    layer_l = BatchNormalization()(layer_l)
    layer_l = LSTM(units=int(cfg["lstmsize2"]), implementation=2, kernel_initializer='glorot_uniform',
                   return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(layer_l)
    layer_l = BatchNormalization()(layer_l)

    out = Dense(n_classes, activation='softmax')(layer_l)
    opt = Nadam(lr=cfg['learning_rate_init'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004,
                clipvalue=3)
    model = Model(inputs=[input_diff2, input_act, input_res, input_att1, input_att2, input_att3, input_att4, input_att5, input_att6], outputs=out)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model


def fit_and_score(cfg):
    print(cfg)
    outfile2 = open("smac_bpi13_problems_AR" + str(f) + ".txt", 'a')
    start_time = perf_counter()
    model = get_model(cfg)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    h = model.fit([diff2_train, sequence_train, resource_train, att1_train, att2_train, att3_train, att4_train, att5_train, att6_train], Y_train, epochs=200, verbose=0, validation_split=0.2, callbacks=[early_stopping, lr_reducer],batch_size=cfg['batch_size'])


    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    end_time = perf_counter()
    global best_score, best_model, best_time, best_numparameters
    ff = open("best_temp.txt", "r")
    best_score = float(ff.read())
    print("best_score->", best_score)
    print("score->", score)
    if best_score > score:
        best_score = score
        best_model = model
        outfile_temp = open("best_temp.txt", 'w')
        outfile_temp.write(str(best_score))
        outfile_temp.close()
        best_numparameters = model.count_params()
        best_time = end_time - start_time
        print("BEST SCORE", best_score)
        best_model.save("models/bpi13_problemsmodel_smac_"+str(f)+"_batch.h5")

    outfile2.write(str(score)+";"+str(len(h.history['loss']))+";"+str(model.count_params())+";"+str(end_time - start_time)+";"+ str(cfg['lstmsize1'])+";"+str(cfg['lstmsize2'])+";"+str(cfg['batch_size'])+";"+str(cfg['learning_rate_init'])+"\n")
    return score#{'loss': score, 'n_epochs': len(h.history['loss']), 'n_params': model.count_params(), 'time': end_time - start_time}


benchmark = 'bpi13_problems'

info_dataset = np.load("fold/bpi13_problems/" + str(benchmark) + "_info_dataset.npy")

lenght_seq = info_dataset[0]
num_act = info_dataset[1]
num_res = info_dataset[2]
num_att1 = info_dataset[3]
num_att2 = info_dataset[4]
num_att3 = info_dataset[5]
num_att4 = info_dataset[6]
num_att5 = info_dataset[7]
num_att6 = info_dataset[8]

for f in range(3):

    # model selection
    print('Starting model selection...')
    best_score = np.inf
    best_model = None
    best_time = 0
    best_numparameters = 0
    outfile_temp = open("best_temp.txt", 'w')
    outfile_temp.write(str(np.inf))
    outfile_temp.close()

    logger = logging.getLogger("bpi13_problems_fold_AR" + str(f))
    logging.basicConfig(level=logging.INFO)

    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    # We can add multiple hyperparameters at once:
    lstmsize1 = CategoricalHyperparameter("lstmsize1", [50, 75, 100], default_value=50)
    lstmsize2 = CategoricalHyperparameter("lstmsize2", [50, 75, 100], default_value=50)
    batch_size = CategoricalHyperparameter("batch_size", [32, 64], default_value=32)
    learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 0.00001, 0.01, default_value=0.001, log=True)
    cs.add_hyperparameters([lstmsize1, lstmsize2, batch_size, learning_rate_init])

    # SMAC scenario object
    # Scenario object
    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 10,  # max. number of function evaluations;
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         "abort_on_first_run_crash": "false"
                         })

    # train views
    print("FOLD numero----------------------->", f)
    print("<--------------------------------->")
    sequence_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_act_" + str(f) + "_train.npy")
    resource_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_res_" + str(f) + "_train.npy")
    diff2_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_diff2_" + str(f) + "_train.npy")
    att1_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_att1_" + str(f) + "_train.npy")
    att2_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_att2_" + str(f) + "_train.npy")
    att3_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_att3_" + str(f) + "_train.npy")
    att4_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_att4_" + str(f) + "_train.npy")
    att5_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_att5_" + str(f) + "_train.npy")
    att6_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_att6_" + str(f) + "_train.npy")
    y_train = np.load("fold/bpi13_problems/" + str(benchmark) + "_y_" + str(f) + "_train.npy")

    # test views
    sequence_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_act_" + str(f) + "_test.npy")
    resource_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_res_" + str(f) + "_test.npy")
    diff2_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_diff2_" + str(f) + "_test.npy")
    att1_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_att1_" + str(f) + "_test.npy")
    att2_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_att2_" + str(f) + "_test.npy")
    att3_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_att3_" + str(f) + "_test.npy")
    att4_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_att4_" + str(f) + "_test.npy")
    att5_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_att5_" + str(f) + "_test.npy")
    att6_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_att6_" + str(f) + "_test.npy")
    y_test = np.load("fold/bpi13_problems/" + str(benchmark) + "_y_" + str(f) + "_test.npy")

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

    max_iters = 200
    # print("Default Value: %.2f" % def_value)
    intensifier_kwargs = {'initial_budget': 20, 'max_budget': max_iters, 'eta': 3}
    # Optimize, using a SMAC-object
    print("Optimizing! Depending on your machine, this might take a few minutes.")
    smac = HB4AC(scenario=scenario,
                 rng=np.random.RandomState(42),
                 tae_runner=fit_and_score,
                 intensifier_kwargs=intensifier_kwargs
                 )
    incumbent = smac.optimize()
    '''
    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent
    inc_value = smac.get_tae_runner().run(config=incumbent, instance='1',budget=max_iters, seed=0)[1]
    print(inc_value)
    print("Optimized Value: %.4f" % inc_value)
    '''
