import logging
import numpy as np
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from sklearn import preprocessing
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Input, concatenate, BatchNormalization, LSTM, Reshape, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from time import perf_counter
import pickle
from smac.facade.hyperband_facade import HB4AC

class MiDA:
    def __init__(self, eventlog):
        self._eventlog = eventlog
        self._cat_view = []
        self._num_view = []
        self._seq_length = 0
        self.list_cat_view_train = []
        self.list_num_view_train = []
        self.y_train = []
        self.y_test = []
        self.n_classes = 0
        self.n_fold = 0

    @staticmethod
    def Union(lst1, lst2):
        final_list = lst1 + lst2
        return final_list

    def load_col(self):
        with open("fold/" + self._eventlog + "/" + self._eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            self._num_view = pickle.load(pickle_file)
        with open("fold/" + self._eventlog + "/" + self._eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            self._cat_view = pickle.load(pickle_file)
        with open("fold/" + self._eventlog + "/" + self._eventlog + '_seq_length.pickle', 'rb') as pickle_file:
            self._seq_length = pickle.load(pickle_file)

    def get_model(self, cfg):
        list_cat_view = []
        list_num_view = []
        list_cat_view_in = []
        list_num_view_in = []
        for c in self._cat_view:
            num_view = np.load("fold/" + self._eventlog + "/" + self._eventlog + '_' + c + '_' + str(0) + "_info.npy")
            size_view = num_view + 1 // 2
            input_cat = Input(shape=(self._seq_length,), dtype='int32', name=c)
            list_cat_view_in.append(input_cat)
            x = Embedding(output_dim=size_view, input_dim=num_view + 1, input_length=self._seq_length)(input_cat)
            list_cat_view.append(x)

        for n in self._num_view:
            input_num = Input(shape=(self._seq_length,), dtype='float32', name=n)
            x = Reshape((self._seq_length, 1))(input_num)
            list_num_view_in.append(input_num)
            list_num_view.append(x)


        layer_in = concatenate((self.Union(list_cat_view, list_num_view)))

        layer_l = LSTM(units=int(cfg["lstmsize1"]), kernel_initializer='glorot_uniform',
                       return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(layer_in)
        layer_l = BatchNormalization()(layer_l)
        layer_l = LSTM(units=int(cfg["lstmsize2"]), kernel_initializer='glorot_uniform',
                       return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(layer_l)
        layer_l = BatchNormalization()(layer_l)

        out = Dense(self.n_classes, activation='softmax')(layer_l)
        opt = Nadam(learning_rate=cfg['learning_rate_init'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004,
                    clipvalue=3)
        model = Model(inputs=self.Union(list_cat_view_in, list_num_view_in), outputs=out)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
        model.summary()

        return model


    def fit_and_score(self, cfg):
        print(cfg)
        outfile2 = open(self._eventlog + '_' + str(self.n_fold) + ".txt", 'a')
        start_time = perf_counter()
        model = self.get_model(cfg)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0)

        list_view = self.Union(self.list_cat_view_train,self.list_num_view_train)

        h = model.fit(list_view,
            self.y_train, epochs=200, verbose=1, validation_split=0.2, callbacks=[early_stopping, lr_reducer],
            batch_size=cfg['batch_size'])

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
            best_model.save("models/"+self._eventlog+"model_smac_"+str(self.n_fold)+"_layer.h5")

        outfile2.write(str(score)+";"+str(len(h.history['loss']))+";"+str(model.count_params())+";"+str(end_time - start_time)+";"+ str(cfg['lstmsize1'])+";"+str(cfg['lstmsize2'])+";"+str(cfg['batch_size'])+";"+str(cfg['learning_rate_init'])+"\n")
        return score

    def smac_opt(self):
        self.load_col()
        for f in range(3):
            self.n_fold = f
            # model selection
            print('Starting model selection...')
            best_score = np.inf
            best_model = None
            best_time = 0
            best_numparameters = 0
            outfile_temp = open("best_temp.txt", 'w')
            outfile_temp.write(str(np.inf))
            outfile_temp.close()

            logger = logging.getLogger("bpic2020_fold_" + str(f))
            logging.basicConfig(level=logging.INFO)

            # Build Configuration Space which defines all parameters and their ranges.
            # To illustrate different parameter types,
            # we use continuous, integer and categorical parameters.
            cs = ConfigurationSpace()

            # We can add multiple hyperparameters at once:
            lstmsize1 = CategoricalHyperparameter("lstmsize1", [50, 75, 100])
            lstmsize2 = CategoricalHyperparameter("lstmsize2", [50, 75, 100])
            batch_size = CategoricalHyperparameter("batch_size", [32, 64, 128, 256, 512, 1024])
            learning_rate_init = UniformFloatHyperparameter('learning_rate_init', 0.00001, 0.01, default_value=0.001,
                                                            log=True)
            cs.add_hyperparameters([lstmsize1, lstmsize2, batch_size, learning_rate_init])

            # SMAC scenario object
            # Scenario object
            scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                                 "runcount-limit": 20,  # max. number of function evaluations;
                                 "cs": cs,  # configuration space
                                 "deterministic": "true",
                                 "abort_on_first_run_crash": "false",
                                 "output_dir": self._eventlog

                                 })

            self.list_cat_view_train = []
            for col in self._cat_view:
                self.list_cat_view_train.append(np.load("fold/"+self._eventlog+"/" + self._eventlog + "_" + col +"_" + str(f) + "_train.npy"))

            self.list_cat_view_test = []
            for col in self._cat_view:
                self.list_cat_view_test.append(np.load("fold/" + self._eventlog + "/" + self._eventlog + "_" + col + "_" + str(f) + "_test.npy"))

            self.list_num_view_train = []
            for col in self._num_view:
                self.list_num_view_train.append(np.load("fold/"+self._eventlog+"/" + self._eventlog + "_" + col +"_" + str(f) + "_train.npy", allow_pickle=True))

            self.list_num_view_test = []
            for col in self._num_view:
                self.list_num_view_test.append(np.load("fold/" + self._eventlog + "/" + self._eventlog + "_" + col + "_" + str(f) + "_test.npy", allow_pickle=True))


            y_train = np.load("fold/" + self._eventlog + "/" + self._eventlog + "_y_" + str(f) + "_train.npy")
            y_test = np.load("fold/" + self._eventlog + "/" + self._eventlog + "_y_" + str(f) + "_test.npy")

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

            self.y_train = Y_train
            self.y_test = Y_test

            self.n_classes = len(df_labels)

            max_iters = 200
            # print("Default Value: %.2f" % def_value)
            intensifier_kwargs = {'initial_budget': 20, 'max_budget': max_iters, 'eta': 3}
            # Optimize, using a SMAC-object
            print("Optimizing! Depending on your machine, this might take a few minutes.")
            smac = HB4AC(scenario=scenario,
                         rng=np.random.RandomState(42),
                         tae_runner=self.fit_and_score,
                         intensifier_kwargs=intensifier_kwargs
                         )

            # Start optimization
            try:
                incumbent = smac.optimize()
            finally:
                incumbent = smac.solver.incumbent
            inc_value = smac.get_tae_runner().run(config=incumbent, instance='1', budget=max_iters, seed=0)[1]
            print(inc_value)
            print("Optimized Value: %.4f" % inc_value)