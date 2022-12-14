import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score

import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.utils import set_gpu_limit, load_data, save_dump, write_score, get_callbacks_list
from core.deep_model import deep_learning_model


# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--memory", default=0, type=int, help="set gpu memory limit")
parser.add_argument("-v", "--version", default="version-0.0", help="version running")
parser.add_argument("-kp", "--k_fold_path", default="./runs/k-fold", help="path log k fold running")
parser.add_argument("-rp", "--result_path", default="./runs/results", help="path result ")
parser.add_argument("-nk", "--number_k_fold", default=2, type=int, help="number k-fold")
parser.add_argument("-ck", "--continue_k_fold", default=1, type=int, help="continue k-fold")
parser.add_argument("-train", "--train_data_path", default="./data_train.data", help="data train")
parser.add_argument("-test", "--test_data_path", default="./data_test.data", help="data test")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
parser.add_argument("-ep", "--epochs", default=1, type=int, help="epochs training")
parser.add_argument("-bsize", "--bath_size", default=32, type=int, help="bath size training")
parser.add_argument("-verbose", "--verbose", default=1, type=int, help="verbose training")
parser.add_argument("--mode_model", default="name-model", help="mobi-v2")
parser.add_argument("--mode_split_data", default=False, help="is split data ? (False or True)")
parser.add_argument("-activation_block", default="relu", help="optional relu")
parser.add_argument("-status_ckpt", default=True, help="True or False")
parser.add_argument("-status_early_stop", default=True, help="True or False")
args = vars(parser.parse_args())

# Set up paramet
activation_block = args["activation_block"]
status_ckpt = args["status_ckpt"]
status_early_stop = args["status_early_stop"]
mode_model = args["mode_model"]
epochs = args["epochs"]
bath_size = args["bath_size"]
verbose = args["verbose"]
model_name = args["name_model"]
version = args["version"]
result_path = args["result_path"]
k_fold_path = args["k_fold_path"]
gpu_memory = args["memory"]
number_k_fold = args["number_k_fold"]
continue_k_fold = args["continue_k_fold"]
train_path = args["train_data_path"]
test_path = args["test_data_path"]
mode_split_data = args["mode_split_data"]

print("=======START=======")
if gpu_memory > 0:
    set_gpu_limit(int(gpu_memory))  # set GPU

global_dataset_train, global_labels_train = load_data(train_path)
global_dataset_test, global_labels_test = load_data(test_path)
X = np.concatenate((global_dataset_train, global_dataset_test), axis=0)
y_labels_name = np.concatenate((global_labels_train, global_labels_test), axis=0)

print("===Labels fit transform ===")
lb = preprocessing.LabelBinarizer()
y_label_one_hot = lb.fit_transform(y_labels_name)
y = np.argmax(y_label_one_hot, axis=1)

print(X.shape, y.shape)
num_classes = len(np.unique(y))
ip_shape = X[0].shape

metrics = [
    'categorical_accuracy'
]

model = deep_learning_model(input_shape=ip_shape, number_class=num_classes, activation_dense='softmax', activation_block=activation_block)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=metrics)
weights_init = model.get_weights()
model.summary()

# created folder
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)

result_path = os.path.join(result_path, model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)

if not os.path.exists(k_fold_path):
    os.makedirs(k_fold_path)
    print("created folder : ", k_fold_path)

k_fold_path = os.path.join(k_fold_path, model_name)
if not os.path.exists(k_fold_path):
    os.makedirs(k_fold_path)
    print("created folder : ", k_fold_path)

k_fold_path = os.path.join(k_fold_path, version)
if not os.path.exists(k_fold_path):
    os.makedirs(k_fold_path)
    print("created folder : ", k_fold_path)

if not mode_split_data:
    # created data k fold
    if number_k_fold > 0 and continue_k_fold == 1:
        cnt_k_fold = 1
        k_fold_split = StratifiedKFold(n_splits=number_k_fold, shuffle=True, random_state=10000)
        for train, test in k_fold_split.split(X, y):
            print(X[train].shape, type(X[train]), y[test].shape, type(y[test]))
            file_k_fold_train = "k-fold-" + str(cnt_k_fold) + "-train-dataset.data"
            file_k_fold_test = "k-fold-" + str(cnt_k_fold) + "-test-dataset.data"
            save_dump(os.path.join(k_fold_path, file_k_fold_train), X[train], y[train])
            save_dump(os.path.join(k_fold_path, file_k_fold_test), X[test], y[test])
            print("created file : ", os.path.join(k_fold_path, file_k_fold_train))
            print("created file : ", os.path.join(k_fold_path, file_k_fold_test))
            cnt_k_fold += 1

fold_dict = {}
pred_folds_list = []
cv_scores = []
file_result_k_fold = model_name + "-" + version + "-k-fold-results.txt"
for cnt_k_fold in range(continue_k_fold, number_k_fold + 1):

    if cnt_k_fold == 1:
        write_score(path=os.path.join(result_path, file_result_k_fold),
                    rows="STT", cols=['F1', 'Acc', "recall", "precision"])
    roc_name = "roc-" + str(cnt_k_fold)
    folder_roc_cnt_k_fold = os.path.join(k_fold_path, roc_name)

    file_ckpt_model = model_name + "-" + version + "-weights-best-k-fold-" + str(cnt_k_fold) + ".h5"
    print("file check point : ", file_ckpt_model)
    flag_checkpoint = status_ckpt
    # callback list
    callbacks_list, save_list = get_callbacks_list(folder_roc_cnt_k_fold,
                                                   status_tensorboard=True,
                                                   status_checkpoint=flag_checkpoint,
                                                   status_earlystop=status_early_stop,
                                                   file_ckpt=file_ckpt_model,
                                                   ckpt_monitor='val_binary_accuracy',
                                                   ckpt_mode='max',
                                                   early_stop_monitor="val_loss",
                                                   early_stop_mode="min",
                                                   early_stop_patience=10
                                                   )

    file_k_fold_train = "k-fold-" + str(cnt_k_fold) + "-train-dataset.data"
    file_k_fold_test = "k-fold-" + str(cnt_k_fold) + "-test-dataset.data"
    X_train, y_train = load_data(path_file=os.path.join(k_fold_path, file_k_fold_train))
    X_test, y_test = load_data(path_file=os.path.join(k_fold_path, file_k_fold_test))

    lb = preprocessing.LabelBinarizer()
    labels_train_one_hot = lb.fit_transform(y_train)
    labels_test_one_hot = lb.fit_transform(y_test)

    print("TRAIN : ", X_train.shape, " - ", labels_train_one_hot.shape)
    print("TRAIN : ", X_test.shape, " - ", labels_test_one_hot.shape)
    # train model
    print("K-Fold =", cnt_k_fold)
    print("=============Training===========")
    model.set_weights(weights_init)
    model.fit(X_train, labels_train_one_hot, epochs=epochs, batch_size=bath_size, verbose=verbose,
              shuffle=True, callbacks=callbacks_list, validation_data=(X_test, labels_test_one_hot))
    print("=============Training Done !!!===========")

    # save model
    print("Save model ....")
    save_model_path = os.path.join(k_fold_path, "model-save-roc-" + str(cnt_k_fold))
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        print("created folder : ", save_model_path)
    model_save_file = "model-" + model_name + "-" + version + "-roc" + str(cnt_k_fold) + ".h5"
    model.save(os.path.join(save_model_path, model_save_file), save_format='h5')
    print("Save model done!!")

    # load model
    if flag_checkpoint:
        model.load_weights(folder_roc_cnt_k_fold + "/checkpt/" + file_ckpt_model)
    # evaluate the model
    scores = model.evaluate(X_test, labels_test_one_hot, verbose=verbose)

    # predict the model
    y_predict = model.predict(X_test)
    y_target = []
    for score in y_predict:
        if score > 0.5:
            y_target.append(1)
        else:
            y_target.append(0)

    write_score(path=os.path.join(result_path, file_result_k_fold),
                mode_write="a",
                rows="K=" + str(cnt_k_fold),
                cols=np.around([f1_score(y_test, y_target, average='weighted'),
                                accuracy_score(y_test, y_target),
                                recall_score(y_test, y_target, average='weighted'),
                                precision_score(y_test, y_target, average='weighted')], decimals=4))

    print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cv_scores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))