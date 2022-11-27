import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from sklearn import preprocessing
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from core.utils import load_data, get_callbacks_list, set_gpu_limit, write_score, print_cmx, plot_model_history
from core.deep_model import deep_learning_model
from sklearn.model_selection import train_test_split


# # Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-m", "--memory", default=0, type=int, help="set gpu memory limit")
parser.add_argument("-v", "--version", default="version-0.2", help="version running")
parser.add_argument("-rp", "--result_path", default="../runs/results", help="path result ")
parser.add_argument("-tp", "--training_path", default="../runs/training", help="path training model")
parser.add_argument("-ep", "--epochs", default=1, type=int, help="epochs training")
parser.add_argument("-bsize", "--bath_size", default=8, type=int, help="bath size training")
parser.add_argument("-verbose", "--verbose", default=1, type=int, help="verbose training")
parser.add_argument("-train", "--train_data_path", default="../dataset/train/data-300x100-5-v1-train.data", help="data training")
parser.add_argument("-val", "--val_data_path", default="../dataset/train/data-valid.data", help="data val")
parser.add_argument("-test", "--test_data_path", default="../dataset/train/data-300x100-5-v1-test.data", help="data test")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
parser.add_argument("--mode_model", default="name-model", help="mobi-v2")
parser.add_argument("-activation_block", default="relu", help="optional relu")
parser.add_argument("-status_ckpt", default=True, type=bool, help="True or False")
parser.add_argument("-status_early_stop", default=True, type=bool, help="True or False")
parser.add_argument("-test_size", default=0.2, type=float, help="split data train")
args = vars(parser.parse_args())

# Set up parameters
test_size = args["test_size"]
status_ckpt = args["status_ckpt"]
status_early_stop = args["status_early_stop"]
activation_block = args["activation_block"]
version = args["version"]
training_path = args["training_path"]
result_path = args["result_path"]
epochs = args["epochs"]
bath_size = args["bath_size"]
verbose = args["verbose"]
gpu_memory = args["memory"]
train_path = args["train_data_path"]
val_path = args["val_data_path"]
test_path = args["test_data_path"]
model_name = args["name_model"]
mode_model = args["mode_model"]

print("=========Start=========")

print("activation_block : ", activation_block)

if gpu_memory > 0:
    set_gpu_limit(int(gpu_memory))  # set GPU

print("=====loading dataset ...======")
global_dataset_train, global_labels_train = load_data(train_path)
global_dataset_test, global_labels_test = load_data(test_path)

global_dataset_train = np.array(global_dataset_train, dtype="float32")
global_dataset_test = np.array(global_dataset_test, dtype="float32")
global_labels_train = np.array(global_labels_train)
global_labels_test = np.array(global_labels_test)

print("TRAIN : ", global_dataset_train.shape, " - ", global_labels_train.shape)
print("TEST : ", global_dataset_test.shape, " - ", global_labels_test.shape)

print("=======loading dataset done!!=======")
num_classes = len(np.unique(global_labels_train))
ip_shape = global_dataset_train[0].shape
metrics = [
    'categorical_accuracy'
]

model = deep_learning_model(input_shape=ip_shape, number_class=num_classes, activation_dense='softmax', activation_block=activation_block)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=metrics)
model.summary()
weights_init = model.get_weights()
print("model loading done!!")

# created folder

if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

training_path = os.path.join(training_path, model_name)
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

training_path = os.path.join(training_path, version)
if not os.path.exists(training_path):
    os.makedirs(training_path)
    print("created folder : ", training_path)

if not os.path.exists(os.path.join(training_path, 'model-save')):
    os.makedirs(os.path.join(training_path, 'model-save'))
    print("created folder : ", os.path.join(training_path, 'model-save'))


if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)


result_path = os.path.join(result_path, model_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)
    print("created folder : ", result_path)

# training

file_ckpt_model = "best-weights-training-file-" + model_name + "-" + version + ".h5"
# callback list
callbacks_list, save_list = get_callbacks_list(training_path,
                                               status_tensorboard=True,
                                               status_checkpoint=status_ckpt,
                                               status_earlystop=status_early_stop,
                                               file_ckpt=file_ckpt_model,
                                               ckpt_monitor='val_categorical_accuracy',
                                               ckpt_mode='max',
                                               early_stop_monitor="val_loss",
                                               early_stop_mode="min",
                                               early_stop_patience=20
                                               )
print("Callbacks List: ", callbacks_list)
print("Save List: ", save_list)

print("===========Training==============")
print("===Labels fit transform ===")
train_data, valid_data, train_labels, valid_labels = train_test_split(global_dataset_train, global_labels_train, test_size=test_size,
                                                                      random_state=1000, shuffle=True, stratify=global_labels_train)
lb = preprocessing.LabelBinarizer()
labels_train_one_hot = lb.fit_transform(train_labels)
labels_valid_one_hot = lb.fit_transform(valid_labels)
labels_test_one_hot = lb.fit_transform(global_labels_test)

print("TRAIN : ", train_data.shape, "-", labels_train_one_hot.shape)
print("VALID : ", valid_data.shape, "-", labels_valid_one_hot.shape)
print("TEST : ", global_dataset_test.shape, "-", labels_test_one_hot.shape)

model.set_weights(weights_init)
model_history = model.fit(train_data, labels_train_one_hot, epochs=epochs, batch_size=bath_size,
                          verbose=verbose, validation_data=(valid_data, labels_valid_one_hot),
                          shuffle=True, callbacks=callbacks_list)
print("===========Training Done !!==============")

model_save_file = "model-" + model_name + "-" + version + ".h5"
model.save(os.path.join(training_path, 'model-save', model_save_file), save_format='h5')
print("Save model done!!")

scores = model.evaluate(global_dataset_test, labels_test_one_hot, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print("testing model.....")
y_predict = model.predict(global_dataset_test)
y_true = np.argmax(labels_test_one_hot, axis=1)
y_target = np.argmax(y_predict, axis=1)

print("save results ......")
print_cmx(y_true=y_true, y_pred=y_target, save_path=result_path, version=version)

file_result = model_name + version + "score.txt"

write_score(path=os.path.join(result_path, file_result),
            mode_write="a",
            rows="STT",
            cols=['F1', 'Acc', "recall", "precision"])

write_score(path=os.path.join(result_path, file_result),
            mode_write="a",
            rows="results",
            cols=np.around([f1_score(y_true, y_target, average='weighted'),
                            accuracy_score(y_true, y_target),
                            recall_score(y_true, y_target, average='weighted'),
                            precision_score(y_true, y_target, average='weighted')], decimals=4))

# # load best check point
# model.load_weights(training_path + "/checkpt/" + file_ckpt_model)
# # evaluate the model
# scores = model.evaluate(X_test, y_test, verbose=verbose)

print("save results done!!")
print("History training loading ...")
cmd = 'tensorboard --logdir "path-tensorboard-logs/"'
print("CMD: ", cmd)
for file_log in save_list:
    print("file_log: ", file_log)
print("============END=============")
