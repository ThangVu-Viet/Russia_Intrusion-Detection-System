import os
import sys
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from keras.models import load_model
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.utils import load_data
from core.utils import write_score, print_cmx
from core.deep_model import deep_learning_model, seg_relu

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_result", default="../runs", help="path save data")
parser.add_argument("--test_path", default="../public-test.data", help="path data image")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
parser.add_argument("--model_path", default="../model.h5", help="model path")
parser.add_argument("--best_ckpt_path", default="../best-ckpt.h5", help="model check point path")
parser.add_argument("-v", "--version", default="0.1", help="version running")
parser.add_argument("--mode_weight", default="check-point", help="check-point or model-save")
parser.add_argument("--mode_model", default="name-model", help="mobi-v2")
parser.add_argument("--custom_objects", default=False, help="True or False")
parser.add_argument("-activation_block", default="relu", help="optional relu")

args = vars(parser.parse_args())
activation_block = args["activation_block"]
test_path = args["test_path"]
version = args["version"]
model_path = args["model_path"]
best_ckpt_path = args["best_ckpt_path"]
model_name = args["name_model"]
save_result_path = args["save_result"]
mode_model = args["mode_model"]
mode_weight = args["mode_weight"]
custom_objects = args["custom_objects"]

print("==== START =======")
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

save_result_path = os.path.join(save_result_path, "results")
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

save_result_path = os.path.join(save_result_path, version)
if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("created folder :", save_result_path)

print("=====loading dataset ...======")
global_dataset_test, global_labels_test = load_data(test_path)
global_dataset_test = np.array(global_dataset_test, dtype="float32")
global_labels_test = np.array(global_labels_test)

print("TEST : ", global_dataset_test.shape, " - ", global_labels_test.shape)
print("loading data test done")

print("loading model ...")
metrics = [
    'categorical_accuracy'
]

print("loading model ...")
num_classes = len(np.unique(global_labels_test))
ip_shape = global_dataset_test[0].shape
model = None
if mode_weight == 'check-point':
    print("loading weight model ...")
    model = deep_learning_model(input_shape=ip_shape, number_class=num_classes, activation_dense='softmax',
                                activation_block=activation_block)
    model.load_weights(best_ckpt_path)
    print("loading weight model done!!")
elif mode_weight == 'model-save':
    # model = deep_learning_model(input_shape=ip_shape, number_class=num_classes, activation_dense='softmax',
    #                             activation_block=activation_block)
    model = load_model(model_path, compile=False, custom_objects={"seg_relu": seg_relu})
print("loading model done")
model.summary()
lb = preprocessing.LabelBinarizer()
labels_test_one_hot = lb.fit_transform(global_labels_test)

print("model predict ...")

y_predict = model.predict(global_dataset_test)
y_true = np.argmax(labels_test_one_hot, axis=1)
y_target = np.argmax(y_predict, axis=1)

print("save results ......")
print_cmx(y_true=y_true, y_pred=y_target, save_path=save_result_path, version=model_name + version)

file_result = model_name + version + "score.txt"

write_score(path=os.path.join(save_result_path, file_result),
            mode_write="a",
            rows="STT",
            cols=['F1', 'Acc', "recall", "precision"])

write_score(path=os.path.join(save_result_path, file_result),
            mode_write="a",
            rows="results",
            cols=np.around([f1_score(y_true, y_target, average='weighted'),
                            accuracy_score(y_true, y_target),
                            recall_score(y_true, y_target, average='weighted'),
                            precision_score(y_true, y_target, average='weighted')], decimals=4))


print("Save results done!!")
print("==== END =======")