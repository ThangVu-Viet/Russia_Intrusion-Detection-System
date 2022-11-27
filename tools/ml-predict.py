import os
import pickle
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from core.utils import load_data, write_score, print_cmx
from core.machine_model import dict_machine_model

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_result", default="../runs", help="path save data")
parser.add_argument("--test_path", default="../public-test.data", help="path data image")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
parser.add_argument("--model_path", default="../model.h5", help="model path")
parser.add_argument("-v", "--version", default="0.1", help="version running")

args = vars(parser.parse_args())
test_path = args["test_path"]
version = args["version"]
model_path = args["model_path"]
model_name = args["name_model"]
save_result_path = args["save_result"]

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
global_dataset_test = np.array(global_dataset_test, dtype="float32").T[0][0].T
global_labels_test = np.array(global_labels_test)

print("TEST : ", global_dataset_test.shape, " - ", global_labels_test.shape)
print("loading data test done")

print("loading model ...")

model = pickle.load(open(model_path, 'rb'))
print("testing model.....")
y_predict = model.predict(global_dataset_test)

print("save results ......")

print_cmx(y_true=global_labels_test, y_pred=y_predict, save_path=save_result_path, version=version)

file_result = model_name + version + "score.txt"

write_score(path=os.path.join(save_result_path, file_result),
            mode_write="a",
            rows="STT",
            cols=['F1', 'Acc', "recall", "precision"])

write_score(path=os.path.join(save_result_path, file_result),
            mode_write="a",
            rows="results",
            cols=np.around([f1_score(global_labels_test, y_predict, average='weighted'),
                            accuracy_score(global_labels_test, y_predict),
                            recall_score(global_labels_test, y_predict, average='weighted'),
                            precision_score(global_labels_test, y_predict, average='weighted')], decimals=4))
print("Save results done!!")
print("==== END =======")