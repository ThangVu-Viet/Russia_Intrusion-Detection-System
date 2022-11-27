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


# # Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--version", default="version-0.2", help="version running")
parser.add_argument("-rp", "--result_path", default="../runs/results", help="path result")
parser.add_argument("-tp", "--training_path", default="../runs/training", help="path training model")
parser.add_argument("-train", "--train_data_path", default="../dataset/train/data-300x100-5-v1-train.data", help="data training")
parser.add_argument("-val", "--val_data_path", default="../dataset/train/data-valid.data", help="data val")
parser.add_argument("-test", "--test_data_path", default="../dataset/train/data-300x100-5-v1-test.data", help="data test")
parser.add_argument("-name", "--name_model", default="model_ai_name", help="model name")
args = vars(parser.parse_args())

# Set up parameters
version = args["version"]
training_path = args["training_path"]
result_path = args["result_path"]
train_path = args["train_data_path"]
val_path = args["val_data_path"]
test_path = args["test_data_path"]
model_name = args["name_model"]

print("=========Start=========")
print("=====loading dataset ...======")
global_dataset_train, global_labels_train = load_data(train_path)
global_dataset_test, global_labels_test = load_data(test_path)

global_dataset_train = np.array(global_dataset_train, dtype="float32").T[0][0].T
global_dataset_test = np.array(global_dataset_test, dtype="float32").T[0][0].T
global_labels_train = np.array(global_labels_train)
global_labels_test = np.array(global_labels_test)

print("TRAIN : ", global_dataset_train.shape, " - ", global_labels_train.shape)
print("TEST : ", global_dataset_test.shape, " - ", global_labels_test.shape)

print("=======loading dataset done!!=======")

print("model loading ...")
model = dict_machine_model(model_name)
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
print("===========Training ...==============")
model = model.fit(global_dataset_train, global_labels_train)

# save the model to disk
print("===========Save model ...==============")
file_save = os.path.join(os.path.join(training_path, 'model-save'), f'{model_name}-{version}.sav')
pickle.dump(model, open(file_save, 'wb'))
print("===========Save model done==============")
print("===========Training Done !!==============")


print("testing model.....")
y_predict = model.predict(global_dataset_test)

print("save results ......")

print_cmx(y_true=global_labels_test, y_pred=y_predict, save_path=result_path, version=version)

file_result = model_name + version + "score.txt"

write_score(path=os.path.join(result_path, file_result),
            mode_write="a",
            rows="STT",
            cols=['F1', 'Acc', "recall", "precision"])

write_score(path=os.path.join(result_path, file_result),
            mode_write="a",
            rows="results",
            cols=np.around([f1_score(global_labels_test, y_predict, average='weighted'),
                            accuracy_score(global_labels_test, y_predict),
                            recall_score(global_labels_test, y_predict, average='weighted'),
                            precision_score(global_labels_test, y_predict, average='weighted')], decimals=4))

print("save results done!!")
print("============END=============")
