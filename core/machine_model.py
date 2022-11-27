from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def dict_machine_model(model_name='tree'):
    dict_model = {
        'tree': DecisionTreeClassifier(random_state=100),
        'svm-rbf': SVC(kernel='rbf', gamma='scale'),
        'svm-linear': SVC(kernel='linear', gamma='scale'),
    }
    return dict_model[model_name]
