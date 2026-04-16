import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from DATASET.dataset import Dataset
from gaussian import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB as SklearnNB

if __name__ == "__main__":
    print("========== STEP 1: INPUT DATA ==========")
    dataset_path = r"D:\AI\Machine-Learning-Practice\BAYESIAN_LEARNING\salmon_seabass.csv"
    df = Dataset(dataset_path)
    X, y = df.get_data()

    X_train, X_test, y_train, y_test = df.split_data()
    print(f"Num. Train: {X_train.shape[0]}")
    print(f"Num. Test: {X_test.shape[0]}\n")

    print("========== EXERCISE 1: CLASSIFIER WITH 1 DIMENSION ==========")
    X_train_1d = X_train[:, 0:1] # diff with [:, 0] -> 1D (N,); [:, 0:1] -> 2D (N,1)
    X_test_1d = X_test[:, 0:1]

    model_1d = GaussianNB()
    model_1d.fit(X_train_1d, y_train)

    y_pred_1d_map = model_1d.predict(X_test_1d)
    y_pred_1d_mle = model_1d.predict(X_test_1d, "MLE")
    acc_1d_map = accuracy_score(y_test, y_pred_1d_map)
    acc_1d_mle = accuracy_score(y_test, y_pred_1d_mle)

    print(f"Accuracy (1 Feature): {acc_1d_map * 100:.2f}% (MAP) and {acc_1d_mle * 100:.2f}% (MLE)")

    print("========== EXERCISE 2: CLASSIFIER WITH 2 FEATURES ==========")
    # keep 2 features in csv
    model_2d = GaussianNB()
    model_2d.fit(X_train, y_train)
    
    y_pred_2d_map = model_2d.predict(X_test)
    acc_2d_map = accuracy_score(y_test, y_pred_2d_map)

    y_pred_2d_mle = model_2d.predict(X_test, "MLE")
    acc_2d_mle = accuracy_score(y_test, y_pred_2d_mle)
    
    print(f"Accuracy (2 Features): {acc_2d_map * 100:.2f}% (MAP) and {acc_2d_mle * 100:.2f}% (MLE)")

    print("========== EXERCISE 3: COMPARE WITH SKLEARN ==========")
    sklearn_model = SklearnNB()
    sklearn_model.fit(X_train, y_train)
    
    y_pred_sk = sklearn_model.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    
    print(f"Accuracy of Sklearn (2 Features):   {acc_sk * 100:.2f}%")
    print(f"My Accuracy (2 Features): {acc_2d_map * 100:.2f}% (MAP) and {acc_2d_mle * 100:.2f}% (MLE)")
    
    if acc_2d_map == acc_sk:
        print("They are the same!")
    else:
        print("There is a little bit different!")