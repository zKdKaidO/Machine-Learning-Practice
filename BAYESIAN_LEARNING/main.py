from dataset import Dataset
from gaussian import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB as SklearnNB

if __name__ == "__main__":
    print("========== STEP 1: INPUT DATA ==========")
    dataset_path = r"E:\HK252\ML\CODE\Machine-Learning-Practice\BAYESIAN_LEARNING\salmon_seabass.csv"
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

    y_pred_1d = model_1d.predict(X_test_1d)
    acc_1d = accuracy_score(y_test, y_pred_1d)

    print(f"Accuracy (1 Feature - Chiều dài): {acc_1d * 100:.2f}%")

    print("========== EXERCISE 2: CLASSIFIER WITH 2 FEATURES ==========")
    # keep 2 features in csv
    model_2d = GaussianNB()
    model_2d.fit(X_train, y_train)
    
    y_pred_2d = model_2d.predict(X_test)
    acc_2d = accuracy_score(y_test, y_pred_2d)
    
    print(f"Accuracy (2 Features): {acc_2d * 100:.2f}%")

    print("========== EXERCISE 3: COMPARE WITH SKLEARN ==========")
    sklearn_model = SklearnNB()
    sklearn_model.fit(X_train, y_train)
    
    y_pred_sk = sklearn_model.predict(X_test)
    acc_sk = accuracy_score(y_test, y_pred_sk)
    
    print(f"Accuracy of Sklearn (2 Features):   {acc_sk * 100:.2f}%")
    print(f"My Accuracy (2 Features):      {acc_2d * 100:.2f}%")
    
    if acc_2d == acc_sk:
        print("They are the same!")
    else:
        print("There is a little bit different!")