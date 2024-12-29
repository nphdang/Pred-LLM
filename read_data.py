import numpy as np
import pandas as pd
from sklearn import datasets
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def gen_train_test_data(dataset="", train_size=1.0, test_size=0.2, normalize_x=True, seed=42):
    print("dataset: {}, seed: {}".format(dataset, seed))
    print("train_size: {}, test_size: {}".format(train_size, test_size))
    # classification
    if dataset == "iris":
        ds = datasets.load_iris(as_frame=True)
        X = ds.data
        y = ds.target
        names = ds.feature_names
    if dataset == "breast_cancer":
        ds = datasets.load_breast_cancer(as_frame=True)
        X = ds.data
        y = ds.target
        names = ds.feature_names
    if dataset == "blood_transfusion":
        dataset_id = 1464
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "steel_plates_fault":
        dataset_id = 1504
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "qsar_biodeg":
        dataset_id = 1494
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "phoneme":
        dataset_id = 1489
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "diabetes":
        dataset_id = 37
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "tested_negative"] = "0"
        y_new[y_new == "tested_positive"] = "1"
        y = y_new
    if dataset == "kc1":
        dataset_id = 1067
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([str(val) for val in y.values])
        y_new[y_new == "False"] = "0"
        y_new[y_new == "True"] = "1"
        y = y_new
    if dataset == "kc2":
        dataset_id = 1063
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "no"] = "0"
        y_new[y_new == "yes"] = "1"
        y = y_new
    if dataset == "australian":
        dataset_id = 40981
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "churn":
        dataset_id = 40701
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "balance_scale":
        dataset_id = 11
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
        y_new = np.array([val for val in y.values])
        y_new[y_new == "L"] = "0"
        y_new[y_new == "B"] = "1"
        y_new[y_new == "R"] = "2"
        y = y_new
    if dataset == "cardiotocography":
        dataset_id = 1466
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "waveform":
        dataset_id = 60
        ds = openml.datasets.get_dataset(dataset_id)
        (X, y, categorical, names) = ds.get_data(target=ds.default_target_attribute)
    if dataset == "adult":
        df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
        numerical_var_names = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        categorical_var_names = ["race", "sex", "workclass", "marital-status", "occupation", "relationship"]
        class_var_name = "income-per-year"
    if dataset == "compas":
        df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
        numerical_var_names = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
        categorical_var_names = ["race", "sex", "c_charge_degree"]
        class_var_name = "two_year_recid"
    if dataset == "bank":
        df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
        numerical_var_names = ["balance", "duration", "campaign", "pdays", "previous"]
        categorical_var_names = ["age", "marital", "job", "education", "default", "housing", "loan", "contact", "poutcome"]
        class_var_name = "subscribe"
    # regression
    if dataset == "california":
        ds = datasets.fetch_california_housing(as_frame=True)
        X = ds.data
        y = ds.target
        names = ds.feature_names
    if dataset == "abalone":
        df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
        numerical_var_names = ["Length", "Diameter", "Height",
                               "Shucked_Weight", "Viscera_Weight", "Shell_Weight", "Rings"]
        categorical_var_names = ["Sex"]
        class_var_name = "Age"
    if dataset == "fuel":
        df = pd.read_csv("./data/{}.csv".format(dataset), header=0, sep=",")
        numerical_var_names = ["ENGINE_SIZE", "CYLINDERS", "COEMISSIONS"]
        categorical_var_names = ["MAKE", "MODEL", "VEHICLE_CLASS", "TRANSMISSION", "FUEL"]
        class_var_name = "CONSUMPTION"

    if dataset == "adult" or dataset == "compas" or dataset == "bank" or dataset == "abalone" or dataset == "fuel":
        X = np.array(df.loc[:, numerical_var_names])
        # convert categorical values to categorical indices
        for categorical_var_name in categorical_var_names:
            categorical_var = np.array(pd.Categorical(df.loc[:, categorical_var_name]))
            categorical_val = np.unique(categorical_var)
            categorical_dict = [{'value': c, 'index': i} for i, c in enumerate(categorical_val)]
            categorical_names = [categorical_dict[idx]['value'] for idx in range(len(categorical_val))]
            categorical_indices = [categorical_dict[idx]['index'] for idx in range(len(categorical_val))]
            for idx in range(len(categorical_val)):
                categorical_var[categorical_var == categorical_names[idx]] = categorical_indices[idx]
            X = np.append(X, categorical_var.reshape(-1, 1), axis=1)
        names = np.append(numerical_var_names, categorical_var_names)
    if dataset == "adult" or dataset == "compas" or dataset == "bank":
        # get label
        labels = pd.Categorical(df.loc[:, class_var_name])
        y = np.copy(labels.codes)
    if dataset == "abalone" or dataset == "fuel":
        # get target
        y = df.loc[:, class_var_name]

    # normalize X
    if normalize_x == True:
        print("normalize X")
        X = MinMaxScaler().fit_transform(X)
        X = np.around(X, 2)
    # don't normalize X
    if normalize_x == False:
        print("don't normalize X")
    # regression task
    if dataset == "abalone" or dataset == "fuel" or dataset == "california":
        # convert y to float array
        y = np.array([float(val) for val in y])
        y = y.reshape(-1, 1)
        # create test set
        X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_size,
                                                              shuffle=True, random_state=seed)
        # create training set
        if train_size == 1:
            X_train = X_train_
            y_train = y_train_
        else:
            X_train, _, y_train, _ = train_test_split(X_train_, y_train_, test_size=(1.0 - train_size),
                                                      shuffle=True, random_state=seed)
    else: # classification task
        # convert y to integer array
        y = np.array([int(val) for val in y])
        # convert y to label indices
        y_values = np.unique(y)
        y_dict = [{'value': y, 'index': i} for i, y in enumerate(y_values)]
        y_names = [y_dict[idx]['value'] for idx in range(len(y_values))]
        y_indices = [y_dict[idx]['index'] for idx in range(len(y_values))]
        for idx in range(len(y_values)):
            y[y == y_names[idx]] = y_indices[idx]
        # create test set
        X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=test_size,
                                                              shuffle=True, stratify=y, random_state=seed)
        # create training set
        if train_size == 1:
            X_train = X_train_
            y_train = y_train_
        else:
            X_train, _, y_train, _ = train_test_split(X_train_, y_train_, test_size=(1.0 - train_size),
                                                      shuffle=True, stratify=y_train_, random_state=seed)
    n_train, n_test, n_feature, n_class = X_train.shape[0], X_test.shape[0], X_train.shape[1], len(np.unique(y_train))
    print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
    print("X_test: {}, y_test: {}".format(X_test.shape, y_test.shape))
    print("n_train: {}, n_test: {}, n_feature: {}, n_class: {}".format(n_train, n_test, n_feature, n_class))
    # print("y_train_labels: {}, y_test_labels: {}".format(np.unique(y_train), np.unique(y_test)))
    print("feature_names: {}".format(names))

    return X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, names

