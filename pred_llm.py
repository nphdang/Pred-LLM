import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tabular_llm.predllm_utils import _encode_row_partial
from tabular_llm.predllm import PredLLM
import read_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="iris", type=str, nargs='?', help='dataset name')
parser.add_argument('--method', default="pred_llm", type=str, nargs='?', help='generative method')
parser.add_argument('--trainsize', default="1.0", type=str, nargs='?', help='size of training set')
parser.add_argument('--testsize', default="0.2", type=str, nargs='?', help='size of test set')
parser.add_argument('--gensize', default="1.0", type=str, nargs='?', help='size of generation set')
parser.add_argument('--runs', default="3", type=str, nargs='?', help='no of times to run algorithm')
args = parser.parse_args()
print("dataset: {}, method: {}, train_size: {}, test_size: {}, gen_size: {}".
      format(args.dataset, args.method, args.trainsize, args.testsize, args.gensize))

dataset_input = args.dataset
method_input = args.method
train_size = float(args.trainsize)
test_size = float(args.testsize)
gen_size = float(args.gensize)
n_run = int(args.runs)

llm_batch_size = 32
llm_epochs = 50

if dataset_input == "classification":
    datasets = ["iris", "breast_cancer", "australian",
                "blood_transfusion", "steel_plates_fault",
                "qsar_biodeg", "phoneme", "waveform",
                "churn", "cardiotocography",
                "kc1", "kc2", "balance_scale",
                "diabetes", "compas", "bank", "adult"]
else:
    datasets = [dataset_input]
print("datasets: {}".format(datasets))

if method_input == "all":
    methods = ["original", "pred_llm"]
else:
    methods = [method_input]
print("methods: {}".format(methods))

xgb_dataset_method_run = []
for dataset in datasets:
    print("dataset: {}".format(dataset))
    # compute no of generated samples
    _, _, _, _, n_generative, _, n_feature, n_class, feature_names = read_data.gen_train_test_data(dataset,
                                                                                                   train_size=gen_size,
                                                                                                   normalize_x=None)
    xgb_method_run = []
    for method in methods:
        print("method: {}".format(method))
        xgb_run = np.zeros(n_run)
        for run in range(n_run):
            print("run: {}".format(run))
            np.random.seed(run)
            X_train, y_train, X_test, y_test, n_train, n_test, _, _, _ = \
                read_data.gen_train_test_data(dataset, train_size, test_size, normalize_x=True, seed=run)
            # train a classifier to predict labels of synthetic samples
            xgb_org = XGBClassifier(random_state=run)
            xgb_org.fit(X_train, y_train)
            y_pred = xgb_org.predict(X_test)
            acc_org = round(accuracy_score(y_test, y_pred), 4)
            print("original accuracy-{}: {}".format(run, acc_org))
            # train a generative method
            if method == "original":
                X_train_new, y_train_new = X_train, y_train
            if method == "pred_llm":
                X_y_train = np.append(X_train, y_train.reshape(-1, 1), axis=1)
                X_y_train_df = pd.DataFrame(X_y_train)
                X_y_train_df.columns = np.append(feature_names, "target")
                predllm = PredLLM(llm='distilgpt2', batch_size=llm_batch_size, epochs=llm_epochs)
                predllm.fit(X_y_train_df)
                # compute length of input sequence
                encoded_text = _encode_row_partial(X_y_train_df.iloc[0], shuffle=False)
                prompt_len = len(predllm.tokenizer(encoded_text)["input_ids"])
                X_y_train_new = predllm.sample_new(n_samples=n_generative, max_length=prompt_len, task="classification")
                X_train_new = X_y_train_new.iloc[:, :-1].to_numpy(dtype=float).reshape(-1, n_feature)
                y_train_new = X_y_train_new.iloc[:, -1:].to_numpy(dtype=float).reshape(-1, )

            print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
            print("n_generative: {}".format(n_generative))
            print("X_train_new: {}, y_train_new: {}".format(X_train_new.shape, y_train_new.shape))
            # convert labels from float to int
            y_train_new = np.array([int(y) for y in y_train_new])
            # save result to text file
            file_name = "ds{}_tr{}_te{}_ge{}_run{}".format(dataset, train_size, test_size, gen_size, run)
            with open("./results/_classification/{}/X_gen_{}.npz".format(method, file_name), "wb") as f:
                np.save(f, X_train_new)
            with open("./results/_classification/{}/y_gen_{}.npz".format(method, file_name), "wb") as f:
                np.save(f, y_train_new)

            # get number of generative classes
            n_class_generative = len(np.unique(y_train_new))
            # train a classifier
            if n_class_generative != n_class:
                print("generate less/more than the number of real classes")
                acc_new = 0
            else:
                xgb_new = XGBClassifier(random_state=run)
                xgb_new.fit(X_train_new, y_train_new)
                y_pred = xgb_new.predict(X_test)
                acc_new = round(accuracy_score(y_test, y_pred), 4)
                print("dataset: {}, method: {}, accuracy-{}: {}".format(dataset, method, run, acc_new))
            # save accuracy of each run of each method for each train_size in each dataset
            xgb_run[run] = acc_new
            # save result to text file
            if run == (n_run - 1):
                with open('./results/_classification/{}/accuracy_{}.txt'.format(method, file_name), 'w') as f:
                    acc_avg = round(np.mean(xgb_run), 4)
                    acc_std = round(np.std(xgb_run), 4)
                    f.write("accuracy ORG: {}\n".format(acc_org))
                    f.write("accuracy GEN: {}\n".format(acc_new))
                    f.write("accuracy AVG: {} ({})\n".format(acc_avg, acc_std))
        # save accuracy of n_run of each method in each dataset
        xgb_method_run.append(xgb_run)
    # save accuracy of n_run of all methods in each dataset
    xgb_dataset_method_run.append(xgb_method_run)

# save all results to csv file
n_dataset = len(datasets)
n_method = len(methods)
file_result = './results/_classification/accuracy_ds{}_me{}_tr{}_te{}_ge{}'.\
    format(dataset_input, method_input, train_size, test_size, gen_size)
with open(file_result + ".csv", 'w') as f:
    f.write("dataset,method,classifier,train_size,test_size,gen_size,run,accuracy\n")
    for data_id in range(n_dataset):
        for method_id in range(n_method):
            for run_id in range(n_run):
                dataset_name = datasets[data_id]
                method_name = methods[method_id]
                classifier = "xgb"
                line = dataset_name + "," + method_name + "," + classifier + "," + \
                       str(train_size) + "," + str(test_size) + "," + str(gen_size) + "," + \
                       str(run_id) + "," + str(xgb_dataset_method_run[data_id][method_id][run_id]) + "\n"
                f.write(line)

# save accuracy of all datasets to text file
with open(file_result + ".txt", 'w') as f:
    acc_avg = round(np.mean(xgb_dataset_method_run), 4)
    acc_std = round(np.std(xgb_dataset_method_run), 4)
    f.write("accuracy: {} ({})\n".format(acc_avg, acc_std))

