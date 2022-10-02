from models import naive_bayes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("datasets/IRIS.csv", index_col=False)
    features = df.iloc[:, :].to_numpy()
    train_set, test_set = train_test_split(features)
    count = 0
    for test in test_set:
        input_data = test[:-1]
        label = test[-1]
        model_naive_bayes = naive_bayes(train_set=train_set, input_data=input_data)
        if model_naive_bayes.predict() == label:
            count += 1
    print(count / test_set.shape[0])
