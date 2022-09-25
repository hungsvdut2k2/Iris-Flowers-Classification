from models import knn
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("datasets/IRIS.csv", index_col=False)
    features = df.iloc[:, :4].to_numpy()
    labels = df.iloc[:, -1].to_numpy().reshape(150, 1)
    # split data into training set and testing set
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, random_state=1
    )
    data_point = [5.2, 3.4, 1.4, 0.2]
    model_knn = knn(features=features_train, labels=labels_train, data_point=data_point)
    print(model_knn.chebyshev_distance())
