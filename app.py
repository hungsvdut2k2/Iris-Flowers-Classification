from models import knn
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("datasets/IRIS.csv", index_col=False)
    features = df.iloc[:, :].to_numpy()
    # split data into training set and testing set
    train_set, test_set = train_test_split(features)
    data_point = [5.2, 3.4, 1.4, 0.2]
    model_knn = knn(features=train_set, data_point=data_point, k=5)
    print(model_knn.voting(method="euclidean"))
