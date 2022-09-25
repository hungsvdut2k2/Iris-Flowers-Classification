from cgi import test
from itertools import count
from models import knn
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv("datasets/IRIS.csv", index_col=False)
    features = df.iloc[:, :].to_numpy()
    # split data into training set and testing set
    train_set, test_set = train_test_split(features)
    # test accuracy with euclidean distance
    count_right_answer_euclidean = 0
    for test_case in test_set:
        data_point = test_case[:4]
        model_knn = knn(features=train_set, data_point=data_point, k=5)
        if model_knn.voting(method="euclidean") == test_case[-1]:
            count_right_answer_euclidean += 1
    print(count_right_answer_euclidean / len(test_set))

    # test accuracy with manhattan distance
    count_right_answer_manhattan = 0
    for test_case in test_set:
        data_point = test_case[:4]
        model_knn = knn(features=train_set, data_point=data_point, k=5)
        if model_knn.voting(method="manhattan") == test_case[-1]:
            count_right_answer_manhattan += 1
    print(count_right_answer_manhattan / len(test_set))

    # test accuracy with chebyshev distance

    count_right_answer_chebyshev = 0
    for test_case in test_set:
        data_point = test_case[:4]
        model_knn = knn(features=train_set, data_point=data_point, k=5)
        if model_knn.voting(method="chebyshev") == test_case[-1]:
            count_right_answer_manhattan += 1
    print(count_right_answer_chebyshev / len(test_set))
