import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

wine_dataset = pd.read_csv('./winequalityN.csv').drop("type", axis=1)
modelRandomForest = RandomForestClassifier(
    random_state = 0,
    min_samples_split = 8,
    max_depth = 6,
)
# svm_model = SVC()
knn_model = KNeighborsClassifier()
cls_cv = StratifiedKFold(shuffle = True, random_state = 0)
accuracy_random_fr = None
recall_random_fr = None
specificity_random_fr = None
accuracy_knn = None
recall_knn = None
specificity_knn = None

X = wine_dataset.drop("quality", axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 6 else 0)

def analyze_wine_dataset(dataset):
    print("Shape of the dataset:")
    print(dataset.shape)
    print("\nFirst 20 rows of the dataset:")
    print(dataset.head(20))

    #Kiểm tra giá trị thiếu
    print("\nMissing values:")
    print(dataset.isnull().sum())
    # Thống kê mô tả
    print("\nStatistical measures of the dataset:")
    print(dataset.describe())

     # Số lượng giá trị cho mỗi chất lượng (quality)
    sns.set_style("darkgrid")
    sns.catplot(x='quality', data=dataset, kind="count").set(title='Số lượng cho mỗi giá trị của cột quality')
    plt.show()

    # Sự tương quan giữa volatile acidity và chất lượng rượu
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x="quality", y="volatile acidity", data=dataset).set(title="Sự tương quan giữa volatile acidity và chất lượng rượu")

    # Sự tương quan giữa citric acid và chất lượng rượu
    plt.figure(figsize=(8, 6))
    sns.barplot(x="quality", y="citric acid", data=dataset).set(title="Sự tương quan giữa citric acid và chất lượng rượu")
    plt.show()

def valueInConfusion(cm):
    TP = cm[1, 1]  # True Positive
    TN = cm[0, 0]  # True Negative
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative

    # Tính toán độ chính xác
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Tính toán độ nhạy (Recall)
    recall = TP / (TP + FN)
    # Tính toán độ đặc hiệu (Specificity)
    specificity = TN / (TN + FP)
    return accuracy, recall, specificity

def confusionMatrix(X_test, Y_test):
    # Dự đoán trên tập kiểm tra
    Y_pred_randomFr = modelRandomForest.predict(X_test.values)
    Y_pred_Knn= knn_model.predict(X_test.values)

    # Tính ma trận confusion
    cm_random_fr = confusion_matrix(Y_test.values, Y_pred_randomFr)
    cm_knn = confusion_matrix(Y_test.values, Y_pred_Knn)
    # In ma trận confusion
    # print("Confusion Matrix:")
    # print(cm)

    # Trích xuất các giá trị từ ma trận confusion
    _accuracy_random_fr, _recall_random_fr, _specificity_random_fr = valueInConfusion(cm_random_fr)
    _accuracy_knn, _recall_knn, _specificity_knn = valueInConfusion(cm_knn)
    global accuracy_random_fr
    global recall_random_fr
    global specificity_random_fr
    global accuracy_knn
    global recall_knn
    global specificity_knn
    accuracy_random_fr = _accuracy_random_fr
    recall_random_fr = _recall_random_fr
    specificity_random_fr = _specificity_random_fr
    accuracy_knn = _accuracy_knn
    recall_knn = _recall_knn
    specificity_knn = _specificity_knn

def score_binary_classifier(classifier, X = X, y = Y, cv = cls_cv,
                            scoring = ['accuracy', 'precision', 'recall', 'f1'],):
    result = cross_validate(
        classifier, X, y,
        scoring = scoring,
        cv = cv,
        return_train_score = True,
        return_estimator = True,
        error_score = 'raise',
    )
    score_df = pd.DataFrame(
        {
            a: b
            for k in scoring
            for a, b in {
                f'train_{k}': result[f'train_{k}'],
                f'test_{k}': result[f'test_{k}'],
            }.items()
        }
    )
    return {**result, 'score_df': score_df}

def input_datas():
    # Nhập vào một chuỗi từ người dùng
    input_string = input("Nhập vào các số, cách nhau bằng dấu phẩy: ")

    # Tách chuỗi thành danh sách các số
    numbers = [float(num.strip()) for num in input_string.split(',')]

    return numbers

def merchineLearning():
    # model training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    # print(X_train, Y_train)
    modelRandomForest.fit(X_train.values, Y_train.values)
    knn_model.fit(X_train.values, Y_train.values)
    confusionMatrix(X_test, Y_test)
    # result = score_binary_classifier(modelRandomForest)
    # print(result['score_df'])
    print("learning success!")

def output():
    datas = input_datas()

    # changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(datas)

    # reshape the data as we are predicting the label for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = modelRandomForest.predict(input_data_reshaped)

    print("random forest : ")
    if(prediction[0] == 1):
        print("Good quality wine")
    else:
        print("bad quality wine")

    print("KNN : ")
    knn_prediction = knn_model.predict(input_data_reshaped)
    if(knn_prediction[0] == 1):
        print("Good quality wine")
    else:
        print("bad quality wine")

    print("Accuracy random forest: {:.2f}%".format(accuracy_random_fr*100))
    print("Recall random forest: {:.2f}%".format(recall_random_fr*100))
    print("Specificity random forest: {:.2f}%".format(specificity_random_fr*100))
    print("Accuracy random knn: {:.2f}%".format(accuracy_knn*100))
    print("Recall random knn: {:.2f}%".format(recall_knn*100))
    print("Specificity random knn: {:.2f}%".format(specificity_knn*100))
    # print("Accuracy knn: {:.2f}%".format(test_data_accuracy_percentage_knn))

def input_numbers(input_string):
    # Tách chuỗi thành danh sách các số
    numbers = [float(num.strip()) for num in input_string.split(',')]
    return numbers

def main_menu():
    while True:
        print("\nMENU:")
        print("1. Phân tích dữ liệu")
        print("2. Học máy")
        print("3. Nhập dữ liệu")
        print("4. Thoát chương trình")
        choice = input("Mời chọn :")
        if choice == '1':
            analyze_wine_dataset(wine_dataset)
            continue
        elif choice == '2':
            merchineLearning()
            continue
        elif choice == '3':
            output()
            continue
        elif choice == '4':
            return
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
main_menu()