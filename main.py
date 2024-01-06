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
import tkinter as tk
from tkinter import messagebox

wine_dataset = pd.read_csv('./winequality_6000.csv').drop("type", axis=1)
modelRandomForest = RandomForestClassifier(
    random_state = 0,
    min_samples_split = 8,
    max_depth = 6,
)
knn_model = KNeighborsClassifier()
accuracy_random_fr = None
recall_random_fr = None
specificity_random_fr = None
precision_random_fr = None
f1_score_random_fr= None
accuracy_knn = None
recall_knn = None
specificity_knn = None
precision_knn = None
f1_score_knn= None

# accuracy_score 
score_accuracy_randomFr = None
score_accuracy_knn = None

# Tạo biến X và Y
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

def accuracy(X_test, Y_test):
    Y_pred_randomFr = modelRandomForest.predict(X_test.values)
    Y_pred_knn = knn_model.predict(X_test.values)
    # Tính độ chính xác
    accuracy_randomFr = accuracy_score(Y_test.values, Y_pred_randomFr)
    accuracy_knn = accuracy_score(Y_test.values, Y_pred_knn)
    global score_accuracy_randomFr
    global score_accuracy_knn
    score_accuracy_randomFr = accuracy_randomFr
    score_accuracy_knn = accuracy_knn

def valueInConfusion(cm):
    print("ma trận đánh giá mô hình confusion matrix: \n")
    print(cm)
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
    # Tính toán độ chính xác dương tính (Precision)
    precision = TP / (TP + FP)
    # Tính toán F1-score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, recall, specificity, precision, f1_score

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
    _accuracy_random_fr, _recall_random_fr, _specificity_random_fr, _precision_random_fr, _f1_score_random_fr = valueInConfusion(cm_random_fr)
    _accuracy_knn, _recall_knn, _specificity_knn, _precision_knn, _f1_score_knn = valueInConfusion(cm_knn)
    global accuracy_random_fr
    global recall_random_fr
    global specificity_random_fr
    global precision_random_fr
    global f1_score_random_fr
    global accuracy_knn
    global recall_knn
    global specificity_knn
    global precision_knn
    global f1_score_knn
    accuracy_random_fr = _accuracy_random_fr
    recall_random_fr = _recall_random_fr
    specificity_random_fr = _specificity_random_fr
    precision_random_fr = _precision_random_fr
    f1_score_random_fr = _f1_score_random_fr
    accuracy_knn = _accuracy_knn
    recall_knn = _recall_knn
    specificity_knn = _specificity_knn
    precision_knn = _precision_knn
    f1_score_knn = _f1_score_knn

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
    accuracy(X_test, Y_test)
    print("learning success!")

def input_data():
    # Tạo cửa sổ
    window = tk.Tk()
    window.title("Form Nhập Thông Tin")
    
    # Tạo và định vị các widget trong cửa sổ
    fields = [
        ("Fixed Acidity:", 0),
        ("Volatile Acidity:", 1),
        ("Citric Acid:", 2),
        ("Residual Sugar:", 3),
        ("Chlorides:", 4),
        ("Free Sulfur Dioxide:", 5),
        ("Total Sulfur Dioxide:", 6),
        ("Density:", 7),
        ("pH:", 8),
        ("Sulphates:", 9),
        ("Alcohol:", 10)
    ]

    entry_values = []

    for field, row in fields:
        label = tk.Label(window, text=field)
        label.grid(row=row, column=0, padx=10, pady=10)

        entry = tk.Entry(window)
        entry.grid(row=row, column=1, padx=10, pady=10)
        entry_values.append(entry)

    result_label = tk.Label(window, text="")
    result_label.grid(row=13, column=0)

    def submit_form():
        try:
            # Lấy giá trị từ các trường nhập liệu
            input_values = [float(entry.get()) for entry in entry_values]
            prediction_random_forest, prediction_knn = output(input_values)
            
            # Hiển thị thông báo với các kết quả
            messagebox.showinfo("Kết quả",f"Random Forest Prediction: {'Good quality wine' if prediction_random_forest[0] == 1 else 'Bad quality wine'}\n"
                                f"Random Forest Accuracy: {accuracy_random_fr:.2f}%\n"
                                f"Random Forest Recall: {recall_random_fr:.2f}%\n"
                                f"Random Forest Specificity: {specificity_random_fr:.2f}%\n"
                                f"Random Forest Precision: {precision_random_fr:.2f}%\n"
                                f"Random Forest F1 Score: {f1_score_random_fr:.2f}%\n"
                                f"Random Forest Accuracy (using accuracy_score): {score_accuracy_randomFr:.2f}%\n"
                                "--------------------**********---------------------------\n"
                                f"KNN Prediction: {'Good quality wine' if prediction_knn[0] == 1 else 'Bad quality wine'}\n"
                                f"KNN Accuracy: {accuracy_knn:.2f}%\n"
                                f"KNN Recall: {recall_knn:.2f}%\n"
                                f"KNN Specificity: {specificity_knn:.2f}%\n"
                                f"KNN Precision: {precision_knn:.2f}%\n"
                                f"KNN F1 Score: {f1_score_knn:.2f}%\n"
                                f"KNN Accuracy (using accuracy_score): {score_accuracy_knn:.2f}%")
        except ValueError:
            result_label.config(text="Lỗi: Vui lòng chỉ nhập các giá trị số.")
            
    # Tạo nút "Submit"
    submit_button = tk.Button(window, text="Submit", command=submit_form)
    submit_button.grid(row=12, column=0, columnspan=2, pady=10)
    # Chạy vòng lặp sự kiện
    window.mainloop()

def output(datas):
    # datas = input_datas()

    # changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(datas)

    # reshape the data as we are predicting the label for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = modelRandomForest.predict(input_data_reshaped)
    knn_prediction = knn_model.predict(input_data_reshaped)
    return prediction, knn_prediction 
    print("random forest: ")
    if(prediction[0] == 1):
        print("Good quality wine")
    else:
        print("bad quality wine")

    print("Accuracy random forest: {:.2f}%".format(accuracy_random_fr*100))
    print("Recall random forest: {:.2f}%".format(recall_random_fr*100))
    print("Specificity random forest: {:.2f}%".format(specificity_random_fr*100))
    print("precision random forest: {:.2f}%".format(precision_random_fr*100))
    print("f1 random forest: {:.2f}%".format(f1_score_random_fr*100))
    print("accuracy dựa vào accuracy_score: {:.2f}%".format(score_accuracy_randomFr*100))
    print("----------------*********************-----------------")

    print("KNN: ")
    if(knn_prediction[0] == 1):
        print("Good quality wine")
    else:
        print("bad quality wine")
    print("Accuracy knn: {:.2f}%".format(accuracy_knn*100))
    print("Recall knn: {:.2f}%".format(recall_knn*100))
    print("Specificity knn: {:.2f}%".format(specificity_knn*100))
    print("precision knn: {:.2f}%".format(precision_knn*100))
    print("f1 knn: {:.2f}%".format(f1_score_knn*100))
    print("accuracy dựa vào accuracy_score: {:.2f}%".format(score_accuracy_knn*100))

def main_menu():
    while True:
        print("\nMENU:")
        print("1. Phân tích dữ liệu")
        print("2. Học máy")
        print("3. Nhập dữ liệu")
        print("4. Thoát chương trình")
        choice = input("Mời chọn:")
        if choice == '1':
            analyze_wine_dataset(wine_dataset)
            continue
        elif choice == '2':
            merchineLearning()
            continue
        elif choice == '3':
            input_data()
            continue
        elif choice == '4':
            return
        else:
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")
main_menu()