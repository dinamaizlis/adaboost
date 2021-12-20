
import matplotlib.pyplot as plt
from adaboost import *

def plot_errors( test_err,train_err):
    plt.title("train and test error :")
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.plot([*range(1, 9)], test_err, color='green', label='test error')
    plt.plot([*range(1, 9)], train_err, color='blue', label='train error')
    plt.legend(loc ="lower right")
    plt.show()


if __name__ == '__main__':

    path="rectangle.txt"
    file = open(path)
    points = []
    H = []
    labels = []
    for line in file:
        splitedLine = line.split()
        points.append([(splitedLine[0]),(splitedLine[1])])
        labels.append((splitedLine[2]))
    for line in file:
        splitedLine = line.split()



    test_errors_ans =[]
    train_errors_ans = []
    train_errors = []
    test_errors = []
    iter=100
    empirical_errors = []
    test_errors = []

    # epochs errors lists for plotting
    epochs_train_errors = [0]*8
    epochs_test_errors = [0]*8
    for i in range(iter):
        train_err_result, test_err_result = adaboost_algo(points, labels, 8)
        empirical_errors.extend(train_err_result)
        test_errors.extend(test_err_result)
        epochs_train_errors = np.add(epochs_train_errors, train_err_result)
        epochs_test_errors = np.add(epochs_test_errors, test_err_result)

        if i%10==0:
            print("train errors mean in iterations: ",i," : ", sum(empirical_errors) / len(empirical_errors))
            print("test errors mean iterations: ",i," : ", sum(test_errors) / len(test_errors))
            print("-------------------------------")
    for i in range(len(epochs_test_errors)):
        epochs_train_errors[i]=epochs_train_errors[i]/iter
        epochs_test_errors[i]=epochs_test_errors[i]/iter

    plot_errors(epochs_train_errors, epochs_test_errors)

