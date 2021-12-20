import sys

import numpy as np
from sklearn.model_selection import train_test_split

#get all the line by group of points -(Each pair of points can define a line that passes through the two points)
def get_lines(points):
    num_of_exampel=len(points)
    lines=[]
    for i in range (num_of_exampel):
        x1=float(points[i][0])
        y1=float(points[i][1])
        for j in range (i+1,num_of_exampel):
            x2=float(points[j][0])
            y2=float(points[j][1])
            if x1!=x2:
                m=(y1-y2)/(x1-x2)
            else:
                m=0
            b=y1-(m*x1)
            lines.append([m,b,0])
            lines.append([m,b,1])
    return lines

#calculation the errors by the "important lines"
def calculation_error( points, labels,best_lines, weights,num_of_role):
    sum_error = 0
    for i in range(len(points)):
        x1=float(points[i][0])
        y1=float(points[i][1])
        label=int(labels[i])
        flag=0
        if (label)!= calculation_h_t( x1, y1, best_lines, weights,num_of_role):
            flag = 1
        sum_error=sum_error+flag
    return sum_error/(len(points))

#calculation the ans by algoritem find_h_t - Calculate the algorithm predicted
def calculation_h_t(x, y, best_line,weights,num_of_role):
    algo_ans = 0
    for i in range(num_of_role+1):
        ans = find_h_t(best_line[i],x, y)
        algo_ans=algo_ans+(ans*weights[i])
    if algo_ans>=0:
        return 1
    else:
        return -1

#Calculate the algorithm predicted
def find_h_t (line,x,y):
    m=line[0]
    b=line[1]
    side=line[2]
    x=float(x)
    y=float(y)
    ans= m*x+b
    if ans>=y:
        if side==0:
            return 1
        else:
            return -1
    else:
        if side==0:
            return -1
        else:
            return 1

def adaboost_algo(points, labels, k):
    X_train,X_test,y_train,y_test = train_test_split(points,labels,test_size=0.5)
    best_line=[]
    train_err_result = []
    test_err_result = []
    weights_of_lines=[]
    num_of_exampel=len(X_train)
    lines=get_lines(X_train)
    weights=[1/num_of_exampel]*num_of_exampel
    for it in range(k):
        min_weighted_error=sys.maxsize
        for line in range (len(lines)):
            if not best_line.count(lines[line]):
                weighted_error = 0
                for ind in range (num_of_exampel):

                    x1=float(X_train[ind][0])
                    y1=float(X_train[ind][1])
                    label=float(y_train[ind])

                        # get the prediction
                    ans=find_h_t(lines[line],x1,y1)

                        #compute weight error
                    if ans==label:
                        flag=0
                    else:
                        flag=1
                    weighted_error=weighted_error+(weights[ind]*flag)

                    #min weight erorr
                if weighted_error<=min_weighted_error:
                    min_weighted_error=weighted_error
                    h_t=lines[line]

        best_line.append(h_t)


        #min weighted error:
        alpha_t=0.5 * np.log((1-min_weighted_error)/min_weighted_error)
        #print("alpha_t  ",alpha_t)
        weights_of_lines.append(alpha_t)

        #train & test errors
        train_err_result.append(calculation_error(X_train, y_train, best_line, weights_of_lines, it))
        test_err_result.append(calculation_error(X_test, y_test, best_line, weights_of_lines, it))

        for j in range (num_of_exampel):
            x1=float(X_train[ind][0])
            y1=float(X_train[ind][1])
            label=float(y_train[ind])
            weights[j]=weights[j]*np.exp(-alpha_t*find_h_t(h_t,x1,y1)*label)
        for i in range(len(weights)):
            weights[i] = weights[i]/sum(weights)
    return train_err_result, test_err_result


