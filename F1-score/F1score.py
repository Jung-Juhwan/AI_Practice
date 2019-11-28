# -*- coding:utf-8 -*-

from sklearn.metrics import f1_score,precision_score,recall_score

def printOutput():
    answer = "./answer.txt"
    result = "./result.txt"

    y_pred = []
    y = []

    with open(result, 'r', encoding='utf-8') as r:
        while True:
            line = r.readline()
            if line is not '\n' and line is not '':
                tokens = list(map(str, line.split('\t')))
                if tokens[0] > tokens[1]:
                    tokens[0] = 1
                    tokens[1] = 0
                else:
                    tokens[0] = 0
                    tokens[1] = 1

                y_pred.append(tokens[1])
            if not line:
                break
        r.close()

    with open(answer, 'r', encoding='utf-8') as a:
        while True:
            line = a.readline()
            if line is not '\n' and line is not '':
                tokens = list(map(int, line[0].split('\n')))
                y.append(tokens[0])
            if not line:
                break
        a.close


    # print the result
    print("")
    print(">> Performance")
    print("")
    print('\t' + "- F1_score :", f1_score(y,y_pred))
    print('\t' + "- Prediction :", precision_score(y,y_pred))
    print('\t' + "- Recall :", recall_score(y,y_pred))
    print("")
    print("")


if __name__ == '__main__':
    printOutput()
