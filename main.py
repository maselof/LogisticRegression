import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from LogisticRegression import LogisticRegression


# убираем Толщина КС, Инсулин

def main():
    diabet_df = pd.read_excel('src/diabetes.xlsx')
    X = diabet_df[["Беременность", "Глюкоза", "АД", "Толщина КС", "Инсулин", "ИМТ", "Наследственность", "Возраст"]]
    y = diabet_df["Диагноз"]
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.2,
                                                        shuffle=True)
    n, m = X_train.shape

    logr = LogisticRegression(m)
    logr.gradient_descent(X_train, y_train)

    train_prediction = np.array(logr.predict(X_train))
    train_accuracy = np.sum((train_prediction > 0.5) == y_train) / len(train_prediction)
    print(f'Точность на обучающейся выборке: {round(train_accuracy * 100, 2)}%')

    train_prediction = np.array(logr.predict(X_test))
    train_accuracy = np.sum((train_prediction > 0.5) == y_test) / len(train_prediction)
    print(f'Точность на тестовой выборке: {round(train_accuracy * 100, 2)}%')

    print("===========================ПОСЛЕ УДАЛЕНИЯ ПАРАМЕТРОВ МАЛО ВЛИЯЮЩИЕ НА РЕЗУЛЬТАТ============================")

    diabet_df = pd.read_excel('src/diabetes.xlsx')
    X = diabet_df[["Беременность", "Глюкоза", "АД", "ИМТ", "Наследственность", "Возраст"]]
    y = diabet_df["Диагноз"]
    X = X.to_numpy()
    y = y.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.2,
                                                        shuffle=True)
    n, m = X_train.shape

    logr = LogisticRegression(m)
    logr.gradient_descent(X_train, y_train)

    train_prediction = np.array(logr.predict(X_train))
    train_accuracy = np.sum((train_prediction > 0.5) == y_train) / len(train_prediction)
    print(f'Точность на обучающейся выборке: {round(train_accuracy * 100, 2)}%')

    train_prediction = np.array(logr.predict(X_test))
    train_accuracy = np.sum((train_prediction > 0.5) == y_test) / len(train_prediction)
    print(f'Точность на тестовой выборке: {round(train_accuracy * 100, 2)}%')


if __name__ == '__main__':
    main()


