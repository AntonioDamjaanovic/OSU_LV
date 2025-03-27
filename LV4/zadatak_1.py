import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm

# b) dio zadatka
def c02_and_city_consumption(X_train, X_test, y_train, y_test):
    plt.figure()
    plt.scatter(X_train['Fuel Consumption City (L/100km)'], y_train, color='blue', alpha=0.5)
    plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test, color='red', alpha=0.5)
    plt.title('Ovisnos CO2 emisije o potrošnji u gradu')
    plt.xlabel('Potrošnja u gradu L/100km')
    plt.ylabel('CO2 emisija g/km')
    plt.show()

# c) dio zadatka
def scale_sets(X_train, X_test):
    sc = MinMaxScaler()
    X_train_n = sc.fit_transform(X_train)
    X_test_n = sc.transform(X_test)

    X_train_column = []
    for i in range(X_train_n.shape[0]):
        X_train_column.append(X_train_n[i][2])

    figure, axis = plt.subplots(1, 2)
    axis[0].hist(X_train['Fuel Consumption City (L/100km)'], bins=30, color='blue', alpha=0.7)
    axis[0].set_title('Fuel Consumption City (L/100km) prije skaliranja')
    axis[1].hist(X_train_column, bins=30, color='blue', alpha=0.7)
    axis[1].set_title('Fuel Consumption City (L/100km) poslije skaliranja')
    plt.show()

    return X_train_n, X_test_n

# d) dio zadatka
def create_lin_model(X_train_n, y_train):
    linearModel = lm.LinearRegression()
    linearModel.fit(X_train_n, y_train)
    print(linearModel.coef_)

    return linearModel

# e) dio zadatka
def evaluate_model(linearModel, X_test_n, y_test):
    y_test_p = linearModel.predict(X_test_n)

    plt.figure()
    plt.scatter(y_test, y_test_p, alpha=0.7)
    plt.title('Evaluacija izlaznih veličina')
    plt.show()


def main():
    df = pd.read_csv('LV4/data_C02_emission.csv')
    numeric_columns = df.select_dtypes(include=['number'])
    

    X = numeric_columns.drop(columns=['CO2 Emissions (g/km)'])
    y = df['CO2 Emissions (g/km)']

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=1)

    #c02_and_city_consumption(X_train, X_test, y_train, y_test)
    X_train_n, X_test_n = scale_sets(X_train, X_test)
    
    linearModel = create_lin_model(X_train_n, y_train)

    evaluate_model(linearModel, X_test_n, y_test)


if __name__ == '__main__':
    main()