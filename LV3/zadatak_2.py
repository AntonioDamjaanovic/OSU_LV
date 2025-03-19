import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a) dio zadatka
def co2_histogram():
    data = pd.read_csv('data_C02_emission.csv')

    plt.figure()
    data['CO2 Emissions (g/km)'].plot(kind = 'hist', bins = 20)
    plt.show()

# b) dio zadatka
def scatter_city_consumption_and_co2_emission():
    data = pd.read_csv('data_C02_emission.csv')

    data.plot.scatter(
        x = 'Fuel Consumption City (L/100km)',
        y = 'CO2 Emissions (g/km)'
    )
    plt.title('Relationship Between City Fuel Consumption and CO2 Emissions')
    plt.xlabel('Fuel Consumption City (L/100km)')
    plt.ylabel('CO2 Emissions (g/km)')
    plt.show()

# c) dio zadatka
def boxplot_highway_consumption_by_fuel_type():
    data = pd.read_csv('data_C02_emission.csv')

    data.boxplot(column='Fuel Consumption Hwy (L/100km)', by='Fuel Type', grid=False)
    plt.title('Distribution of Highway Fuel Consumption by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Fuel Consumption Hwy (L/100km)')
    plt.show()

# d) dio zadatka
def bar_chart_vehicles_by_fuel_type():
    data = pd.read_csv('data_C02_emission.csv')

    fuel_type_counts = data.groupby('Fuel Type').size().reset_index(name='Number of Vehicles')

    plt.figure()
    plt.bar(fuel_type_counts['Fuel Type'], fuel_type_counts['Number of Vehicles'])
    plt.title('Number of Vehicles by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Number of Vehicles')
    plt.show()

# e) dio zadatka
def bar_chart_avg_co2_by_cylinders():
    data = pd.read_csv('data_C02_emission.csv')

    avg_co2_by_cylinders = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean().reset_index()

    plt.figure()
    plt.bar(avg_co2_by_cylinders['Cylinders'], avg_co2_by_cylinders['CO2 Emissions (g/km)'])
    plt.title('Average CO2 Emissions by Number of Cylinders')
    plt.xlabel('Number of Cylinders')
    plt.ylabel('Average CO2 Emissions (g/km)')
    plt.show()

def main():
    #co2_histogram()
    #scatter_city_consumption_and_co2_emission()
    #boxplot_highway_consumption_by_fuel_type()
    #bar_chart_vehicles_by_fuel_type()
    bar_chart_avg_co2_by_cylinders()

if __name__ == '__main__':
    main()