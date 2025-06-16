import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a) dio zadatka
def basic_info():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    print(f'Broj mjerenja: {len(data)}\n')
    print(data.info())
    print(f'\nBroj izostalih vrijednosti:\n{data.isnull().sum()}')
    print(f'\nBroj dupliciranih vrijednosti:\n{data.duplicated().sum()}\n')

    data = data.dropna(axis=0)
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)

    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = data[col].astype('category')

    print('\nNakon konvertiranja tipa object u category:\n')
    print(data.info())

# b) dio zadatka
def highest_lowest_city_consumption():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    highest_city_consumption = data.sort_values(ascending = False, by = ['Fuel Consumption City (L/100km)']).head(3)
    print('Najveća potrošnja:')
    print(highest_city_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

    lowest_city_consumption = data.sort_values(ascending = True, by = ['Fuel Consumption City (L/100km)']).head(3)
    print('Najmanja potrošnja:')
    print(lowest_city_consumption[['Make', 'Model', 'Fuel Consumption City (L/100km)']])

# c) dio zadatka
def size_of_motor_and_emission():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    filtered_motor_size = data[(data['Engine Size (L)'] >= 2.5) & (data['Engine Size (L)'] <= 3.5)]
    print(f'Broj vozila s veličinom motora između 2.5 i 2.5 L: {len(filtered_motor_size)}')

    avg_co2_emission = filtered_motor_size['CO2 Emissions (g/km)'].mean()
    print(f'prosječna C02 emisija plinova za ova vozila: {avg_co2_emission}')

# d) dio zadatka
def filter_audi_cars():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    audi_cars = data[data['Make'] == 'Audi']
    num_of_audi_cars = len(audi_cars)
    print(f'Broj Audi vozila: {num_of_audi_cars}')

    audi_cars_with_4_cylinders = audi_cars[audi_cars['Cylinders'] == 4]
    avg_co2_emission = audi_cars_with_4_cylinders['CO2 Emissions (g/km)'].mean()
    print(f'prosječna C02 emisija plinova za Audi vozila: {avg_co2_emission}')

# e) dio zadatka
def filter_by_cylinders():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    num_of_cars_by_cylinders = data.groupby('Cylinders')['Model'].count()
    print("Broj vozila po cilindrima:\n")
    print(num_of_cars_by_cylinders)

    avg_emissions_by_cylinders = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
    print("\nProsječna emisija CO2 plinova s obzirom na broj cilindara:")
    print(avg_emissions_by_cylinders)

# f) dio zadatka
def filter_diesel_and_bensin_city_consumption():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    avg_diesel_consumption = data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].mean()
    print(f'Prosječnja gradska potrošnja dizel auta: {avg_diesel_consumption}')
    medial_diesel_consumption = data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].median()
    print(f'Medijalna gradska potrošnja dizel auta: {medial_diesel_consumption}')

    avg_bensin_consumption = data[data['Fuel Type'] == 'Z']['Fuel Consumption City (L/100km)'].mean()
    print(f'Prosječnja gradska potrošnja benzin auta: {avg_bensin_consumption}')
    medial_bensin_consumption = data[data['Fuel Type'] == 'Z']['Fuel Consumption City (L/100km)'].median()
    print(f'Medijalna gradska potrošnja dizel auta: {medial_bensin_consumption}')

# g) dio zadatka
def diesel_car_with_max_consumption():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    highest_city_consumption = data.sort_values(ascending = False, by = ['Fuel Consumption City (L/100km)'])
    car = highest_city_consumption[(highest_city_consumption['Cylinders'] == 4) & (highest_city_consumption['Fuel Type'] == 'D')].head(1)
    print(car)

# h) dio zadatka
def num_of_manual_cars():
    data = pd.read_csv('LV3/data_C02_emission.csv')

    manual_cars = data[data['Transmission'].str.startswith('M')]
    print(f'Broj manualnih auta: {len(manual_cars)}')

# i) dio zadatka
def calculate_correlation():
    data = pd.read_csv('LV3/data_C02_emission.csv')
    
    print(data.corr(numeric_only=True))

def main():
    basic_info()
    highest_lowest_city_consumption()
    size_of_motor_and_emission()
    filter_audi_cars()
    filter_by_cylinders()
    filter_diesel_and_bensin_city_consumption()
    diesel_car_with_max_consumption()
    num_of_manual_cars()
    calculate_correlation()

if __name__ == '__main__':
    main()

