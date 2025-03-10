import numpy as np
import matplotlib.pyplot as plt

def height_and_mass(data):
    heights_of_people = data[:, 1]
    weights_of_people = data[:, 2]
    plt.figure(figsize=(10,6))
    plt.scatter(heights_of_people, weights_of_people, color='blue', label='Osobe', alpha=0.5)

    plt.title('Odnos visine i mase osobe')
    plt.xlabel('Visina (cm)')
    plt.ylabel('Masa (kg)')
    plt.legend()
    plt.grid(True)
    plt.show()

# every 50th person
def height_and_mass_every_50(data):
    heights_of_people = data[::50, 1]
    weights_of_people = data[::50, 2]

    plt.figure(figsize=(10,6))
    plt.scatter(heights_of_people, weights_of_people, color='blue', label='Osobe', alpha=0.5)

    plt.title('Odnos visine i mase svake 50te osobe')
    plt.xlabel('Visina (cm)')
    plt.ylabel('Masa (kg)')
    plt.legend()
    plt.grid(True)
    plt.show()

# min, max and avg height of person
def filter_heights(data):
    heights_of_people = data[:, 1]
    print(f'\tMin height: {heights_of_people.min()}')
    print(f'\tMax height: {heights_of_people.max()}')
    print(f'\tAverage height: {heights_of_people.mean()}')

def filter_heights_of_men(data):
    men = data[data[:, 0] == 1]
    print('Men height statistics: ')
    filter_heights(men)

def filter_heights_of_women(data):
    women = data[data[:, 0] == 0]
    print('Women height statistics: ')
    filter_heights(women)

def main():
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

    number_of_people = len(data)
    print(f'a) number of people: {number_of_people}')

    #height_and_mass(data)
    #height_and_mass_every_50(data)
    filter_heights(data)
    filter_heights_of_men(data)
    filter_heights_of_women(data)

if __name__ == '__main__':
    main()