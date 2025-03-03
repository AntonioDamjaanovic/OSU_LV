try:
    number = float(input('Enter a number between 0.0 and 1.0: '))

    if number < 0.0 or number > 1.0:
        raise ValueError('Number is not in specified range!')
    
    if number >= 0.9:
        print('A')
    elif number >= 0.8:
        print('B')
    elif number >= 0.7:
        print('C')
    elif number >= 0.6:
        print('D')
    else:
        print('F')

except ValueError as ve:
    if str(ve) == 'Number is not in specified range!':
        print('Error: Number is not in the specified range (0.0 to 1.0)!')
    else:
        print('Error: Invalid input. Please enter a valid number!')
