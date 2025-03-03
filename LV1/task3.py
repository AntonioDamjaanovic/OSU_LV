numbers = []

while True:
    user_input = input('Enter a number: ')
    if user_input == 'Done':
        break

    try:
        number = int(user_input)
        numbers.append(number)
    except ValueError:
        print('Wrong input, please enter a number or "Done".')

if len(numbers) > 0:
    print(f"\nYou entered {len(numbers)} numbers.")
    print(f"Average of your numbers is: {sum(numbers) / len(numbers):.2f}.")
    print(f"Min value is: {min(numbers)}")
    print(f"Max value is: {max(numbers)}")

    sorted_numbers = sorted(numbers)
    print(f"Sorted numbers: {sorted_numbers}")
else:
    print("You didn't enter a single number")