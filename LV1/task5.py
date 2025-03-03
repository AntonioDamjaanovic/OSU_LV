def average_sms_count(sms_list):
    total_words = 0
    for sms in sms_list:
        words = len(sms.split(" "))
        total_words += words

    return total_words / len(sms_list)

def count_sms_ending_with_exclamation_mark(sms_list):
    count = 0
    for sms in sms_list:
        if sms[-1]  == '!':
            count += 1

    return count


def main():
    filename = 'SMSSpamCollection.txt'
    ham_sms = []
    spam_sms = []

    try:
        with open(filename, 'r', encoding="utf-8") as file:
            for line in file:
                line = line.rstrip()
                parts = line.split('\t')

                if parts[0] == 'ham':
                    ham_sms.append(parts[1])
                elif parts[0] == 'spam':
                    spam_sms.append(parts[1])
        
    except FileNotFoundError:
        print("Error: Provided file not found!")

    print(f'\nAverage number of words in ham messages is: {average_sms_count(ham_sms)}')
    print(f'Average number of words in spam messages is: {average_sms_count(spam_sms)}')
    print(f'\nNumber of spam sms ending with exclamation mark is: {count_sms_ending_with_exclamation_mark(spam_sms)}')
    

if __name__ == "__main__":
    main()
