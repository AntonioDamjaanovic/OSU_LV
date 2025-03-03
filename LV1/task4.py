def filter_song(filename):
    words_in_song = {}

    try:
        with open(filename, 'r', encoding="utf-8") as file:
            for line in file:
                line = line.rstrip()
                words = line.split(" ")
                for word in words:
                    word = word.strip(',.;:!?')

                    if word not in words_in_song:
                        words_in_song[word] = 1
                    else:
                        words_in_song[word] += 1

    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None
    
    return words_in_song


def main():
    filename = 'song.txt'
    words_in_song = filter_song(filename)

    if words_in_song is None:
        return "There are no words in file."
    
    words_once = [key for key, value in words_in_song.items() if value == 1]

    print(f"\nWords in song are: {words_in_song}")
    print(f'\nNumber of words that appear only once: {len(words_once)}')
    print(f"These words are: {words_once}")


if __name__ == "__main__":
    main()