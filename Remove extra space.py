while True:
    word = input("please enter your name(The first letter of the first name and last name should be uppercase): ")
    if word == "done": break
    else:
        word = word.strip()
        a = word.find(' ')
        b = len(word)
        if a == -1:
            for i in range(1,b):
                if word[i].isupper():
                    FirstName = word[0:i]
                    LastName = word[i:b]
                    print(FirstName + " " + LastName)
        else:
            FirstName = word[0:a]
            LastName = word[a:b].strip()
            print(FirstName + " " + LastName)