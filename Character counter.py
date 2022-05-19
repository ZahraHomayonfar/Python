def countchar(n,a): 
    count = 0
    for letter in a:
        if letter == n: 
            count += 1
    return count
while True:
    a1 = input("please enter a word: ")
    if a1 == "done":
        break
    else:
        n1 = input("please enter a letter: ")
        print(countchar(n1,a1))
