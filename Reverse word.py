def reverse(n):
    b = ""
    a = len(n)
    for i in range(a-1,-1,-1):
        b += n[i]
    return b
while True:
    x = input("please enter a word: ")
    if x == "done": break
    else:
        output = reverse(x)
        print("your reverse word is: ",output)
