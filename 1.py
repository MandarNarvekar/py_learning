print("Hello world!")

with open("Sample.txt", "r") as file:
    content = file.read()
    print("File content:")
    print(content)

 
f = open('output.txt', 'r')
data = f.read()
print(data)
f.close()
 