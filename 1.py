print("Hello world!")

with open("Sample.txt", "r") as file:
    content = file.read()
    print("File content:")
    print(content)