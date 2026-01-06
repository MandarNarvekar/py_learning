print ("Hello world!")

f=open("Sample.txt","r")
content = f.read()
print("Original File content: ")
print(content)
f.close()


f=open("Sample1.txt","w") 
f.write(content)
f.close()

f=open("Sample1.txt","a")
f.write("\nCopied file.")
f.close()

f=open("Sample1.txt","r")
content = f.read()
print("Copied File content: ")
print(content)
f.close()

#for i in range(10):
#    filename = f"file_{i}.txt"
#    with open(filename, "w") as f:
#        f.write(f"This is file number {i}\n")   
#        f.close()

    


         