import os

#os.mkdir('Mandar')

#os.rename('Mandar', 'MandarRenamed')

#os.mkdir('MandarRenamed/Subfolder')
#os.mkdir('MandarRenamed/Subfolder/Sample.txt')

#os.rmdir('MandarRenamed/Subfolder/Sample.txt')
#os.rmdir('MandarRenamed/Subfolder')

#print(os.listdir("."))

import os
 
#os.mkdir("amar")
 
#os.rename("amar", "anil")
 
#os.mkdir("anil/kumar.txt")
 
print(os.listdir("."))
print("Current Working Directory:", os.getcwd())
print(os.path.exists("anil/kumar.txt"))
 
 
print(os.path.isfile("anil/kumar.txt"))
print(os.path.isdir("anil/kumar.txt"))
print(os.path.splitext("anil/kumar.txt"))
print(os.path.join("anil", "kumar.txt"))
print(os.path.basename("anil/kumar.txt"))
print(os.path.dirname("anil/kumar.txt"))
print(os.path.getsize("anil/kumar.txt"))
print(os.path.abspath("anil/kumar.txt"))
print(os.path.getctime("anil/kumar.txt"))
print(os.path.getmtime("anil/kumar.txt"))
print(os.path.getatime("anil/kumar.txt"))
print(os.path.samefile("anil/kumar.txt", "anil/kumar.txt"))
 
