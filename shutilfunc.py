import shutil
#shutil.copy("anil/kumar.txt", "anil/kumar_copy.txt")
#shutil.move("anil/kumar_copy.txt", "anil/kumar_moved.txt")
shutil.rmtree("anil")
shutil.make_archive("anil_archive", 'zip', "anil")
shutil.unpack_archive("anil_archive.zip", "anil_unpacked", 'zip')
 