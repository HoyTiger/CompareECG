import os
import matplotlib.pyplot as plt

def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize / float(1024)
    return round(fsize, 3)

sizes = []
for root, dirs, files in os.walk('/Volumes/HOY‘s DISK/20220115170517'):
    for file in files:
        if file.endswith('.xml'):
            path = os.path.join('/Volumes/HOY‘s DISK/20220115170517', file)
            sizes.append(get_FileSize(path))

plt.hist(sizes)
plt.show()