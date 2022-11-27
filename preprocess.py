from os import listdir
from os.path import isfile, join
from PIL import Image
import PIL

mypath = 'D:/UCI/1-Q1/Deep Generative Model/Final Project/CS274E-EigenGAN/data/test'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(files)

new_path = 'D:/UCI/1-Q1/Deep Generative Model/Final Project/CS274E-EigenGAN/data/new-test'
for f in files:
    img = Image.open(join(mypath, f))
    im1 = img.resize((64, 64))
    im1.save(join(new_path, f))