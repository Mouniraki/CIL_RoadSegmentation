import os
from os import listdir
from os.path import isfile, join
import re
from PIL import Image

mypath = './training/groundtruth'
all_images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for img_path in all_images:
    im = Image.open('./training/groundtruth/' + img_path).convert("L")
    im.save('./training/groundtruth/' + img_path + '.png')

exit()
mypath = './training/images/'
all_images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for img_path in all_images:
    if 'png' in img_path:
        os.remove('./training/images/'+img_path)
        continue
    #im = Image.open('./training/images/' + img_path)
    #im.save('./training/images/' + img_path + '.png')
exit()




exit()
mypath = './training/groundtruth'

all_images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
next_id = 0
for img_path in all_images:
    img_new_name = 'satimage_'+ str(next_id)

    im = Image.open('./training/groundtruth/'+img_path[:-4]+'.jpg')
    im.save('./training/groundtruth/'+img_path[:-4]+'.png')

    im = Image.open('./training/images/'+img_path[:-4]+'.jpg')
    im.save('./training/images/'+img_path[:-4]+'.png')


    os.rename('./training/groundtruth/'+img_path, './training/groundtruth/'+img_new_name)
    os.rename('./training/images/'+img_path, './training/images/'+img_new_name)


    next_id += 1




exit()


mypath = './training/'

all_images = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for img_path in all_images:
    if 'mask' in img_path:
        target_folder = 'groundtruth/'
    else:
        target_folder = 'images/'
    numbers = [int(s) for s in re.findall(r'\d+', img_path)]
    assert len(numbers) == 1
    img_new_name = 'satimage_'+ str(numbers[0])
    os.rename('./training/'+img_path, './training/'+target_folder+img_new_name)