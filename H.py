import os,shutil

imspath = r'tmp/ims'
lbspath = r'tmp/lbs'

count = 0
for i in range(100):
    for img_name in os.listdir(imspath):
        lb_name = img_name.replace('jpg','txt')
        img_path = os.path.join(imspath,img_name)
        lb_path = os.path.join(lbspath,lb_name)

        shutil.copy(img_path, os.path.join('train_datas/images', str(count) + '.jpg'))
        shutil.copy(lb_path, os.path.join('train_datas/labels', str(count) + '.txt'))
        count+=1