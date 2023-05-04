from PIL import Image
import os
import os.path
import glob
import time
from torchvision import transforms as transforms

start_time = time.time()

def convertjpg(jpgfile,outdir,width=224,height=224):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        save_img = os.path.join(outdir, os.path.basename(jpgfile))
        new_img.save(save_img)
        print('save', jpgfile, 'to', save_img, ' done.')
    except Exception as e:
        print(e)
def enhance(jpgfile, outdir):
    im = Image.open(jpgfile)
    # 进行随机的灰度化
    new_im = transforms.RandomGrayscale(p=0.5)(im)    # 以0.5的概率进行灰度化
    save_gray = os.path.join(outdir, 'gray_' + os.path.basename(jpgfile))
    new_im.save(save_gray)
    # 色度、亮度、饱和度、对比度的变化
    new_im = transforms.ColorJitter(brightness=1)(im)
    new_im = transforms.ColorJitter(contrast=1)(im)
    new_im = transforms.ColorJitter(saturation=0.5)(im)
    new_im = transforms.ColorJitter(hue=0.5)(im)
    save_color = os.path.join(outdir, 'color_' + os.path.basename(jpgfile))
    new_im.save(save_color)
    # 随机角度旋转
    new_im = transforms.RandomRotation(45)(im)    #随机旋转45度
    save_rotate = os.path.join(outdir, 'rotate_' + os.path.basename(jpgfile))
    new_im.save(save_rotate)
    # 随机水平/垂直翻转
    new_im = transforms.RandomHorizontalFlip(p=1)(im)   # p表示概率
    save_hor = os.path.join(outdir, 'hor_'+os.path.basename(jpgfile))
    new_im.save(save_hor)
    new_im = transforms.RandomVerticalFlip(p=1)(im)
    save_ver = os.path.join(outdir, 'ver_' + os.path.basename(jpgfile))
    new_im.save(save_ver)

def merge_file(from_path, to_path):
    os.system('mkdir -p ' + to_path + '/scissors')
    os.system('mkdir -p ' + to_path + '/rock')
    os.system('mkdir -p ' + to_path + '/paper')
    os.system('cp ' + from_path + '/scissors/* ' + to_path + '/scissors/')
    os.system('cp ' + from_path + '/rock/* ' + to_path + '/rock/')
    os.system('cp ' + from_path + '/paper/* ' + to_path + '/paper/')

# main process
import sys
stage = 1 if len(sys.argv) <= 1 else int(sys.argv[1])
raw_dataset = "mini-rps-dataset" if len(sys.argv) <= 2 else sys.argv[2]
enhance_dataset = "enhance-mini-rps-dataset" if len(sys.argv) <= 2 else sys.argv[3]
merge_from_dataset = "enhance-test-mini-dataset" if len(sys.argv) <= 2 else sys.argv[2]
merge_to_dataset = "merge-dataset" if len(sys.argv) <= 2 else sys.argv[3]

if stage <= 1:
    for classfile in glob.glob(raw_dataset + "/*"):
        for file_type in ['jpg', 'png', 'JPG', 'PNG']:
            for jpgfile in glob.glob(classfile+"/*." + file_type):
                save_path = enhance_dataset + "/" + os.path.basename(classfile)
                os.system('mkdir -p ' + save_path)
                #convertjpg(jpgfile, save_path)
                enhance(jpgfile, save_path)
else:       
    merge_file(merge_from_dataset, merge_to_dataset) 

print('cost time:', time.time() - start_time)
print('finish!')
