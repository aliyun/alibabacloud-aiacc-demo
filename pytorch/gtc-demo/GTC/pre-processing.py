from PIL import Image
import os
import os.path
import glob
import time

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

for classfile in glob.glob("test-mini-dataset/*"):
    for jpgfile in glob.glob(classfile+"/*.JPG"):
        save_path = "mini-dataset/" + os.path.basename(classfile)
        os.system('mkdir -p ' + save_path)
        convertjpg(jpgfile, save_path)

print('cost time:', time.time() - start_time)
print('finish!')
