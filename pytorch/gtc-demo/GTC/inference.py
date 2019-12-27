import cv2
import torch
import torchvision
from torchvision import transforms as transforms
import PIL.Image as Image

#model = torchvision.models.alexnet(pretrained=False)
#model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
model = torchvision.models.resnet50(pretrained=False)
model.eval()
model.fc = torch.nn.Linear(2048, 3)

model.load_state_dict(torch.load('save_model.pth'))

device = torch.device('cuda')
model = model.to(device)

import cv2
import numpy as np

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess_2(img):
    x = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean/255.0, stdev/255.0),
        ])(img)
    x = x[None, ...]
    return x
import datetime
import glob
img_list = []
import sys
print(str(sys.argv))
if sys.argv[1] == '0':
    img_list = glob.glob('test-mini-dataset/scissors/IMG_*.JPG')
elif sys.argv[1] == '1':    
    img_list = glob.glob('test-mini-dataset/rock/IMG_*.JPG')
else:    
    img_list = glob.glob('test-mini-dataset/paper/IMG_*.JPG')

count = 0
count_paper = 0
count_rock = 0
count_scissors = 0
res = []
for img_file in img_list:
    count += 1
    img = Image.open(img_file)
    x = preprocess_2(img).to(device)
    print(x.shape)
    begin = datetime.datetime.now()
    y = model(x)
    import torch.nn.functional as F
    y = F.softmax(y, dim=1)
    
    predict = y.argmax(1)
    print(predict)

    a=[float(y.flatten()[0]),float(y.flatten()[1]),float(y.flatten()[2])]
    end = datetime.datetime.now()
    k = end - begin
    m = np.where(a==np.max(a))
    p_paper = float(y.flatten()[0])
    p_rock = float(y.flatten()[1])
    p_scissors = float(y.flatten()[2])

    print("布的概率："+str(float(y.flatten()[0])))
    print("石头的概率："+str(str(float(y.flatten()[1]))))
    print("剪刀的概率："+str(float(y.flatten()[2])))
    label_ = int(m[0][0])
    label_ = predict
    assert predict == int(m[0][0])
    if label_ == 0:
        res.append(img_file)
    if label_ == 0:
        filename = "test.JPG"
        count_paper += 1
        print("你出的是布")

    if label_ == 1:
        filename="test2.JPG"
        count_rock += 1
        print("你出的是石头")

    if label_ == 2:
        filename="test1.JPG"
        count_scissors += 1
        print("你出的是剪刀")
    print("推理时间："+str(k.total_seconds()*1000)+"毫秒")

print('paper acc:', 1.0 * count_paper / count) #len(img_list))
print('rock acc:', 1.0*count_rock / count) #len(img_list))
print('scissors acc:', 1.0*count_scissors / count) #len(img_list))

print('res:', res)
