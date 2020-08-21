# Object Detection using faster-rcnn .

## prerequisite
* Aliyun account
* ncluster package installed.

## Training
* 1.Register your aliyun acount using below command.
```Bash
export ALIYUN_ACCESS_KEY_ID=xxxxx
export ALIYUN_ACCESS_KEY_SECRET=xxxxx
export ALIYUN_DEFAULT_REGION=cn-beijing
```

* 2.Run the training job with
```Bash
python train_faster_rcnn.py
```
After the training job deployed to cloud, the console display as followed.
```Bash
Logging to /ncluster/runs/perseus-faster-rcnn-1
training deploy time is: xxxs.
```

* 3.Use `ecluster ls` to display the cloud machine.

* 4.Attach to running console using `ecluster tmux task0.perseus-faster-rcnn`. 

## Time
The deploy time is about 35 min, the training time is about 15min per epoch, The total one epoch time is about 50 min.


