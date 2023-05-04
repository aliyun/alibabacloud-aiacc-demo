#!/usr/bin/env python

import argparse
import fastgpu
import os
import time
from fastgpu import fastgpu_globals

# setting parameters
INSTANCE_TYPE = 'ecs.gn6v-c8g1.16xlarge'
NUM_GPUS = 8
IMAGE_TYPE = 'aiacc'

fastgpu.set_backend('aliyun')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='fastgpu-gtc-demo',
                    help="name of the current run, used for machine naming and tensorboard visualization")

args = parser.parse_args()

def main():
  print('start job ...')
  start_time = time.time()

  # 0. setup 
  supported_regions = ['cn-huhehaote', 'cn-shanghai', 'cn-zhangjiakou', 'cn-hangzhou', 'cn-beijing', 'cn-shenzhen']
  assert fastgpu.get_region() in supported_regions, f"required AMI {IMAGE_TYPE} has only been made available in regions {supported_regions}, but your current region is {fastgpu.get_region()} (set $ALYUN_DEFAULT_REGION)"
  fastgpu_globals.set_should_disable_nas(True)
  
  # 1. create a job
  job = fastgpu.make_job(name=args.name,
                          run_name=f"{args.name}-1",
                          num_tasks=1,
                          instance_type=INSTANCE_TYPE,
                          image_type=IMAGE_TYPE,
                          disable_nas=True,
                          spot=True,
                          install_script='') 

  init_fastgpu = time.time()
  print('init fastgpu: %.2fs'%(init_fastgpu - start_time))

  # 2. upload GTC code
  job.run('apt install -y unzip')
  job.upload('GTC')
  job.run("chmod -R 744 GTC")
  job.run('cd GTC && wget http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/gtc-demo/dataset.zip ' + 
                 '&& wget http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/gtc-demo/test.JPG ' + 
                 '&& wget http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/gtc-demo/resnet50-19c8e357.pth ') 
  upload_data = time.time()
  print('upload_data time: %.2fs'%(upload_data - init_fastgpu))

  # 3. prepare the dataset
  job.run('unzip -o dataset.zip')
  unzip_time = time.time()
  print('unzip data: %.2fs'%(unzip_time - upload_data))

  # 4. run the training job
  job.run('conda init bash && conda activate torch_1.3.0_cu10.0_py36')

  job.run('pip install opencv-python')
  job.run("pip install 'pillow<7.0.0'")

  job.tasks[0].run('./run-perseus.sh', show_realtime=True)
  train_time = time.time()
  print('training time: %.2f'%(train_time - unzip_time))

  # 5. run the inference job
  job.tasks[0].run('python inference.py', show_realtime=True)
  print('inference time: %.2fs'%(time.time() - train_time))

  eclapse_time = time.time() - start_time
  print(f'training and inference deploy time is: %.2fs.'%eclapse_time)

  # 6. stop the instance
  job.stop()

  # 7. kill the instance (optional) 
  job.kill()

if __name__ == '__main__':
  main()

