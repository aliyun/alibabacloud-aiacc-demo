#!/usr/bin/env python

import argparse
import ncluster
import os
import time
from ncluster import ncluster_globals

# setting parameters
INSTANCE_TYPE = 'ecs.gn6v-c8g1.2xlarge'
NUM_GPUS = 1

ncluster.set_backend('aliyun')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='fastgpu-gtc-demo-f',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
args = parser.parse_args()

def main():
  print('start job ...')
  start_time = time.time()

  # 1. create infrastructure
  supported_regions = ['cn-huhehaote', 'cn-shanghai', 'cn-zhangjiakou', 'cn-hangzhou', 'cn-beijing']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()} (set $ALYUN_DEFAULT_REGION)"
  
  ncluster_globals.set_should_disable_nas(True)

  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          instance_type=INSTANCE_TYPE,
                          disable_nas=True,
                          spot=True,
                          install_script='') 

  init_ncluster = time.time()
  print('init ncluster:', init_ncluster - start_time)

  # 2. upload GTC code
  job.run('yum install -y unzip')
  job.upload('GTC')
  job.run('cd GTC && wget http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/gtc-demo/dataset.zip ' + 
                 '&& wget http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/gtc-demo/test.JPG ' + 
                 '&& wget http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/gtc-demo/resnet50-19c8e357.pth ' +
                 '&& conda activate torch_1.3_cu10.0_py36') 
  upload_data = time.time()
  print('upload_data time:', upload_data - init_ncluster)

  # 3. prepare the dataset
  job.run('unzip -o dataset.zip')
  unzip_time = time.time()
  print('unzip data:', unzip_time - upload_data)

  # 4. run the training job
  job.tasks[0].run('conda activate torch_1.3_cu10.0_py36')
  job.tasks[0].run('./run-perseus.sh 2>&1 | tee logs.log', non_blocking=False)
  train_time = time.time()
  print('training time:', train_time - unzip_time)

  # 5. run the inference job
  job.tasks[0].run('python inference.py 2>&1 | tee logs.inference.log', non_blocking=False)
  print('inference time:', time.time() - train_time)

  eclapse_time = time.time() - start_time
  print(f'training and inference deploy time is: {eclapse_time} s.')

  # 6. stop the instance (optional)
  job.stop()

if __name__ == '__main__':
  main()

