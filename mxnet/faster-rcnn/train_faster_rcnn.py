#!/usr/bin/env python

import argparse
import ncluster
import os
import time

from ncluster import ncluster_globals
ncluster_globals.set_should_disable_nas(True)

INSTANCE_TYPE = 'ecs.gn6v-c10g1.20xlarge' # V100
#INSTANCE_TYPE = 'ecs.gn6v-c8g1.16xlarge'
#INSTANCE_TYPE = 'ecs.gn5-c8g1.14xlarge'
NUM_GPUS = 8

ncluster.set_backend('aliyun')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='perseus-faster-rcnn',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
args = parser.parse_args()

def main():
  start_time = time.time()
  # 1. Create infrastructure
  supported_regions = ['cn-huhehaote', 'cn-zhangjiakou', 'cn-shanghai', 'cn-hangzhou', 'cn-beijing']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()} (set $ALYUN_DEFAULT_REGION)"
  
  ncluster_globals.set_should_disable_nas(True)

  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          #image_name='aiacc-dlimg-centos7:1.3.0.post3',
                          num_tasks=args.machines,
                          instance_type=INSTANCE_TYPE,
                          spot=True,
                          disable_nas=True,
                          )
  # 2. Upload perseus faster-rcnn code.
  job.upload('gluon-cv')
  job.run('conda activate mxnet_1.5.1.post0_cu10.0_py36')
 
  # 3. Download pretrain model and dataset.
  job.run('mkdir /root/mscoco')
  job.run('cd /root/mscoco && wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/annotations/annotations_trainval2017.zip')
  job.run('wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/zips/train2017.zip')
  job.run('wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/zips/test2017.zip')
  job.run('wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/zips/val2017.zip')

  job.run('mkdir -p /root/.mxnet/models')
  job.run('cd /root/.mxnet/models && wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/pretrain_model/resnet50_v1b-0ecdba34.params')

  # 4. install requirements.
  job.run('cd /root/gluon-cv/')
  job.run('pip install -r requirements.txt')
  
  job.run('python mscoco.py')

  # 5. Run the training job.
  hosts = [task.ip + f':{NUM_GPUS}' for task in job.tasks]
  host_str = ','.join(hosts)

  mpi_cmd = ['mpirun --allow-run-as-root',
            f'-np {args.machines * NUM_GPUS}',
            f'--npernode {NUM_GPUS}',
            f'--host {host_str}',
            '--bind-to none',
            '-x NCCL_DEBUG=INFO',
            '-x PATH',
            '-x LD_LIBRARY_PATH',]

  insightface_cmd = './train-perseus.sh'
 
  cmd = mpi_cmd 
  cmd = " ".join(cmd) + " " + insightface_cmd
  job.tasks[0].run(f'echo {cmd} > {job.logdir}/task-cmd')
  job.tasks[0].run(cmd, non_blocking=True)
  print(f"Logging to {job.logdir}")

  eclapse_time = time.time() - start_time
  print(f'training deploy time is: {eclapse_time} s.')


if __name__ == '__main__':
  main()

