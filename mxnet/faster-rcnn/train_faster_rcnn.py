#!/usr/bin/env python

import argparse
import fastgpu
import os
import time
import sys

from fastgpu import fastgpu_globals
fastgpu_globals.set_should_disable_nas(True)

INSTANCE_TYPE = 'ecs.gn6v-c8g1.2xlarge' # V100
#INSTANCE_TYPE = 'ecs.gn6v-c8g1.16xlarge'
#INSTANCE_TYPE = 'ecs.gn5-c8g1.14xlarge'
NUM_GPUS = 1
IMAGE_TYPE = 'aiacc'
CONDA_ENVS = [
  "mxnet_1.4.1_cu10.0_py36",
  "mxnet_1.4.1_cu10.1_py36",
  "mxnet_1.5.0_cu10.0_py36",
  "mxnet_1.5.0_cu10.1_py36",
  "mxnet_1.6.0_cu10.1_py36",
  "mxnet_1.6.0_cu10.2_py36",
  "mxnet_1.7.0_cu10.0_py36",
  "mxnet_1.7.0_cu10.1_py36",
  "mxnet_1.7.0_cu10.2_py36",
  "mxnet_1.9.0_cu10.1_py36",
  "mxnet_1.9.0_cu10.2_py36",
  "mxnet_1.9.0_cu11.0_py36"
]

fastgpu.set_backend('aliyun')
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
  assert fastgpu.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {fastgpu.get_region()} (set $ALYUN_DEFAULT_REGION)"
  
  fastgpu_globals.set_should_disable_nas(True)

  job = fastgpu.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          #image_name='aiacc-dlimg-centos7:1.3.0.post3',
                          num_tasks=args.machines,
                          instance_type=INSTANCE_TYPE,
                          spot=True,
                          disable_nas=True,
                          image_type=IMAGE_TYPE
                          )
  # 2. Upload perseus faster-rcnn code.
  job.upload('gluon-cv')
  job.run('conda activate mxnet_1.9.0_cu11.0_py36')

  # 2.5(alternative) install nccl-2.9.6
  # job.run("sudo apt update")
  # job.run("sudo apt install -y software-properties-common")
  # job.run("wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin")
  # job.run("sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600")
  # job.run("sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub")
  # job.run('sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"')
  # job.run("sudo apt-get update")
  # job.run("sudo apt install -y libnccl2=2.9.6-1+cuda11.0 libnccl-dev=2.9.6-1+cuda11.0")


  # 2.5(old) nccl install
  job.run("wget https://ali-perseus-release.oss-cn-huhehaote.aliyuncs.com/AIACC-Dev/zijian/nccl_2.9.6-1%2Bcuda11.0_x86_64.txz -O nccl_2.9.6-1+cuda11.0_x86_64.txz")
  job.run("tar -Jxvf nccl_2.9.6-1+cuda11.0_x86_64.txz")
  job.run("cp -f nccl_2.9.6-1+cuda11.0_x86_64/include/*.h /usr/local/cuda/include/")
  job.run("cp -f nccl_2.9.6-1+cuda11.0_x86_64/lib/libnccl* /usr/local/cuda/lib64/")
 
  # 3. Download pretrain model and dataset.
  job.run('if [ ! -d /root/mscoco ];then mkdir /root/mscoco;fi')
  job.run('cd /root/mscoco && wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/annotations/annotations_trainval2017.zip')
  job.run('wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/zips/train2017.zip')
  job.run('wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/zips/test2017.zip')
  job.run('wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/coco2017/zips/val2017.zip')

  job.run('mkdir -p /root/.mxnet/models')
  job.run('cd /root/.mxnet/models && wget -c -t 10 http://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/pretrain_model/resnet50_v1b-0ecdba34.params')

  # 4. install requirements.
  job.run('chmod -R 744 /root/gluon-cv/')
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
  job.tasks[0].run(cmd)
  print(f"Logging to {job.logdir}")

  eclapse_time = time.time() - start_time
  print(f'training deploy time is: {eclapse_time} s.')


if __name__ == '__main__':
  main()

