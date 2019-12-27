#!/usr/bin/env python

import argparse
import ncluster
import os
import time
from ncluster import ncluster_globals

# setting parameters
INSTANCE_TYPE = 'ecs.gn6v-c10g1.20xlarge'
NUM_GPUS = 8

ncluster.set_backend('aliyun')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='fastgpu-perseus-bert',
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
                          num_tasks=args.machines,
                          disable_nas=True,
                          spot=True,
                          instance_type=INSTANCE_TYPE)

  # 2. Upload perseus bert code.
  job.run('yum install -y unzip')
  job.upload('perseus-bert')
  job.run('conda activate tensorflow_1.14_cu10.0_py36')

  # 3. Download pretrain model and dataset.
  BERT_CHINESE_BASE_DIR = '/root/chinese_L-12_H-768_A-12'
  DATA_DIR = '/root/toutiao_data'
  job.run('wget -c -t 10 https://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/chinese_L-12_H-768_A-12.zip  && unzip chinese_L-12_H-768_A-12.zip')
  job.run('wget -c -t 10 https://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/toutiao_data.tgz && tar xvf toutiao_data.tgz')

  
  # 4. Run the training job.
  job.run('cd perseus-bert')
  hosts = [task.ip + f':{NUM_GPUS}' for task in job.tasks]
  host_str = ','.join(hosts)

  mpi_cmd = ['mpirun --allow-run-as-root',
            f'-np {args.machines * NUM_GPUS}',
            f'--npernode {NUM_GPUS}',
            f'--host {host_str}',
            '--bind-to none',
            '-x NCCL_DEBUG=INFO',
            '-x PATH',
            '-x PYTHONPATH', 
            '-x LD_LIBRARY_PATH',
            '-x XLA_FLAGS']

  bert_classifier_cmd = ['python run_classifier.py', 
                         '--task_name=news',
                         '--do_train=true',
                         '--do_eval=true',
                         f'--data_dir={DATA_DIR}',
                         f'--vocab_file={BERT_CHINESE_BASE_DIR}/vocab.txt',
                         f'--bert_config_file={BERT_CHINESE_BASE_DIR}/bert_config.json',
                         f'--init_checkpoint={BERT_CHINESE_BASE_DIR}/bert_model.ckpt',
                         '--max_seq_length=128',
                         '--train_batch_size=48',
                         '--learning_rate=8e-5',
                         '--num_train_epochs=3.0',
                         '--warmup_proportion=0.8',
                         '--output_dir=/root/output_dir',
                         '--use_amp=true',
                         '--use_perseus=true',
                         '--use_xla=true']
  
  cmd = mpi_cmd + bert_classifier_cmd
  cmd = " ".join(cmd)
  job.tasks[0].run(f'echo {cmd} > {job.logdir}/task-cmd')
  job.tasks[0].run(cmd, non_blocking=True)
  print(f"Logging to {job.logdir}")

  eclapse_time = time.time() - start_time
  print(f'training deploy time is: {eclapse_time} s.')

  job.stop()
    
if __name__ == '__main__':
  main()
