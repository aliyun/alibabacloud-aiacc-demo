#!/usr/bin/env python

import argparse
import ncluster
import os
import itertools
import time

JOB_NAME = 'fastgpu-aiacc-image-classification'
IMAGE_NAME = 'aiacc-dlimg-centos7:1.3.2'
INSTANCE_TYPE = 'ecs.gn6v-c10g1.20xlarge'
MACHINES = 1

# conda environment available
env_names = [
'tensorflow_1.12_cu9.0_py27',
'tensorflow_1.12_cu9.0_py36',
'tensorflow_1.14_cu10.0_py27',
'tensorflow_1.14_cu10.0_py36',
'tensorflow_1.15_cu10.0_py27',
'tensorflow_1.15_cu10.0_py36',
'tensorflow_2.1_cu10.1_py27',
'tensorflow_2.1_cu10.1_py36']


supported_regions = ['cn-huhehaote', 'cn-zhangjiakou', 'cn-shanghai', 'cn-hangzhou', 'cn-beijing']
assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()} (set $ALYUN_DEFAULT_REGION)"

ncluster.set_backend('aliyun')
parser = argparse.ArgumentParser()

# FastGPU and instance relevant
parser.add_argument('--name', type=str, default=JOB_NAME,
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=MACHINES,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default=INSTANCE_TYPE,
                    help='choose the type of instance, make sure you have access and enough quota for this type.')

# Running environment relevant
parser.add_argument('--gpus', type=int, default=8,
                    help="how many gpu you want to use in one machine")
parser.add_argument('--env', type=str, default='1.15',
                    help='version of tensorflow. 1.12, 1.14, 1.15 or 2.1.')
parser.add_argument('--py2',action='store_true',
                    help='flag to use python2')
parser.add_argument('--hvd', action='store_true',
                    help="flag to use horovod as variable_update instead of perseus.")

# tfbenchmark relevant
parser.add_argument('--data_dir', type=str, default="/data/imagenet-dataset/imagenet")
parser.add_argument('--use_ramdisk', type=bool, default=False,
                    help="use ramdisk to store training dataset.")
parser.add_argument('--image_size', type=int, default=224,
                    help='image size for training model')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size per device')
parser.add_argument('--model', type=str, default='resnet50_v1.5',
                    help='model to train')
parser.add_argument('--disable_xla', action='store_true',
                    help="disable XLA auto-jit compilation")



args = parser.parse_args()

def get_mpi_command(np_param, npernode_param, host_param, hvd=False):
    if not hvd:
        mpi_cmd = f'mpirun --allow-run-as-root -np {np_param} --npernode {npernode_param} --host {host_param} ' \
                '--bind-to none ' \
                '-map-by slot ' \
                '-x NCCL_SOCKET_IFNAME=^lo,docker0 ' \
                '-mca btl_tcp_if_exclude lo,docker0 ' \
                '-x PATH -x XLA_FLAGS -x PYTHONPATH -x LD_LIBRARY_PATH ' \
                '-x PERSEUS_ALLREDUCE_STREAMS=6 ' \
                '-x PERSEUS_ALLREDUCE_DISABLE_STREAM_TUNING=1 ' \
                '-x PERSEUS_ALLREDUCE_MODE=1 ' \
                '-x NCCL_MAX_NRINGS=3 '
    else:
        mpi_cmd = f'mpirun --allow-run-as-root -np {np_param} --npernode {npernode_param} --host {host_param} ' \
                '--bind-to none ' \
                '-map-by slot ' \
                '-x NCCL_SOCKET_IFNAME=^lo,docker0 ' \
                '-mca btl_tcp_if_exclude lo,docker0 ' \
                '-x PATH -x XLA_FLAGS -x PYTHONPATH -x LD_LIBRARY_PATH ' \
                '-x NCCL_MAX_NRINGS=3 '
    return mpi_cmd


def gen_benchmark_command_file(benchamrk_command):
    with open('command.sh', 'w') as f:
        f.write('COMMAND=\''+benchamrk_command+'\'\n')
        f.write('if [ $OMPI_COMM_WORLD_RANK -eq 0 ] ; then\n')
        f.write('$COMMAND\n')
        f.write('else\n')
        f.write('$COMMAND >> /dev/null 2>&1\n')
        f.write('fi\n')

def setup_data_disk_env():
    ncluster.ncluster_globals.set_should_disable_nas(True)
    os.environ['FASTGPU_SKIP_COST_CONFORM'] = '1'
    os.environ['ALIYUN_DATA_DISK_SIZE'] = '2048'
    snapshot_id = {"cn-huhehaote": "s-hp339cokv04es714tx43",
                   "cn-zhangjiakou": "s-8vbersrfqlf0cvz7azb1",
                   "cn-shanghai": "s-uf627xzyllp9yisblade",
                   "cn-hangzhou": "s-bp18u85sr8ny1wydcf6i"
                   }

    os.environ['NCLUSTER_ALIYUN_SNAPSHOT_ID'] = snapshot_id[
        ncluster.get_region()]  # imagenet snapshot id at cn-hangzhou


def main():

    # 1. Create Job

    NUM_GPUS = args.gpus

    #setup_data_disk_env()
    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}-{args.machines}",
                            num_tasks=args.machines,
                            image_name=IMAGE_NAME,
                            spot=True,
                            instance_type=args.instance_type)

    print('=============== Instances info ====================================')
    for task in job.tasks:
        print(f'{task.instance.instance_name()} : {task.instance.instance_id()}')
    print('===================================================================')

    #job.run('mkdir /ncluster')
    job.upload('scripts', '/ncluster/scripts')


    # 2. Select Conda Environment
    py = 'py27' if args.py2 else 'py36'
    env_name = None
    for name in env_names:
        if (args.env in name) and (py in name):
            env_name = name
            break
    if env_name:
        job.run(f'conda activate %s'%env_name)
    else:
        job.run('conda activate tensorflow_1.15_cu10.0_py36')


    job.run('pip install mpi4py')
    if args.hvd:
        job.run('conda install -y openmpi')
        job.run('conda install -y gxx_linux-64')
        job.run('HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL pip install --no-cache-dir horovod==0.19.2')

    # To enhance the IO, using ramdis.
    if args.use_ramdisk:
        job.run('cp -r /data/imagenet-dataset/imagenet /dev/shm/')


    # 3. Generate command.
    job.run('cd /ncluster/')
    hosts = [task.ip + f':{NUM_GPUS}' for task in job.tasks]
    host_str = ','.join(hosts)
    print('Host: ', host_str)

    variable_update = 'horovod' if args.hvd else 'perseus'
    xla = not args.disable_xla

    mpi_command = get_mpi_command(f'{args.machines*NUM_GPUS}', f'{NUM_GPUS}', host_str, args.hvd)
    benchmark_cmd = ['python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py',
                     f'--image_size={args.image_size}',
                     f'--model={args.model}',
                     f'--batch_size={args.batch_size}',
                     '--display_every=100',
                     '--data_name=imagenet',
                     '--nodistortions',
                     f'--variable_update={variable_update}',
                     '--num_gpus=1',
                     '--use_fp16=True',
                     '--fp16_gradient=True',
                     '--batch_group_size=4',
                     '--num_inter_threads=8',
                     '--datasets_use_prefetch=True',
                     '--datasets_num_private_threads=5',
                     '--device=gpu',
                     f'--xla={xla}',
                     '--allow_growth=True',
                     '--optimizer=momentum',
                     '--momentum=0.9',
                     '--weight_decay=1e-4',
                     '--use_datasets=False',
                     '--num_eval_epochs=1',
                     '--eval_during_training_every_n_epochs=1',
                     '--num_warmup_batches=500',
                     '--num_batches=1500'
                     ]
                   # '--num_epochs=90',
                   # f'--data_dir={args.data_dir}',
                   # '--data_format=NHWC',
                   # '--resize_method=bicubic',
    gen_benchmark_command_file(" ".join(benchmark_cmd))
    job.upload('command.sh', '/ncluster/')
    cmd = mpi_command + f' bash ./command.sh ' # > machine_{args.machines}_{NUM_GPUS}.log & '

    # 4. Run command
    # job.tasks[0].run(f'echo {cmd} > {job.logdir}/task-0-cmd')
    job.tasks[0].run(cmd, non_blocking=False) #True)
    print(f"Logging to {job.logdir}")

    # 6. stop the instance (optional)
    #job.stop()

if __name__ == '__main__':
    main()
