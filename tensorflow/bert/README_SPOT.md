# 一键部署并训练 bert-perseus 应用

## 简介

bert-perseus 为阿里云 AI 加速库 aiacc 支持分布式训练的 bert 新闻分类任务代码。其中数据集，您可以查看[中文新闻（文本）分类数据集](https://github.com/fatecbf/toutiao-text-classfication-dataset)

Spot实例 为抢占式实例相对于按量付费有较大优惠，您能稳定持有实例一小时。之后当市场价格高于您的出价或者资源供需关系变化时，实例会被自动释放，请做好数据备份工作。
本教程介绍了使用 FastGPU 从 0 开始，一键部署 Spot 实例并训练 bert-perseus 应用。

<tutorial-nav></tutorial-nav>

## 环境准备

**说明**：Cloud Shell 已经预装了 FastGPU 工具，建议您直接使用 [Cloud Shell](https://shell.aliyun.com)

首先需要进行环境部署的大区选择，您可以根据需要设置对应的 Region，以 `cn-beijing` 为例: 

```bash
export ALIYUN_DEFAULT_REGION=cn-beijing
```

您可以通过下述命令方便的查询支持的 Region 列表：

```bash
aliyun ecs DescribeRegions --output cols=RegionId,LocalName rows=Regions.Region
```

然后，您需要下载 bert-perseus 源码，该源码是使用阿里云 Ai 加速引擎库 aiacc 优化的 bert 分布式训练。

```bash
git clone https://github.com/aliyun/alibabacloud-aiacc-demo.git
cd alibabacloud-aiacc-demo/tensorflow/bert/
```

## 启动 FastGPU 任务 

所有任务都是通过 train_news_classifier_spot.py 一个脚本来完成，以下步骤是可选的修改配置参数操作，可以使用默认参数进行一键启动.

1. 配置机器型号

    默认是采用单机 8 卡 v100 的 spot 机器 `ecs.gn6v-c10g1.20xlarge`;
可以修改 `INSTANCE_TYPE` 变量，以及 spot 配置，比如改为单机单卡 v100 的 spot 机器 `ecs.gn6v-c8g1.2xlarge`，参见 train_news_classifier_spot_1.py 。

2. 配置ECS实例名称

    修改 `args.name`，默认是 `fastgpu-perseus-bert`

3. 启动任务

    运行下述命令，即可启动完整过程，包括购买 ECS 实例、安装 GPU 深度学习环境、部署 bert 项目、启动 training

    ```py
    python train_news_classifier_spot.py
    ```

这里使用了 阿里AI加速库Aiacc 支持的分布式训练，具体超参数的调整您可以修改 `bert_classifier_cmd` 的相关配置;

脚本运行成功显示如下:

```
Logging to /ncluster/runs/perseus-bert-1
training deploy time is: 196.9283847808838 s.
```

在使用单机 8 卡 V100 的情况下，accuracy 可达 0.888，部署时间 3.5min，训练时间 11.5min，从创建实例到最后的训练过程总体耗时 15 分钟，单机单卡 V100 总耗时 50 分钟。


## FastGPU任务解释

下面对FastGPU启动任务脚本fastgpu.py进行分步解释

1. ECS机器的选型/购买

    ```
    INSTANCE_TYPE = 'ecs.gn6v-c10g1.20xlarge' 
    NUM_GPUS = 8
    parser.add_argument('--name', type=str, default='fastgpu-perseus-bert',
                       help="name of the current run, used for machine naming and tensorboard visualization")
    parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
    args = parser.parse_args()
    ```

2. GPU环境的自动化部署

    ```
    job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE)
    ```

3. python特定深度学习环境激活

    ```
    job.run('conda activate tensorflow_1.14_cu10.0_py36')
    ```

4. 上传项目和代码
    
    ```
    # 2. Upload perseus bert code.
    job.run('yum install -y unzip')
    job.upload('perseus-bert')
  
    # 3. Download pretrain model and dataset.
    BERT_CHINESE_BASE_DIR = '/root/chinese_L-12_H-768_A-12'
    DATA_DIR = '/root/toutiao_data'
    job.run('wget -c -t 10 https://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/chinese_L-12_H-768_A-12.zip  && unzip chinese_L-12_H-768_A-12.zip')
    job.run('wget -c -t 10 https://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/toutiao_data.tgz && tar xvf toutiao_data.tgz')
    ```

5. 运行训练和推理

    ```
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
    ```

6. 自动关机停止计费

    由于涉及到ECS资源的计费,所以默认结束任务后会关机
    ```
    # 6. stop the instance (optional)
    job.stop()
    ```

## FastGPU快捷使用

上述通过 `train_news_classifier.py` 脚本进行 FastGPU 的使用是 runtime 的方式。

此外,FastGPU 还提供便捷的 `ecluster` 命令行进行云上资源的生命周期，查看运行过程的日志和日常开发等。

上述创建的任务完成后，您可以通过下述命令查看实例状态：

```bash
ecluster ls
```

执行后，您可以看到具体的实例状态，如下所示：

```
region=cn-beijing
Using region  cn-beijing
Using region  cn-beijing
------------------------------------------------------------------------------------------------------------------------
name            hours_live   instance_type       public_ip       key/owner      private_ip     instance_id
------------------------------------------------------------------------------------------------------------------------
task0.gtc-perseus-bert-spot        3.9 ecs.gn6v-c10g1.20xlarge  39.105.180.135  ncluster-shell   192.168.8.129 i-2zeetszs0750hqeuzjm8
```

最后您可以参考 FastGPU 常用命令，对您的云上资源进行管理。

- list instances of current user.

    ```
    ecluster ls
    ```

- login to specified instance

    ```
    ecluster ssh {task name}
    ```

- stop specified instance

    ```
    ecluster stop {task name}
    ```

- start specified instance

    ```
    ecluster start {task name}
    ```

- kill specified instance

    ```
    ecluster kill {task name}
    ```

- ssh and tmux attach to running task. If there is no tmux session, will fall back to ssh.

    ```
    ecluster tmux {task name}
    ```

- mount nas to /ncluster for specified instance.

    ```
    ecluster mount {task name}
    ```

- scp file/folder from souce to destination.

    ```
    ecluster scp {source} {destination}
    ```

- create instance by a config file.

    ```
    ecluster create --config create.cfg
    ```

- create instance by parameters

    ```
    ecluster create --name ncluster1 --machines 1 ...
    ```

- add the public ip of current machine  to security group for the specific task.

    ```
    ecluster addip {task name}
    ```

- rename name of specified instance

    ```
    ecluster rename {old_name} {new_name}
    ```
