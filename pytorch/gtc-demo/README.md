# 一键部署并训练手势识别应用

## 简介

GTC-demo 为阿里云在GTC2019中国GTC之夜展示的1小时做出手势识别AI猜拳小程序的展示用例，具体为手势识别imageNet系列模型的 finetune 代码,基于pytorch框架,能够识别剪刀/石头/布的三种手势,训练采用阿里云AI加速库 aiacc 进行性能优化,无缝支持多机分布式训练。其中的数据集由手机拍摄，并经过批量 resize 的手势图片，数据集规模是 173 张图片。

本教程介绍了使用 FastGPU 从 0 开始，一键部署并训练手势识别应用。

<tutorial-nav></tutorial-nav>

## 环境准备

**说明**：Cloud Shell 已经预装了 FastGPU 工具，建议您直接使用 [Cloud Shell](https://shell.aliyun.com)

需要配置您想要创建的资源所在大区，您可以根据需要设置对应的 Region，以 `cn-beijing` 为例: 

```bash
export ALIYUN_DEFAULT_REGION=cn-beijing
```

您可以通过下述命令方便的查询支持的 Region 列表, 当然目前使用默认镜像脚本下仅支持 `cn-hangzhou,cn-beijing,cn-shanghai,cn-huhehaote,cn-zhangjiakou`：

```bash
aliyun ecs DescribeRegions --output cols=RegionId,LocalName rows=Regions.Region
```

然后，您需要下载 GTC-demo 源码。

```bash
git clone https://github.com/aliyun/alibabacloud-aiacc-demo.git
cd alibabacloud-aiacc-demo/pytorch/gtc-demo
```


## 启动 FastGPU 任务 

所有任务都是通过 fastgpu.py 一个脚本来完成，以下步骤是可选的修改配置参数操作，可以跳至步骤3使用默认参数进行一键启动.

1. 配置机器型号

    默认是采用单机单卡 V100 的 spot 机器 `ecs.gn6v-c8g1.2xlarge`，
可以修改 `INSTANCE_TYPE` 变量，以及 spot 配置，不同实例的价钱请参考实际官网的价格

2. 配置 ECS 实例名称

    修改 `args.name`，默认是 `fastgpu-gtc-demo`

3. 启动任务

    运行下述命令，即可启动完整过程，包括购买 ECS 实例、安装 GPU 深度学习环境、准备数据集、部署手势识别项目并启动 training 和 inference。

    ```py
    python fastgpu.py
    ```
    
4. 费用确认

    由于涉及到ECS实例购买,所以提示后续操作涉及到 **费用** 问题,如下所示:
    ```
    the next operation will cost some money, confirm to continue? (yes/no)
    ```

5. 等待结果

    确认继续后,脚本运行成功的显示如下：

    ```
    init ncluster: 55.53943347930908
    upload_data time: 40.93303346633911
    unzip data: 2.7041561603546143
    training time: 19.79602551460266
    inference time: 11.877527952194214
    training and inference deploy time is: 130.8502094745636 s.
    ```

    其中具体的 inference.py 推理结果如下：

    ```
    0: 0.960000
    epoch  0 cost time: 2.373793601989746
    1: 1.000000
    epoch  1 cost time: 2.6852715015411377
    2: 0.960000
    epoch  2 cost time: 1.2226910591125488
    all train time: 6.281756162643433
    布的概率：0.00313576334156096
    石头的概率：0.9850518703460693
    剪刀的概率：0.011812333017587662
    你出的是石头
    推理时间：15.715毫秒
    ```

6. 自动关机

    防止用户遗忘关机而造成不必要的费用损失，所以默认运行完任务后自动关机，关机后可以按照提示，使用fastgpu快捷命令进行再次启动
    
    ```
    stop the instance: task0.fastgpu-gtc-demo
    after stop the instance,you can use cmd:
    1.  'ecluster ls' to look up all instances(including stopped instance)
    2.  'ecluster start task0.fastgpu-gtc-demo' to start the stopped instance
    ```

## FastGPU 任务解释

下面对FastGPU启动任务脚本fastgpu.py进行分步解释

1. ECS机器的选型/购买

    ```
    # setting parameters
    INSTANCE_TYPE = 'ecs.gn6v-c8g1.2xlarge'
    NUM_GPUS = 1
    parser.add_argument('--name', type=str, default='fastgpu-gtc-demo',
                        help="name of the current run, used for machine naming and tensorboard visualization")
    parser.add_argument('--machines', type=int, default=1,
                        help="how many machines to use")
    args = parser.parse_args()
    ```

2. GPU环境的自动化部署

    默认镜像集成了TensorFlow/pytorch/Mxnet的各种框架
    ```
    job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          instance_type=INSTANCE_TYPE,
                          disable_nas=True,
                          spot=True,
                          install_script='')
    ```

3. python特定深度学习环境激活

    ```
    job.run('conda activate torch_1.3_cu10.0_py36')
    ```

4. 上传项目和代码

    ```
    # 2. upload GTC code
    job.run('yum install -y unzip')
    job.upload('GTC')
    job.run('cd GTC && conda activate torch_1.3_cu10.0_py36')
    upload_data = time.time()
    print('upload_data time:', upload_data - init_ncluster)
    # 3. prepare the dataset
    job.run('unzip dataset.zip')
    unzip_time = time.time()
    print('unzip data:', unzip_time - upload_data)
    ```

5. 运行训练和推理

    训练采用阿里云AI加速器aiacc 进行性能加速,支持分布式训练,原始程序可以简单修改配置,无需修改模型代码,即可结合fastgpu进行一键分布式训练.

    ```
    # 4. run the training job
    job.tasks[0].run('conda activate torch_1.3_cu10.0_py36')
    job.tasks[0].run('./run-perseus.sh 2>&1 | tee logs.log', non_blocking=False)
    train_time = time.time()
    print('training time:', train_time - unzip_time)

    # 5. run the inference job
    job.tasks[0].run('python inference.py 2>&1 | tee logs.inference.log', non_blocking=False)
    print('inference time:', time.time() - train_time)
    ```

6. 自动关机停止计费

    由于涉及到ECS资源的计费,所以默认结束任务后会关机

    ```
    # 6. stop the instance (optional)
    job.stop()
    ```


## FastGPU 快捷使用

上述通过 `fastgpu.py` 脚本进行 FastGPU 的使用是 runtime 的方式。

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
task0.cloudshell-fastgpu-gtc-demo      187.7 ecs.gn5t.14xlarge  39.104.101.167  ncluster-shell  192.168.13.208 i-hp3hwvu0xtelntkkp9wq
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
