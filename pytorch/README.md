# 一键部署并训练手势识别应用

## 简介

GTC-demo 为阿里云在GTC2019中国GTC之夜展示的1小时做出手势识别AI猜拳小程序的展示用例，具体为手势识别imageNet系列模型的 finetune 代码,基于pytorch框架,能够识别剪刀/石头/布的三种手势,训练采用阿里云AI加速库 aiacc 进行性能优化,无缝支持多机分布式训练。其中的数据集由手机拍摄，并经过批量 resize 的手势图片，数据集规模是 173 张图片。

本教程介绍了使用 FastGPU 从 0 开始，一键部署并训练手势识别应用。



## 环境准备

**说明**：Cloud Shell 已经预装了 FastGPU 工具，建议您直接使用 [Cloud Shell](https://shell.aliyun.com)

1.

需要配置您的[阿里云AccessKey](https://help.aliyun.com/document_detail/38738.html)到环境变量中

```bash
export ALIYUN_ACCESS_KEY_ID=L****      # Your actual aliyun access key id
export ALIYUN_ACCESS_KEY_SECRET=v****   # Your actual aliyun access key secret
```

2. 

需要配置您想要创建的资源所在大区，您可以根据需要设置对应的 Region，以 `cn-beijing` 为例:

```bash
export ALIYUN_DEFAULT_REGION=cn-beijing
```

您可以通过下述命令方便的查询支持的 Region 列表, 当然目前使用默认镜像脚本下仅支持 `cn-hangzhou,cn-beijing,cn-shanghai,cn-huhehaote,cn-zhangjiakou,cn-shenzhen`：

```bash
aliyun ecs DescribeRegions --output cols=RegionId,LocalName rows=Regions.Region
```

3.

代码已经保存在本地路径`GTC`下。

## 启动 FastGPU 任务

所有任务都是通过 `fastgpu_script.py` 一个脚本来完成，以下步骤是可选的修改配置参数操作，可以跳至步骤3使用默认参数进行一键启动.

1. 配置机器型号

   默认是采用单机单卡 V100 的 spot 机器 `ecs.gn6v-c8g1.2xlarge`，
   可以修改 `INSTANCE_TYPE` 变量，以及 spot 配置，不同实例的价钱请参考实际官网的价格
2. 配置 ECS 实例名称

   修改 `args.name`，默认是 `fastgpu-gtc-demo`
3. 启动任务

   运行下述命令，即可启动完整过程，包括购买 ECS 实例、安装 GPU 深度学习环境、准备数据集、部署手势识别项目并启动 training 和 inference。

   ```py
   python fastgpu_script.py
   ```
4. 费用确认

   由于涉及到ECS实例购买,所以提示后续操作涉及到 **费用** 问题,如下所示:

   ```
   Costs of the resourses (ECS/VPC/NAS) will be charged on your Aliyun account, confirm to continue? (y/n)
   ```
5. 等待结果

   确认继续后,脚本运行成功的显示如下：

   ```
   init fastgpu: 80.23s
   upload_data time: 20.23s
   unzip data: 0.71s
   training time: 32.87
   inference time: 5.71s
   training and inference deploy time is: 139.75s.
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
6. 自动停机

   防止用户遗忘停机而造成不必要的费用损失，所以默认运行完任务后自动停机。停机后可以按照提示，使用fastgpu快捷命令进行再次启动

   ```
   # stop task0.fastgpu-gtc-demo and stop charging : success
   Hits: After stop the instance, you can use the command line:
   1. 'fastgpu ls --stop' to look up all instances(including stopped instance)
   2. 'fastgpu start task0.fastgpu-gtc-demo' to start the stoppend intance
   3. 'fastgpu kill task0.fastgpu-gtc-demo' to release the instance
   ```

## FastGPU 任务解释

下面对FastGPU启动任务脚本fastgpu.py进行分步解释

1. ECS机器的选型/购买

   ```python
   # setting parameters
   INSTANCE_TYPE = 'ecs.gn6v-c8g1.2xlarge'
   NUM_GPUS = 1
   IMAGE_NAME = 'aiacc-dlimg-ubuntu2004:1.5.0'

   fastgpu.set_backend('aliyun')
   parser = argparse.ArgumentParser()
   parser.add_argument('--name', type=str, default='fastgpu-gtc-demo',
                       help="name of the current run, used for machine naming and tensorboard visualization")
   parser.add_argument('--machines', type=int, default=1,
                       help="how many machines to use")
   args = parser.parse_args()
   ```
2. 创建机器和自动化部署GPU环境

   默认的系统镜像集成已安装

   - GPU驱动和CUDA
   - conda和TensorFlow/pytorch/Mxnet的各个版本的环境

   ```python
   job = fastgpu.make_job(name=args.name,
                             run_name=f"{args.name}-{args.machines}",
                             num_tasks=args.machines,
                             instance_type=INSTANCE_TYPE,
                             image_name=IMAGE_NAME,
                             disable_nas=True,
                             spot=True,
                             install_script='') 
   ```
3. 激活特定的深度学习环境

   ```python
   job.tasks[0].run('conda init bash && conda activate torch_1.3.0_cu10.0_py36')
   ```
4. 上传项目和下载数据

   ```
   # 2. upload GTC code
   job.run('yum install -y unzip')
   job.upload('GTC')
   ...

   ```
5. 运行训练和推理

   训练采用[阿里云AI加速器（AIACC训练加速）](https://help.aliyun.com/document_detail/276016.html)进行性能加速，支持分布式训练。原始程序可以简单修改配置，无需修改模型代码，即可结合fastgpu进行一键分布式训练.

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

   由于涉及到ECS资源的计费,所以默认结束任务后会关机,并释放。

   ```
   # 6. stop the instance (optional)
   job.stop()
   # 7. kill the instance (optional) 
   job.kill()
   ```

## FastGPU 快捷使用

上述通过 `fastgpu_script.py` 脚本进行 FastGPU 的使用是 runtime 的方式。

此外，FastGPU 还提供便捷的 `fastgpu` 命令行进行云上资源的生命周期，查看运行过程的日志和日常开发等。

上述创建的任务完成后，您可以通过下述命令查看实例状态：

```bash
fastgpu ls
```

执行后，您可以看到具体的实例状态，如下所示：

```shell
# fastgpu ls                           
In Region cn-xxx

Running Instances:
---------------------------------------------------------------------------------------------------
Instance Name          | Age(hr) | Public IP     | Private IP    | GPU      | Instance Type  
---------------------------------------------------------------------------------------------------
task0.fastgpu-gtc-demo | 64.3    | xx.xx.xx.xx | 192.168.8.103 | V100 x 1 | ecs.gn6v-c8g1.2xlarge
```

最后您可以参考 [FastGPU 常用命令](https://help.aliyun.com/document_detail/187616.html)，对您的云上资源进行管理。
