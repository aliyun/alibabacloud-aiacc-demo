# Using Toutiao dataset to classify the news data. 
* Dataset:  https://public-ai-datasets.oss-cn-huhehaote.aliyuncs.com/toutiao_data.tgz
* pretrain_model: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

## Training script with 1 card.

```Bash
export BERT_CHINESE_BASE_DIR=/root/chinese_L-12_H-768_A-12
export DATA_DIR=/root/toutiao_data
python run_classifier.py \
--task_name=news \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$DATA_DIR \
--vocab_file=$BERT_CHINESE_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_CHINESE_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_CHINESE_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=48 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=/tmp/news_output/ \
--use_tpu=false \
--use_fp16=false \
--use_perseus=true \
--use_amp=true
```

## Training script with 8 card.

```Bash
export BERT_CHINESE_BASE_DIR=/root/chinese_L-12_H-768_A-12
export DATA_DIR=/root/toutiao_data
mpirun --allow-run-as-root -np 8 \
python run_classifier.py \
--task_name=news \
--do_train=true \
--do_eval=true \
--do_predict=true \
--data_dir=$DATA_DIR \
--vocab_file=$BERT_CHINESE_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_CHINESE_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_CHINESE_BASE_DIR/bert_model.ckpt \
--max_seq_length=128 \
--train_batch_size=48 \
--learning_rate=8e-5 \
--num_train_epochs=3.0 \
--warmup_proportion=0.8 \
--output_dir=/ncluster/news_output/ \
--use_tpu=false \
--use_perseus=true \
--use_amp=true

```

## Result.
The final result after 3 epoch with 1 card training and 8 card training can both reach to 0.88, below is the final result of 8 card training.
```
eval_accuracy = 0.8886091
eval_loss = 0.40453392
global_step = 1619
loss = 0.40447024
```

## Benchmark.
On Instance of 8x V100 (gn6v-c10g1.20xlarge) the benchmark result in examples/sec.

| method | 1 GPU | 8 GPU | speed up |
| ------ | ----- | ----- | -------- |
| base   | 118.5 | 865.6 | 7.3      |
| amp+xla | 400  | 2720.8 | 6.8     |
| speed up | 3.37 | 3.14 |          |


