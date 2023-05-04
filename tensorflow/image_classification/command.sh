COMMAND='python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --image_size=224 --model=resnet50_v1.5 --batch_size=128 --display_every=100 --data_name=imagenet --nodistortions --variable_update=perseus --num_gpus=1 --use_fp16=True --fp16_gradient=True --batch_group_size=4 --num_inter_threads=8 --datasets_use_prefetch=True --datasets_num_private_threads=5 --device=gpu --xla=True --allow_growth=True --optimizer=momentum --momentum=0.9 --weight_decay=1e-4 --use_datasets=False --num_eval_epochs=1 --eval_during_training_every_n_epochs=1 --num_warmup_batches=500 --num_batches=1500'
if [ $OMPI_COMM_WORLD_RANK -eq 0 ] ; then
$COMMAND
else
$COMMAND >> /dev/null 2>&1
fi
