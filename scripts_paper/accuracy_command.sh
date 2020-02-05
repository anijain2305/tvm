#!/bin/bash
rm -rf perf.txt
touch perf.txt
# For Mxnet, best baseline, this is required
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0

# For TVM, best numbers, following line is required.
export TVM_BIND_MASTER_THREAD=1 

NUM_ITERS=5
MODEL_PATH=/home/ubuntu/mxnet/incubator-mxnet/example/quantization/model
rm -rf accuracy.txt
touch accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/resnet18_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet18_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375  --num-skipped-batches=50  --image-shape=3,224,224   --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/resnet50_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1-quantized-0000.params  --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50  --image-shape=3,224,224   --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/resnet50_v1b-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet50_v1b-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375  --num-skipped-batches=50  --image-shape=3,224,224   --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/resnet101_v1-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/resnet101_v1-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375  --num-skipped-batches=50  --image-shape=3,224,224   --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-resnet-152-quantized-0000.params --num-skipped-batches=50  --image-shape=3,224,224   --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/mobilenet1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenet1.0-quantized-0000.params --image-shape=3,224,224  --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50    --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/mobilenetv2_1.0-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/mobilenetv2_1.0-quantized-0000.params --image-shape=3,224,224  --rgb-mean=123.68,116.779,103.939 --rgb-std=58.393,57.12,57.375 --num-skipped-batches=50    --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50  --image-shape=3,224,224   --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
python3 measure_accuracy.py --symbol-file=$MODEL_PATH/inceptionv3-quantized-5batches-naive-symbol.json --param-file=$MODEL_PATH/inceptionv3-quantized-0000.params  --rgb-mean=123.68,116.779,103.939  --rgb-std=58.393,57.12,57.375  --num-skipped-batches=50 --image-shape=3,299,299    --batch-size=1 --num-inference-batches=10000  --dataset=./data/val_256_q90.rec --score_tvm=True --ctx=cpu |& tee -a accuracy.txt
