python imagenet_gen_qsym_mkldnn.py --model=resnet18_v1 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=resnet101_v1 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=resnet50_v1b --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=mobilenet1.0 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=mobilenetv2_1.0 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=inceptionv3 --image-shape=3,299,299 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-resnet-152 --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym_mkldnn.py --model=imagenet1k-inception-bn --num-calib-batches=5 --calib-mode=naive
