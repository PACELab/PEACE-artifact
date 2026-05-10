

#write a test script that iterates through batch_size on CPU and GPU
batch_sizes=(1 2 4 8 16 32 64)
#bacth_size=2
device="cpu"
epoch=3
echo "baseline inference..."
cd ../../workloads/inference
for batch_size in "${batch_sizes[@]}"; do
    echo "Testing with batch size: $batch_size"
    # Execution 1: Training with imgclassification-inference.py
    python imgclassification-inference.py --model_name microsoft/resnet-50 --log_dir ../../tests/mps/rtx6000_logs/baseline/inference/batch${batch_size} --batch_size $batch_size --device $device
    echo "Execution 1 (imgclassification-inference.py) completed for batch size: $batch_size"
    # Execution 2: Training with recommend-train.py
    python speech-recognition-inference.py --model_name facebook/wav2vec2-base-960h --log_dir ../../tests/mps/rtx6000_logs/baseline/inference/batch${batch_size} --batch_size $batch_size --device $device
    echo "Execution 2 (speech-recognition-inference.py) completed for batch size: $batch_size"

    #execution 3: inference with imgclassification-inference.py on model google/mobilenet_v2_1.0_224
    python imgclassification-inference.py --model_name google/mobilenet_v2_1.0_224 --log_dir ../../tests/mps/rtx6000_logs/baseline/inference/batch${batch_size} --batch_size $batch_size --device $device

    #execution 4: inference with imgclassification-inference.py on model google/vit-base-patch16-224
    python imgclassification-inference.py --model_name google/vit-base-patch16-224 --log_dir ../../tests/mps/rtx6000_logs/baseline/inference/batch${batch_size} --batch_size $batch_size --device $device
    echo "------------------------------------------"
done

#training
cd ../training
echo "baseline training..."
for batch_size in "${batch_sizes[@]}"; do
    echo "Testing with batch size: $batch_size"
    #execution5: train with imgclassification-train.py on model microsoft/resnet-50
    python imgclassification-train.py --model_name microsoft/resnet-50 --log_dir ../../tests/mps/rtx6000_logs/baseline/training/batch${batch_size} --batch_size $batch_size --device $device --n_epoch $epoch
    
    #execution6: train with recommend-train.py on model bert-base-cased
    python recommend-train.py --model_name bert-base-cased --log_dir ../../tests/mps/rtx6000_logs/baseline/training/batch${batch_size} --batch_size $batch_size --device $device --n_epoch $epoch

    #execution7: train with imgclassification-train.py on model mobilenetv2
    python imgclassification-train.py --model_name mobilenet --log_dir ../../tests/mps/rtx6000_logs/baseline/training/batch${batch_size} --batch_size $batch_size --device $device --n_epoch $epoch
    echo "------------------------------------------"

done