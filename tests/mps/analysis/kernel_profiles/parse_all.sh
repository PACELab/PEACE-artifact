#input 
#create batch_size as array of 2 8 16
batch_size=(2 8 16)
for i in "${batch_size[@]}"
do
    echo "generating profile for batch size $i"
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/bert-inf_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/bert-inf_batch$i.nsys-rep bert-inf_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/bert-inf_batch$i.log
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/mobile-inf_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/mobile-inf_batch$i.nsys-rep mobile-inf_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/mobile-inf_batch$i.log
    #repeat for other modeles: vit_h_14-train,vit-inf,wav2vec-inf,whisper-inf,resnet50-inf
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/vit_h_14-train_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/vit_h_14-train_batch$i.nsys-rep vit_h_14-train_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/vit_h_14-train_batch$i.log
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/vit-inf_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/vit-inf_batch$i.nsys-rep vit-inf_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/vit-inf_batch$i.log
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/wav2vec-inf_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/wav2vec-inf_batch$i.nsys-rep wav2vec-inf_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/wav2vec-inf_batch$i.log
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/whisper-inf_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/whisper-inf_batch$i.nsys-rep whisper-inf_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/whisper-inf_batch$i.log
    bash generate_profiles.sh /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/resnet50-inf_batch$i.ncu-rep /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/nsys/resnet50-inf_batch$i.nsys-rep resnet50-inf_batch$i > /home/cc/mlProfiler/tests/mps/analysis/kernel_profiles/source/ncu/resnet50-inf_batch$i.log
done
