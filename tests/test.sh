#python ../template/imgclassification-train.py --batch_size 32 --profile --profile_n_epoch 1 --n_epoch 1 --profile_run 1 --model_name cnn --wandb_logging --exp_name img-classrun1
#python ../template/imgclassification-train.py --batch_size 64 --profile --profile_n_epoch 3 --n_epoch 30 --profile_run 1 --model_name cnn --wandb_logging --exp_name img-classrun1
python ../template/imgclassification-train.py --batch_size 64  --n_epoch 30 --model_name cnn --wandb_logging --exp_name img-classrun1
