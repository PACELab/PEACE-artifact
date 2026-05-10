wg=$1
wc=$2
ag=$3
ac=$4
hg=$5
hc=$6
DIRPATH="/home/bing/llm-queue/build/FlexGen-main/flexgen/apps/data_wrangle"
OUTPUTDIR="$7/"
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "$timestamp: Starting batch#1..."
python3  $DIRPATH/data_wrangle_run.py\
    --num_run 189 \
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir $DIRPATH/data/datasets/entity_matching/structured/Fodors-Zagats \
    --output_dir $OUTPUTDIR \
    --batch_run --pad-to-seq-len 744 --model facebook/opt-6.7b --percent $wg $wc $ag $ac $hg $hc --gpu-batch-size 8 --num-gpu-batches 1
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "$timestamp: Starting batch#2..."
python3  $DIRPATH/data_wrangle_run.py\
    --num_run 91 \
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir $DIRPATH/data/datasets/entity_matching/structured/Beer \
    --output_dir $OUTPUTDIR \
    --batch_run --pad-to-seq-len 592 --model facebook/opt-6.7b --percent $wg $wc $ag $ac $hg $hc --gpu-batch-size 8 --num-gpu-batches 1

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "$timestamp: Starting batch#3..."
python3  $DIRPATH/data_wrangle_run.py\
    --num_run 109 \
    --num_trials 1 \
    --nan_tok "" \
    --do_test \
    --sample_method manual \
    --data_dir $DIRPATH/data/datasets/entity_matching/structured/iTunes-Amazon \
    --output_dir $OUTPUTDIR \
    --batch_run --pad-to-seq-len 529 --model facebook/opt-6.7b --percent $wg $wc $ag $ac $hg $hc --gpu-batch-size 8 --num-gpu-batches 1
