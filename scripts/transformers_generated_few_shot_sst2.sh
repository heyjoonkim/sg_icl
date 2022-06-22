export CUDA_VISIBLE_DEVICES=0

## TASKS ##
task="sst2"
benchmark="glue"

## MODELS ##
# main_model="gpt2-xl"
# main_model="EleutherAI/gpt-neo-1.3B"
# main_model="EleutherAI/gpt-neo-2.7B"
main_model="EleutherAI/gpt-j-6B"

## directory ##
main_path="./test_results/paper_results"
dataset_path="./generated_datasets"

##############
## FEW-SHOT ##
##############

# seeds="13 21 42 87 100"
seeds="21 42 87 100"

n_samples="8"

# generation template
generation_template="template1"

# inference template
inference_template="template1"

# Manual template #
for n_sample in $n_samples; do
    for seed in $seeds; do
python transformers_generated_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task/$main_model/$n_samples-shot/generated-$inference_template/ \
    --dataset_dir $dataset_path/$task/$main_model/$generation_template/$n_samples-shot/$seed/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix 'Review : ' \
    --infix '
Sentiment :' \
    --postfix ''
    done
done
# Manual template #

# Minimal template #
for n_sample in $n_samples; do
    for seed in $seeds; do
python transformers_generated_main.py \
    --task_name $task \
    --benchmark_name $benchmark \
    --model_name_or_path $main_model \
    --output_dir $main_path/$task/$main_model/$n_samples-shot/generated-minimal/ \
    --dataset_dir $dataset_path/$task/$main_model/$generation_template/$n_samples-shot/$seed/ \
    --seed $seed \
    --n_samples $n_sample \
    --overwrite_output_dir \
    --prefix '' \
    --infix '
' \
    --postfix ''
    done
done
# Minimal template #


# sh scripts/transformers_generate_sst5.sh
