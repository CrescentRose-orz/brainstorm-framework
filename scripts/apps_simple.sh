SPLIT=test
CONFIG=simple
DIR=apps_${CONFIG}
API_KEY_DIR=$1
typeset -i NUM_API_KEYS=$2

for SHARD in {0..9}; do 
    for process_idx in {0..199}; do
        python turbo.py \
            --config ${CONFIG}_stage1 \
            --dataset apps \
            --input-path data/apps/${SPLIT}_${SHARD}.jsonl \
            --output-path results/${DIR}/stage1/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --api-key-path data/api_keys/${API_KEY_DIR}/$((${process_idx} % ${NUM_API_KEYS})).jsonl \
            --p 0.95 --t 0.8 &
    done
    wait
done 
wait
for SHARD in {0..9}; do 
    for process_idx in {0..199}; do
        python turbo.py \
            --config ${CONFIG}_stage1 \
            --dataset apps \
            --input-path  results/${DIR}/stage1/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --output-path  results/${DIR}/stage2/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --api-key-path data/api_keys/${API_KEY_DIR}/$((${process_idx} % ${NUM_API_KEYS})).jsonl \
            --p 0.95 --t 0.8 &
    done
    wait
done 
wait

for SHARD in {0..9}; do
    for process_idx in {0..199}; do
        python turbo_postprocess.py \
            --input results/${DIR}/stage2/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --output results/${DIR}/post_processed/${SPLIT}_${SHARD}_${process_idx}.jsonl &
    done 
    wait
done 
wait 

for SHARD in {0..9}; do 
    for process_idx in {0..199}; do
        python apps_execute.py \
            --generation-path results/${DIR}/post_processed/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --result-path  results/${DIR}/results/${SPLIT}_${SHARD}_${process_idx}.jsonl &
    done
    wait 
done
wait

python apps_eval.py --result-path results/${DIR}/results
