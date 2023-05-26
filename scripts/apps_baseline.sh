SPLIT=test
CONFIG=baseline
DIR=apps_${CONFIG}
API_KEY_DIR=$1
typeset -i NUM_API_KEYS=$2

for SHARD in {0..9}; do 
    for process_idx in {0..199}; do
        python turbo.py \
            --config ${CONFIG} \
            --dataset apps \
            --input-path data/apps/${SPLIT}_${SHARD}.jsonl \
            --output-path results/${DIR}/completion/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --api-key-path data/api_keys/${API_KEY_DIR}/$((${process_idx} % ${NUM_API_KEYS})).jsonl \
            --p 0.95 --t 0.8 &
    done
    wait
done 

for SHARD in {0..9}; do
    for process_idx in {0..199}; do
        python turbo_postprocess.py \
            --input results/${DIR}/completion/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --output results/${DIR}/post_processed/${SPLIT}_${SHARD}_${process_idx}.jsonl &
    done 
    waitq
done 

for SHARD in {0..9}; do 
    for process_idx in {0..199}; do
        python apps_execute.py \
            --generation-path results/${DIR}/post_processed/${SPLIT}_${SHARD}_${process_idx}.jsonl \
            --result-path  results/${DIR}/results/${SPLIT}_${SHARD}_${process_idx}.jsonl &
    done
    wait 
done

python apps_eval.py --result-path results/${DIR}/results
