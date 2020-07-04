durations=(70 50 30)
thresholds=(0.5 0.3 0.7)

for (( j=0; j<${#durations[@]}; j++ )); do
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        CUDA_VISIBLE_DEVICES=0 python ./track_algorithm.py -d ${durations[$j]} -t ${thresholds[$i]} -l "fn"
    done
done