durations=(30 50 70)
thresholds=(0.3 0.5 0.7)

for (( j=0; j<${#durations[@]}; j++ )); do
    for (( i=0; i<${#thresholds[@]}; i++ )); do
        CUDA_VISIBLE_DEVICES=1 python ./track_algorithm.py -d ${durations[$j]} -t ${thresholds[$i]} -l "sum"
    done
done