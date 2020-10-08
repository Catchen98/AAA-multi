eval "$(conda shell.bash hook)"
# "CenterTrack" "DeepMOT" "DeepSORT" "MOTDT" "Tracktor"
experts=("UMA")
for (( i=0; i<${#experts[@]}; i++ )); do
    echo "${experts[$i]}"
    conda activate ${experts[$i]}
    CUDA_VISIBLE_DEVICES=1 python ./track_expert.py -n ${experts[$i]}
    conda deactivate
done