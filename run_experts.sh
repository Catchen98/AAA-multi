eval "$(conda shell.bash hook)"

# "CenterTrack" "DeepMOT" "DeepSORT" "MOTDT" "Tracktor" "UMA"
experts=("CenterTrack" "DeepMOT" "DeepSORT" "MOTDT" "Tracktor" "UMA")

# CUDA_VISIBLE_DEVICES=1
for (( i=0; i<${#experts[@]}; i++ )); do
    echo "${experts[$i]}"
    conda activate ${experts[$i]}
    python ./track_expert.py -n ${experts[$i]}
    conda deactivate
done