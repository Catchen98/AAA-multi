eval "$(conda shell.bash hook)"

# "CenterTrack" "DeepMOT" "GCNNMatch" "MOTDT" "Tracktor" "UMA"
experts=("GCNNMatch")

# CUDA_VISIBLE_DEVICES=1
for (( i=0; i<${#experts[@]}; i++ )); do
    echo "${experts[$i]}"
    conda activate ${experts[$i]}
    python ./track_expert.py -n ${experts[$i]}
    conda deactivate
done