eval "$(conda shell.bash hook)"
experts=("CenterTrack" "DAN" "DeepMOT" "Tracktor" "UMA")
for (( i=0; i<${#experts[@]}; i++ )); do
    echo "${experts[$i]}"
    conda activate ${experts[$i]}
    python ./track_expert.py -n ${experts[$i]}
    conda deactivate
done