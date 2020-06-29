experts=("DAN")
for (( i=0; i<${#experts[@]}; i++ )); do
    python ./track_expert.py -n ${experts[$i]}
done