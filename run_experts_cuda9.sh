experts=("DAN" "DeepMOT" "DeepSort" "IOU" "MOTDT" "Sort" "Tracktor_cuda9" "VIOU")
for (( i=0; i<${#experts[@]}; i++ )); do
    python ./run_expert.py -n ${experts[$i]}
done