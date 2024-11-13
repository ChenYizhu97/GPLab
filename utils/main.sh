log=logs/screenshot/
# file="sum_classifier_asapool_GraphConv_noinragnn_baseline"
file="sum_classifier_lspool_GCN_v6"
# "ENZYMES"FRANKENSTEIN"
# "nopool" "lspool" "topkpool" "sagpool" "diffpool" "mincutpool" "densepool" "sparsepool" "asapool"
# "PROTEINS" "ENZYMES"  "Mutagenicity" "DD" "NCI1" "COX2"
# for pool in nopool" "topkpool" "sagpool" "asapool" "diffpool" "mincutpool"  
for dataset in  "PROTEINS" "ENZYMES"  "Mutagenicity" "DD" "NCI1" "COX2" 
do
    for pool in "lspool"
        do
        python3 main.py  --logging $log$file --pooling $pool --dataset $dataset --comment "batch norm all. v6. A3. graphconv collapse for new x before diff and score."
        done
done