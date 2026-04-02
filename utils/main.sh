log=examples/logs/
# file="sum_classifier_asapool_GraphConv_noinragnn_baseline"
file="sum_classifier_sparsepool_GCN_v6"
# "ENZYMES"FRANKENSTEIN"
# "nopool" "topkpool" "sagpool" "diffpool" "mincutpool" "densepool" "sparsepool" "asapool"
# "PROTEINS" "ENZYMES"  "Mutagenicity" "DD" "NCI1" "COX2"
# for pool in nopool" "topkpool" "sagpool" "asapool" "diffpool" "mincutpool"  
for dataset in  "PROTEINS" "ENZYMES"  "Mutagenicity" "DD" "NCI1" "COX2" 
do
    for pool in "sparsepool"
        do
        python3 main.py  --log-file $log$file --pool $pool --dataset $dataset --tag "batch_norm_all_v6"
        done
done
