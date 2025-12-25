# #!/bin/bash  
  

configs=(
    "GOODFormer_configs/GOODHIV/scaffold/covariate/GOODFormer.yaml"
    "GOODFormer_configs/GOODHIV/size/covariate/GOODFormer.yaml"
    "GOODFormer_configs/GOODMotif/basis/covariate/GOODFormer.yaml"
    "GOODFormer_configs/GOODMotif/size/covariate/GOODFormer.yaml"
    "GOODFormer_configs/GOODSST2/length/covariate/GOODFormer.yaml"
)  

rounds=(1 2 3)
for round in "${rounds[@]}"; do
    for cfg in "${configs[@]}"; do
        goodtg --config_path "$cfg" --exp_round "$round" --gpu_idx 0 &  
        PID=$!  
        wait $PID
    done
done  