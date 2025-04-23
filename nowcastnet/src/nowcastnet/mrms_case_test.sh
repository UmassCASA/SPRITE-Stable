
python -u run.py \
    --worker 1 \
    --device cuda:0 \
    --cpu_worker 2 \
    --dataset_name radar \
    --dataset_path /work/pi_mzink_umass_edu/nowcasting/NowcastNet/data/dataset/mrms/figure \
    --pretrained_model /work/pi_mzink_umass_edu/nowcasting/NowcastNet/data/checkpoints/mrms_model.ckpt \
    --gen_frm_dir /work/pi_mzink_umass_edu/nowcasting/NowcastNet/results/us/test_result \
    --num_save_samples 10 \
    --model_name NowcastNet \
    --img_height 512 \
    --img_width 512 \
    --case_type normal \
    --img_ch 2 \
    --input_length 9 \
    --total_length 29 \
    --batch_size 1 


    # --pretrained_model /data/checkpoints/mrms_model.ckpt \
