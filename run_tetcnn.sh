# apptainer exec --nv --bind /scratch/ychen855/tetCNN:/tetCNN,/data/hohokam/Yanxi/Data/tetCNN/328_test/lh:/data /scratch/ychen855/tetcnn_docker/tetcnn2.0.sif python3 /tetCNN/src/tetCNN_lm.py --pos /data/ad --neg /data/cn --n_lmk 1000 --k 500 --epoch 30 --load
# module load cuda-11.8.0-gcc-12.1.0
python /scratch/ychen855/tetCNN/src/tetCNN_lm.py --pos /data/hohokam/Yanxi/Data/plasma/bai_pos --neg /data/hohokam/Yanxi/Data/plasma/bai_neg --n_lmk 200 --k 100 --epoch 30 --network mlp --lr 0.001 --wd 0.0001 --cv 0 --load