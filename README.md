# PeerCollaboration


You can execute the order:
CUDA_VISIBLE_DEVICES=0 /home/sunyang/anaconda3/envs/py36torch10/bin/python3 -u SASRec_layercooperation.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 1 --a 30 --seed 10 --cos --difflr > 1.log & CUDA_VISIBLE_DEVICES=1 /home/sunyang/anaconda3/envs/py36torch10/bin/python3 -u SASRec_layercooperation.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 2 --a 30 --seed 11 --cos --difflr > 2.log
