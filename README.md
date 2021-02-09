# PeerCollaboration


## Layer-wise cooperation
You can run the following order:  
CUDA_VISIBLE_DEVICES=0 /home/sunyang/anaconda3/envs/py36torch10/bin/python3 -u SASRec_layercooperation.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 1 --a 30 --seed 10 --cos --difflr --cooperation_type 'cooperation_type' > 1.log & CUDA_VISIBLE_DEVICES=1 /home/sunyang/anaconda3/envs/py36torch10/bin/python3 -u SASRec_layercooperation.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 2 --a 30 --seed 11 --cos --difflr --cooperation_type 'cooperation_type' > 2.log  

The import configuration:  
--cooperation_type: This is used to control which cooperation type is used. There are 5 parameters in SASRec_layercooperation.py:  
alllayer_entropy：All layers are used by layer-wise cooperation with entropy criterion.
alllayer_eL1norm：All layers are used by layer-wise cooperation with L1-norm criterion.
onlyembed：Only the embedding layer uses layer-wise cooperation with entropy criterion.
onlymiddle：All layers exception the embedding and final layer use layer-wise cooperation with entropy criterion.  
onlyfinal： Only the final layer uses layer-wise cooperation with entropy criterion.


## Parameter-wise cooperation  
You can run the following order:  
CUDA_VISIBLE_DEVICES=0 /home/sunyang/anaconda3/envs/py36torch10/bin/python3 -u SASRec_singleweight.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 1 --percent 50 --seed 10 --cos --difflr > 1.log & CUDA_VISIBLE_DEVICES=1 /home/sunyang/anaconda3/envs/py36torch10/bin/python3 -u SASRec_singleweight.py --datapath ml20m_removecold5_seq.csv --savedir ml20m/ --max_len 100 --num 2 --i 2 --percent 50 --seed 11 --cos --difflr > 2.log  


you can download a large sequential dataset of movielen-20m that has been pre-processed: https://drive.google.com/file/d/1AlzZyjTsceayYvSvUJL37J-5jtwOclpi/view?usp=sharing

# References
https://github.com/fajieyuan/nextitnet code  
https://github.com/kang205/SASRec code  
