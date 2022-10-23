#!/bin/sh

#$ -q gpu
#$ -l gpu=1

echo $PATH

source /afs/crc.nd.edu/user/t/tphung/.bashrc

#module load conda

conda activate nn-project

python network_v2.py
