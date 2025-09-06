# BBBP
python main.py --task downstream --data bbbp --criterion bce --lr 5e-5 --nim_classes 1

# Clintox
python main.py --task downstream --data clintox --criterion bce --lr 5e-5 --num_classes 2

# Tox21
python main.py --task downstream --data tox21 --criterion bce --lr 5e-3 --num_classes 12

# HIV
python main.py --task downstream --data hiv --criterion bce --lr 1e-4 --num_classes 1

# ESOL
python main.py --task downstream --data esol --criterion rmse --lr 5e-5 --num_classes 1

# Freesolv
python main.py --task downstream --data freesolv --criterion rmse --lr 1e-4 --num_classes 1

# Lipophilicity
python main.py --task downstream --data lipophilicity --criterion rmse --lr 5e-5 --num_classes 1

# bace
python main.py --task downstream --data bace --criterion bce --split scaffold --lr 5e-5 --batch_size 32 --num_classes 1 --split scaffold

# sider
python main.py --task downstream --data sider --criterion bce --num_classes 27 --lr 5e-5

# qm7
python main.py --task downstream --data qm7 --criterion mae --batch_size 1 --lr 5e-6

# qm8
python main.py --task downstream --data qm8 --criterion mae --num_classes 12 --batch_size 8 --lr 6e-5 



