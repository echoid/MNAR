# How to run script:

TabCSDS:

cd MNAR/TabCSDI
python exe.py --dataset banknote --missingtype logistic --missingpara single_quantile --config 1ep --nsample 1


GAIN:
cd MNAR/GAIN
python GAIN.py banknote quantile

MCflow

cd MNAR/MCFlow
python main.py --dataset banknote --missingtype quantile --missingpara single_quantile

