
#mkdir output_cnmci_MDTO
#mkdir output_cnmci_MDTS
Mkdir output_cnmci_MDT_LDISlope




#python3 run.py --type c --input ./data/cnmci_MDTO_nodemo.csv --outcome class --out ./output_cnmci_MDTO/ --n_bootstraps 1000 --bootstrap_sampling_rate 1.0


#python3 run.py --type c --input ./data/cnmci_MDTS_nodemo.csv --outcome class --out ./ output_cnmci_MDTS/ --n_bootstraps 1000 --bootstrap_sampling_rate 1.0

python3 run.py --type c --input ./data/cnmci_MDT_nodemo_addLDIslope.csv --outcome class --out ./output_cnmci_MDT_LDISlope/ --n_bootstraps 1000 --bootstrap_sampling_rate 1.0