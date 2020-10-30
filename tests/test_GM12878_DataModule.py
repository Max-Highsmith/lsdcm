import matplotlib.pyplot as plt
import pdb
import sys
sys.path.append("../")
sys.path.append("Data")
sys.path.append(".")
import GM12878_DataModule as dm


x = dm.GM12878Module()
x.prepare_data()
x.setup('fit')
for batch in x.train_dataloader():
    pdb.set_trace()
#x.download_raw_data()
#x.extract_constraint_mats()
#x.extract_create_numpy()
#x.split_numpy()a

