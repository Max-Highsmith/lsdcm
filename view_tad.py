import matplotlib.pyplot as plt
import numpy as np
full_insulation_dist = np.load('tad_dict.npy', allow_pickle=True).item()
#print(full_insulation_dist['hicplus'])
#plt.hist(full_insulation_dist['down'])
box_data = [
        np.array(full_insulation_dist['down']).tolist(),
        np.array(full_insulation_dist['hicplus']).tolist(),
        np.array(full_insulation_dist['deephic']).tolist(),
        np.array(full_insulation_dist['hicsr']).tolist(),
        np.array(full_insulation_dist['vehicle']).tolist()]
positions = [1,2,3,4,5]

fig, ax = plt.subplots()
bp = ax.boxplot(box_data,
    positions=positions)
plt.show()
