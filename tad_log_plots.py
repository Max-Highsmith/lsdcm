import matplotlib.pyplot as plt
import numpy as np

colors = ['black', 'silver', 'blue', 'darkviolet', 'coral']

full_insulation_dist = np.load("tad_dict.npy", allow_pickle=True).item()
fig, ax = plt.subplots(1)
box_data = [
        np.array(full_insulation_dist['down'], ),
        full_insulation_dist['hicplus'],
        full_insulation_dist['deephic'],
        full_insulation_dist['hicsr'],
        full_insulation_dist['vehicle']]
positions = [1,2,3,4,5]

bp = ax.boxplot(box_data,
        positions=positions,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        notch=True)

ax.set_xticklabels(['down', 'hicplus', 'deephic', 'hicsr', 'vehicle'])
ax.set_ylabel("TAD Border Distance (10kb)")
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

for patch, color in zip(bp['medians'], colors):
    patch.set_c('white')
plt.show()
