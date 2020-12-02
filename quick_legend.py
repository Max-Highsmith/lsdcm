import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

CELL_LINES       = ['IMR90', 'HMEC', 'GM12878', 'K562']
CELL_LINE_SHAPES = ["o","v","P","*"]

patchs = []
for li, shape in zip(CELL_LINES, CELL_LINE_SHAPES):
    temp_patch = mlines.Line2D([],[], linestyle='None', marker=shape, color='green', label=li, markeredgewidth=4)
    patchs.append(temp_patch)

plt.legend(handles=patchs)
plt.show()

METHODS        = ['down', 'hicplus', 'deephic', 'hicsr', 'vehicle']
METHOD_COLORS  = ['black', 'silver', 'blue', 'darkviolet', 'coral']
legs = []
for me, col in zip(METHODS, METHOD_COLORS):
    temp = mlines.Line2D([],[], color=col, label=me, linewidth=4)
    legs.append(temp)

plt.legend(handles=legs)
plt.show()
