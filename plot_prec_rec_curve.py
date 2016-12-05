import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# data needs to be python list [1,2,2,3,...] in order: score[label1],prec[label1],rec[label1],score[label2],...
# data ist liste von listen mit strings --> eval
data = [line.strip() for line in open("myDataSet_full_SSD300_prk_data.txt", 'r')]
for element in range(len(data)):
  data[element] = eval(data[element])

color_list = ['g', 'crimson', '#FFBF00', 'b']
class_list = ['robot', 'base', 'battery', 'mug']

plt.style.use('classic')
matplotlib.rcParams.update({'font.size': 18})

# plot recall-precision curve
fig = plt.figure()
ax = fig.gca()
plt.grid(linestyle=':', linewidth=1, color='0.5')
ax.set_xticks(np.arange(0,1.1,0.1))
ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 1])

for i in range(0, len(data), 3):
    plt.plot(data[i+2], data[i+1], linewidth=2, color=color_list[i/3], label=class_list[i/3])

#plt.axis([0, 1, 0, 1.05])
plt.ylabel("Precision", fontsize=25, labelpad=20)
plt.xlabel("Recall", fontsize=25, labelpad=20)

ax.legend(loc=3)
plt.show()


# plot threshold-recall curve
fig = plt.figure()
ax = fig.gca()
plt.grid(linestyle=':', linewidth=1, color='0.5')
ax.set_xticks(np.arange(0,1.1,0.1))
ax.set_yticks(np.arange(0,1.1,0.1))
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 1])

for i in range(0, len(data), 3):
  plt.plot(data[i+2], data[i], linewidth=2, color=color_list[i/3], label=class_list[i/3])

#plt.axis([0, 1, 0, 1.05])
plt.ylabel("Score", fontsize=25, labelpad=20)
plt.xlabel("Recall", fontsize=25, labelpad=20)

ax.legend(loc=3)
plt.show()
