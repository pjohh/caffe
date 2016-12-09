import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# data needs to be python list [1,2,2,3,...] in order: score[label1],prec[label1],rec[label1],score[label2],...
# data ist liste von listen mit strings --> eval
data = [line.strip() for line in open("robot_dataset_SSD300_conv3_8_prk_data.txt", 'r')]
for element in range(len(data)):
  data[element] = eval(data[element])
  
for i in range(3, len(data), 4):
  data[i] = [x for x in data[i] if x != 1]
  if len(data[i]) < len(data[i-1]):
    data[i].append(1)
  del data[i-1][len(data[i]):]
  del data[i-3][len(data[i]):]
  
  # search in precision for first false positiv -> prec < 1 -> output score
  for j in range(len(data[i-1])):
    if data[i-1][j] < 1:
      print("first positive for label: {} | with score: {}".format(i/4+1, data[i-3][j]))
      break

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

for i in range(0, len(data), 4):
    plt.plot(data[i+3], data[i+2], linewidth=2, color=color_list[i/4], label=class_list[i/4])

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

for i in range(0, len(data), 4):
  plt.plot(data[i+3], data[i], linewidth=2, color=color_list[i/4], label=class_list[i/4])

#plt.axis([0, 1, 0, 1.05])
plt.ylabel("Score", fontsize=25, labelpad=20)
plt.xlabel("Recall", fontsize=25, labelpad=20)

ax.legend(loc=3)
plt.show()
