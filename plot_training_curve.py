import numpy as np
import re
import click
from matplotlib import pylab as plt
import matplotlib
from scipy import polyval, polyfit

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x):
    return 99.25436 + (29.50511 - 99.25436)/(1 + (x/0.764983)**82.02377)**0.008900094

#xn = np.asarray([1,5, 10, 50, 250])
#yn = [50,70, 85, 95, 98.9]

x = np.linspace(0,250, 100)
y = func(x)

#plt.figure()
#plt.plot(xn, yn, 'ko', label="Original Noised Data")
#plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
#plt.legend()
#plt.show()



y_train = [100, 100,100, 100, 100, 100]
y_val = [0, 41.8, 82.5, 87.6, 95.4, 98.9]
x_train = [0, 1,5, 10, 50, 250]

#a, b, c, d = polyfit(x, y_val, 3)
#y_val_out = polyval([a, b, c, d], x) 

#x_out = np.linspace(0, 100, 100)   # choose 20 points, 10 in, 10 outside original range
#y_val_out = polyval([a, b, c], x_out)

plt.style.use('classic')
matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure()
ax = fig.gca()
plt.grid(linestyle=':', linewidth=1, color='0.5')
ax.set_xlabel('Anzahl Trainingsbilder', fontsize=25, labelpad=20)
ax.set_ylabel('mAP (in %)', fontsize=25, labelpad=20)
ax.set_yticks(np.arange(0,101,10))
ax.set_ylim([0, 105])
 
train_data = ax.plot(x_train, y_train, 'k', linewidth=2, label='Trainingsdaten')
test_data = ax.plot(x_train, y_val, 'g.', x, y, 'b-', linewidth=2, label='Validierungsdaten')
    
handles, labels = ax.get_legend_handles_labels()
ax.legend(loc=2, bbox_to_anchor=(-0.01, 1.082)) #-0.01 1.087
plt.legend(loc=1, bbox_to_anchor=(1.01, 1.13))

plt.show()
