import numpy as np
import re
import click
from matplotlib import pylab as plt
import matplotlib

@click.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def main(files):
    plt.style.use('classic')
    matplotlib.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots()
    plt.grid(linestyle=':', linewidth=1, color='0.5')
    ax2 = ax1.twinx()
    ax1.set_xlabel('Iteration', fontsize=25, labelpad=20)
    ax1.set_ylabel('Fehler', fontsize=25, labelpad=20)
    ax2.set_ylabel('mAP (in %)', fontsize=25, labelpad=20)

    ax1.set_yticks(np.arange(0,21,2))
    ax1.set_ylim([0, 21])
    ax2.set_yticks(np.arange(0,101,10))
    ax2.set_ylim([0, 105])
    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations2, accuracies2, accuracies_iteration_checkpoints_ind2 = parse_log(log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations2, accuracies2, accuracies_iteration_checkpoints_ind2, color_ind=i)
    plt.show()


def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    #accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracy_pattern = r"Test net output #0: detection_eval = (?P<detection_eval>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracy_pattern2 = r"Test net output #1: detection_eval = (?P<detection_eval>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracies2 = []
    accuracy_iterations = []
    accuracy_iterations2 = []
    accuracies_iteration_checkpoints_ind = []
    accuracies_iteration_checkpoints_ind2 = []
	
    i = 0
    for r in re.findall(accuracy_pattern, log):      
        iteration = i*50
        i += 1 
        accuracy = float(r[0]) * 100

        if iteration % 10 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)

    i = 0
    for r in re.findall(accuracy_pattern2, log):      
        iteration = i*50
        i += 1 
        accuracy = float(r[0]) * 100

        if iteration % 10 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind2.append(len(accuracy_iterations2))

        accuracy_iterations2.append(iteration)
        accuracies2.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)

    accuracy_iterations2 = np.array(accuracy_iterations2)
    accuracies2 = np.array(accuracies2)

    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations2, accuracies2, accuracies_iteration_checkpoints_ind2


def disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, accuracy_iterations2, accuracies2, accuracies_iteration_checkpoints_ind2, color_ind=0):
    #handles, labels = ax2.get_legend_handles_labels()
    #ax2.legend(handles, labels)
    loss = ax1.plot(loss_iterations, losses, linewidth=2, color='r', label='Kostenfunktion')
    train_data = ax2.plot(accuracy_iterations2, accuracies2, 'k', linewidth=2, label='Trainingsdaten')
    test_data = ax2.plot(accuracy_iterations, accuracies, 'b', linewidth=2, label='Validierungsdaten')
    dots_test = ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[accuracies_iteration_checkpoints_ind], 'o', color='b')
    dots_train = ax2.plot(accuracy_iterations2[accuracies_iteration_checkpoints_ind2], accuracies2[accuracies_iteration_checkpoints_ind2], 'o', markersize=5, color='k')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)
    ax1.legend(loc=2, bbox_to_anchor=(-0.01, 1.082)) #-0.01 1.087
    plt.legend(loc=1, bbox_to_anchor=(1.01, 1.13))

if __name__ == '__main__':
	main()
