import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import time

def get_train_loss(line):
    splitted_line = line.split(" ")
    return float(splitted_line[2]), float(splitted_line[4])

def get_val_loss(line):
    splitted_line = line.split(" ")
    if len(splitted_line)>19:
        return float(splitted_line[19])
    return None

def read(logfile):
    with open(logfile) as f:
        train_y = []
        val_y = []
        train_epoches = []
        val_epoches = []

        while True:
            line = f.readline()

            if line:
                epoch, train_data = get_train_loss(line)
                val_data = get_val_loss(line)
                train_y.append(train_data)
                train_epoches.append(epoch)

                if val_data is not None:
                    val_y.append(val_data)
                    val_epoches.append(epoch)
                
                yield train_epoches, train_y, val_epoches, val_y
            else:

                time.sleep(0.1)

def main():
    if len(sys.argv)<2:
        print("Usage: python %s [training log file]" % sys.argv[0])
        return

    log_file = sys.argv[1]

    fig, ax = plt.subplots()
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.grid()

    train_line, = ax.plot([], [])
    val_line, = ax.plot([], [])
    train_line.set_label("Train")
    val_line.set_label("Val")
    ax.legend()
    def animate(values):
        train_x, train_y, val_x, val_y = values
        print(train_x[-1], train_y[-1])
        train_line.set_data(train_x, train_y)
        val_line.set_data(val_x, val_y)

        ax.set_xlim([train_x[0]-1, train_x[-1]])
        max_y = max(train_y)
        min_y = min(train_y)
        if val_y:
            max_y = max(max_y, max(val_y))
            min_y = min(min_y, min(val_y))
        max_y = min(max_y, 10)
        ax.set_ylim([min_y, max_y])

    ani = FuncAnimation(fig, animate, frames=read(log_file), interval=1)
    plt.show()
 
if __name__ == '__main__':
    main()

