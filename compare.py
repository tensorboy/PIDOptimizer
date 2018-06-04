import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from collections import OrderedDict

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
   

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))         
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

def plot_overlap(logger, names, axs, indexes):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        axs[indexes[0],indexes[1]].plot(x, np.asarray(numbers[name]))
  
    return [logger.title for name in names]       
    
class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names, axs, indexes):

        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names, axs, indexes)
        
        axs[indexes[0],indexes[1]].set_title(names[0])
        return legend_text
        
paths = OrderedDict()
fields = [['Train Loss'],['Valid Loss'], ['Train Acc.'],['Valid Acc.']]  

paths['SGD-Momentum']='momentum.txt'
I=float(1)
learning_rate = 0.01

name = 'pid.txt'
paths['PID']=name 
fig, axs = plt.subplots(2, 2)   

i = 0
indexes=[0,0]

field = fields[i]
monitor = LoggerMonitor(paths)
legend_text = monitor.plot(names=field, axs=axs, indexes=indexes)
axs[indexes[0],indexes[1]].legend(legend_text, bbox_to_anchor=(1., 1.0), loc=1, borderaxespad=0.)
axs[indexes[0],indexes[1]].set_xlabel('Epoch')

i = 1
indexes=[0,1]
field = fields[i]
monitor = LoggerMonitor(paths)
legend_text = monitor.plot(names=field, axs=axs, indexes=indexes)
axs[indexes[0],indexes[1]].legend(legend_text, bbox_to_anchor=(1., 1.0), loc=1, borderaxespad=0.)
axs[indexes[0],indexes[1]].set_xlabel('Epoch')

i = 2
indexes=[1,0]
field = fields[i]
monitor = LoggerMonitor(paths)
legend_text = monitor.plot(names=field, axs=axs, indexes=indexes)
axs[indexes[0],indexes[1]].legend(legend_text, bbox_to_anchor=(1., 0.), loc=4, borderaxespad=0.)
axs[indexes[0],indexes[1]].set_xlabel('Epoch')

i = 3
indexes=[1,1]
field = fields[i]
monitor = LoggerMonitor(paths)
legend_text = monitor.plot(names=field, axs=axs, indexes=indexes)
axs[indexes[0],indexes[1]].legend(legend_text, bbox_to_anchor=(1., 0.), loc=4, borderaxespad=0.)
axs[indexes[0],indexes[1]].set_xlabel('Epoch')
 
fig.tight_layout()

savefig('moment_vs_pid.jpg')
