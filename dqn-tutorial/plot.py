import matplotlib
import matplotlib.pyplot as plt
import torch

import load_hyperparameters # EPISODE_DURATIONS

EPISODE_DURATIONS = load_hyperparameters.EPISODE_DURATIONS

###

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(EPISODE_DURATIONS, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


if __name__ == 'plot':
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()
