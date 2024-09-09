import torch
import yaml

###

def __load_hyperparameters(config_file):
    with open(config_file, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


if __name__ == 'load_hyperparameters':
    config_file = 'conf.yaml'
    hyperparameters = __load_hyperparameters(config_file)

    BATCH_SIZE = int(hyperparameters['batch_size'])
    GAMMA = float(hyperparameters['gamma'])

    EPS_START = float(hyperparameters['eps_start'])
    EPS_END = float(hyperparameters['eps_end'])
    EPS_DECAY = int(hyperparameters['eps_decay'])

    TAU = float(hyperparameters['tau'])
    LR = float(hyperparameters['lr'])

    STEPS_DONE = 0
    EPISODE_DURATIONS = []

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        NUM_EPISODES = 600
    else:
        NUM_EPISODES = 50
