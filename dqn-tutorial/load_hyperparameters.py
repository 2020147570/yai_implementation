import torch
import yaml


def __load_hyperparameters(config_file):
    with open(config_file, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


def get_steps_done():
    return STEPS_DONE


def get_episode_durations():
    return EPISODE_DURATIONS


def update_episode_durations(each_episode_duration):
    global EPISODE_DURATIONS
    EPISODE_DURATIONS.append(each_episode_duration)


def update_steps_done():
    global STEPS_DONE
    STEPS_DONE = STEPS_DONE + 1


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

    EPISODE_DURATIONS = []
    STEPS_DONE = 0
    
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        NUM_EPISODES = 1000
    else:
        NUM_EPISODES = 50
