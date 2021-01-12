from CONSTANTS import *


class ALGDataset(torch.utils.data.IterableDataset):
    def __init__(self, model, env, sample_size: int = 1):
        self.model = model
        self.env = env
        self.sample_size = sample_size

    def __iter__(self):  # -> Tuple:
        state = self.env.reset()
        log_probs = []
        rewards = []
        states = []
        actions = []
        self.model.eval()
        for steps in range(MAX_LENGTH_OF_A_GAME):
            action, log_prob = self.model.select_action(state)
            new_state, reward, done, _ = self.env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = new_state
            if done:
                break
        self.model.train()
        yield rewards, log_probs, states, actions

    def append(self, experience):
        self.buffer.append(experience)
