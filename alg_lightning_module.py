from CONSTANTS import *
from alg_net import ALGNet


class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.env = gym.make(ENV)
        self.state = self.env.reset()
        self.obs_size = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.net = ALGNet(self.obs_size, self.n_actions)

        # self.agent = Agent()
        # self.total_reward = 0
        # self.episode_reward = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):

        rewards = torch.cat(batch[0]).numpy()
        log_probs = batch[1]
        states = torch.cat(batch[2])
        actions = torch.cat(batch[3])

        discounted_rewards = self._get_discounted_rewards(rewards)
        discounted_rewards = torch.tensor(discounted_rewards)
        # normalize discounted rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        loss = torch.stack(policy_gradient).sum()

        self.log('total_reward', np.sum(rewards))
        self.log('train loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=LR)

    @staticmethod
    def _get_discounted_rewards(rewards):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0.0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + GAMMA ** pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        return discounted_rewards

    def _get_action(self, state) -> int:
        q_values = self.net(state)
        _, action = torch.max(q_values, dim=1)
        return int(action.item())
