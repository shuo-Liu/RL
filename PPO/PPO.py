import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#1

# 定义数组变量
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# 定义Actor Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor网络：input：state_dim;output:action_dim
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic网络：input：state_dim;output:value(dim = 1)
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)  # 将state带进网络
        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # 定义新旧policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict()) # 将定义的policy复制到policy_old

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # 蒙特卡洛状态奖励估计
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)  # 乘以一个折扣因子gamma
            rewards.insert(0, discounted_reward)

        # 标准化rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # 标准化rewards

        # 格式转换
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # 训练policy K_epochs次
        for _ in range(self.K_epochs):
            # 用当前policy评估old_states和old_actions
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 计算ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 计算梯度
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 将新的权重更新给old_policy
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## 参数设定 ##############
    env_name = "CartPole-v0"
    # 生成仿真环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    render = True

    log_interval = 20  # 输出间隔次数
    max_episodes = 50000  # 最大训练episodes
    max_timesteps = 300  # 单个episode的最大步数
    n_latent_var = 64  # 隐藏层数量
    update_timestep = 2000  # 更新policy的间隔
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # 折扣因子
    K_epochs = 4  # 更新policy次数
    eps_clip = 0.2  # clip参数
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)


    # 参数初始化
    running_reward = 0
    avg_length = 0
    timestep = 0

    # 主循环
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()  # 初始化（重新玩）
        for t in range(max_timesteps):
            timestep += 1

            # 运行policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)  # 得到（新的状态，奖励，是否终止，额外的调试信息）

            # 保存数据:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # 更新policy
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t



        # 打印log
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
