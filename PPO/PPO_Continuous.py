import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义数组变量
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # del语句作用在变量上，而不是数据对象上。删除的是变量，而不是数据。
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# 定义Actor Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_std):
        super(ActorCritic, self).__init__()
        # actor网络：input：state_dim;output:action_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh()
        )
        # critic网络：input：state_dim;output:value(dim = 1)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1),
        )
        # 方差
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        # 手动设置异常
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)  # 将state带进网络
        cov_mat = torch.diag(self.action_var).to(device)  # 求方差的对角线数

        dist = MultivariateNormal(action_mean, cov_mat)  # 按照给定的概率分布来进行采样
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        # torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
        # Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input
        cov_mat = torch.diag_embed(action_var).to(device)
        # 生成一个多元高斯分布矩阵
        dist = MultivariateNormal(action_mean, cov_mat)
        # 用这个随机的去逼近真正的选择动作action的高斯分布
        action_logprobs = dist.log_prob(action)
        # log_prob 是action在前面那个正太分布的概率的log ，我们相信action是对的 ，
        # 那么我们要求的正态分布曲线中点应该在action这里，所以最大化正太分布的概率的log， 改变mu,sigma得出一条中心点更加在a的正太分布。
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, n_latent_var):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # 定义新旧policy
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 将定义的policy复制到policy_old

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 标准化rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 格式转换
        # 使用stack可以保留两个信息：[1. 序列] 和 [2. 张量矩阵] 信息，属于【扩张再拼接】的函数；
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

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
    env_name = "Pendulum-v0"
    render = False
    solved_reward = 300  # stop training if avg_reward > solved_reward
    log_interval = 20  # 输出间隔次数
    max_episodes = 10000  # 最大训练episodes
    max_timesteps = 1500 # 单个episode的最大步数
    n_latent_var = 64  # 隐藏层数量
    update_timestep = 2000  # 更新policy的间隔
    action_std = 0.5  # action分布（多元分布）的标准差常量
    K_epochs = 80  # 更新policy次数
    eps_clip = 0.2   # clip参数
    gamma = 0.99  # 折扣因子

    lr = 0.0003  # Adam optimizer参数
    betas = (0.9, 0.999)
    load_model = 1
    save_model = 1
    seed = None
    #############################################

    # 生成仿真环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if seed:
        print("Random Seed: {}".format(seed))
        torch.manual_seed(seed)
        env.seed(seed)
        np.random.seed(seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, n_latent_var)

    # 参数初始化
    running_reward = 0
    avg_length = 0
    time_step = 0

    #加载模型文件
    file_name = f"{'PPO'}_{env_name}_{seed}"
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if load_model:
        policy_file = file_name
        ppo.policy.load_state_dict( torch.load(f"./models/{policy_file}") )
        ppo.policy_old.load_state_dict(torch.load(f"./models/{policy_file}"))


    # 主循环
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        for t in range(max_timesteps):
            time_step += 1

            # 运行policy_old:
            action = ppo.select_action(state, memory)
            state, reward, done, _ = env.step(action)  # 得到（新的状态，奖励，是否终止，额外的调试信息）

            # 保存数据:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # 更新policy
            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
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

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            if save_model:
                torch.save(ppo.policy.state_dict(), f"./models/{file_name}")
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
