import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)

# 经验Buffer
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)  # 限位

    # 随机采样
    def sample(self, batch_size):
        # 随机抽取要训练的数据
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, n_latent_var):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, n_latent_var)
        self.l2 = nn.Linear(n_latent_var, n_latent_var)
        self.l3 = nn.Linear(n_latent_var, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a)) # 通过tanh后再进行尺寸缩放成实际Action值


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.l2 = nn.Linear(n_latent_var, n_latent_var)
        self.l3 = nn.Linear(n_latent_var, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.l5 = nn.Linear(n_latent_var, n_latent_var)
        self.l6 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            n_latent_var,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action, n_latent_var).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, n_latent_var).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # 从buffer里随机采样数据
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # 添加clipped噪声
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # 计算Q_target_value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # 取min
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # 得到当前Q估计
        current_Q1, current_Q2 = self.critic(state, action)

        # 计算当前Q_critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 求梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟policy更新（actor更新慢）
        if self.total_it % self.policy_freq == 0:

            # 计算actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # 求梯度
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新target网络参数
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


def main():
    ############## 参数设定 ##############
    env_name = "Pendulum-v0"
    # 生成仿真环境
    env = gym.make(env_name)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    render = True

    seed = 0  # 设置随机数种子
    n_latent_var = 256  # 隐藏层数量
    start_timesteps = 25e3  # 初始随机步数
    eval_freq = 5e3  # How often (time steps) we evaluate
    max_timesteps = 1e6  # 最大步数Max time steps to run environment
    expl_noise = 0.1  # 高斯噪声
    batch_size = 256  # 采样数据个数
    discount = 0.99  # 折扣因子
    tau = 0.005  # 目标网络更新比例
    policy_noise = 0.2  # 更新critic时添加到target_policy的噪声Noise added to target policy during critic update
    noise_clip = 0.5  # 添加到target_policy的噪声范围
    policy_freq = 2  # policy延迟更新频率（actor）
    #############################################

    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "n_latent_var": n_latent_var,
        "discount": discount,
        "tau": tau,
        # Target policy 平滑处理
        "policy_noise": policy_noise * max_action,
        "noise_clip": noise_clip * max_action,
        "policy_freq": policy_freq
    }

    # 初始化policy
    policy = TD3(**kwargs)
    # 初始化buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    # 参数初始化
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # 主循环
    for t in range(int(max_timesteps)):
        state, done = env.reset(), False
        episode_timesteps += 1

        # 选择Action
        if t < start_timesteps: # 初始先随机跑，存数据
            action = env.action_space.sample()  # 随机从动作空间中选取动作
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * expl_noise, size=action_dim)  # 加入噪声
            ).clip(-max_action, max_action)

        # 得到（新的状态，奖励，是否终止，额外的调试信息）
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # 保存数据
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward
        env.render()

        # 训练policy
        if t >= start_timesteps:  # 收集到足够多的数据
            policy.train(replay_buffer, batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # 数据清零
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # # Evaluate episode
        # if (t + 1) % eval_freq == 0:
        #     evaluations.append(eval_policy(policy, env, seed))
        #     np.save(f"./results/{file_name}", evaluations)
        #     if save_model: policy.save(f"./models/{file_name}")


if __name__ == '__main__':
    main()
