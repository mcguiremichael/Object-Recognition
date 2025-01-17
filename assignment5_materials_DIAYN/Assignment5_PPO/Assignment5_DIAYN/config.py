# Hyperparameters for DQN agent, memory and training
EPISODES = 500000
HEIGHT = 84
WIDTH = 84
HISTORY_SIZE = 8
learning_rate = 2.5e-4
evaluation_reward_length = 100
render_breakout = True
batch_size = 32
Update_target_network_frequency = 1000
train_frame = 1024
Memory_capacity = train_frame
clip_param = 0.1
num_envs = 8
env_mem_size = int(train_frame / num_envs)

