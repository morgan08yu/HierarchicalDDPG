#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

class Config:
    q_target = 0
    #expected_sarsa_target = 1
    def __init__(self):
        self.task_fn = None
        self.task_fn_H = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.network_fn = None
        self.network_fn_H = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.replay_Hierachical = None
        self.random_process_fn = None
        self.discount = None
        self.target_network_update_freq = 0
        self.max_episode_length = 1000000
        self.exploration_steps = 0
        self.logger = None
        self.history_length = 1
        self.test_interval = 100
        self.test_repetitions = None
        self.double_q = False
        self.tag = 'DDPG'
        self.num_workers = 1
        self.worker = None
        self.worker_H = None
        self.update_interval = 1
        self.gradient_clip = 20
        self.entropy_weight = 0.01
        self.gae_tau = 1.0
        self.noise_decay_interval = 0
        self.target_network_mix = 0.001
        self.action_shift_fn = lambda a: a
        self.reward_shift_fn = lambda r: r
        self.reward_weight = 1
        self.hybrid_reward = False
        self.target_type = self.q_target
        self.episode_limit = 0
        self.min_memory_size = 600
        self.master_fn = None
        self.master_optimizer_fn = None
        self.num_heads = 10
        self.min_epsilon = 0
        self.save_interval = 0
        self.max_steps = 0
        self.success_threshold = float('inf')
        self.render_episode_freq = 0
        self.actor_high = None
        self.critic_high = None

class Configration:
    q_target = 0
    #expected_sarsa_target = 1
    def __init__(self):
        self.task_fn = None
        self.task_fn_H = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.network_fn = None
        self.network_fn_H = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.policy_fn = None
        self.replay_fn = None
        self.replay_Hierachical = None
        self.random_process_fn = None
        self.discount = 0.9
        self.target_network_update_freq = 0
        self.max_episode_length = 1000000
        self.exploration_steps = 0
        self.logger = None
        self.history_length = 1
        self.test_interval = 100
        self.test_repetitions = 20
        self.double_q = False
        self.tag = 'DDPG-Hi'
        self.num_workers = 1
        self.worker = None
        self.worker_H = None
        self.update_interval = 1
        self.gradient_clip = 40
        self.entropy_weight = 0.01
        self.gae_tau = 1.0
        self.noise_decay_interval = 0
        self.target_network_mix = 0.01
        self.action_shift_fn = lambda a: a
        self.reward_shift_fn = lambda r: r
        self.reward_weight = 1
        self.hybrid_reward = False
        self.target_type = self.q_target
        self.episode_limit = 0
        self.min_memory_size = 600
        self.master_fn = None
        self.master_optimizer_fn = None
        self.num_heads = 10
        self.min_epsilon = 0
        self.save_interval = 0
        self.max_steps = 0
        self.success_threshold = float('inf')
        self.render_episode_freq = 0
        self.actor_high = None
        self.critic_high = None