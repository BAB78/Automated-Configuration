import numpy as np

class NTPEnv:
    def __init__(self):
        """Initialize NTP environment with initial offset and configuration options."""
        self.offset = 5.0  # Initial offset in ms
        self.step_count = 0  # Track simulation steps
        self.servers = ["192.168.1.1", "192.168.1.2"]  # Sample NTP servers
        self.intervals = [64, 128]  # Poll intervals in seconds

    def step(self, action):
        """Simulate a step: update offset based on action (server and interval choice)."""
        self.step_count += 1
        server_idx, interval_idx = divmod(action, len(self.intervals))  # Decode action
        server = self.servers[server_idx]
        interval = self.intervals[interval_idx]
        # Simulate offset change: random noise minus reduction (better for server 0)
        self.offset += np.random.normal(0, 0.1) - (0.05 if server_idx == 0 else 0.02)
        reward = -abs(self.offset)  # Negative offset as reward (minimize offset)
        done = self.step_count >= 100  # End after 100 steps
        return self.offset, reward, done, {"server": server, "interval": interval}

def q_learning_agent(env, episodes=100):
    """Train Q-learning agent to find optimal NTP configuration."""
    q_table = np.zeros((len(env.servers) * len(env.intervals), 100))  # Q-table for actions and states
    alpha, gamma, epsilon = 0.1, 0.9, 0.1  # Learning parameters
    for _ in range(episodes):
        state = 0
        env = NTPEnv()  # Reset environment
        while not env.step_count >= 100:
            if np.random.random() < epsilon:
                action = np.random.randint(0, len(q_table))  # Exploration
            else:
                action = np.argmax(q_table[:, state])  # Exploitation
            offset, reward, done, info = env.step(action)
            new_state = min(int(abs(offset) * 10), 99)  # State based on offset
            q_table[action, state] += alpha * (reward + gamma * np.max(q_table[:, new_state]) - q_table[action, state])
            state = new_state
    best_action = np.argmax(q_table[:, 0])
    server_idx, interval_idx = divmod(best_action, len(env.intervals))
    return env.servers[server_idx], env.intervals[interval_idx]

# Run and configure
best_server, best_interval = q_learning_agent(NTPEnv())
print(f"Recommended NTP Configuration: Server={best_server}, Poll Interval={best_interval}s")
