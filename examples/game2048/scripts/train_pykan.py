#!/usr/bin/env python3
"""
2048 DQN training using pykan (efficient-kan) for comparison with ArKan.
Same hyperparameters as ArKan version.
OPTIMIZED VERSION - less training per step
"""

import random
import numpy as np
from collections import deque
import time

# Try to import efficient-kan or kan
try:
    from efficient_kan import KAN
    print("Using efficient-kan")
except ImportError:
    try:
        from kan import KAN
        print("Using pykan")
    except ImportError:
        print("Please install: pip install efficient-kan")
        print("Or: pip install pykan")
        exit(1)

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============== 2048 Game ==============

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.add_random_tile()
        self.add_random_tile()
    
    def add_random_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r, c = random.choice(empty)
            self.board[r, c] = 2 if random.random() < 0.9 else 4
    
    def slide_row_left(self, row):
        non_zero = row[row != 0]
        merged = []
        score_add = 0
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged.append(non_zero[i] * 2)
                score_add += non_zero[i] * 2
                skip = True
            else:
                merged.append(non_zero[i])
        merged = np.array(merged + [0] * (4 - len(merged)), dtype=np.int32)
        return merged, score_add
    
    def move(self, direction):
        if self.game_over:
            return 0, False
        
        old_board = self.board.copy()
        score_add = 0
        
        if direction == 0:
            for i in range(4):
                self.board[i], s = self.slide_row_left(self.board[i])
                score_add += s
        elif direction == 1:
            for i in range(4):
                self.board[i] = self.board[i][::-1]
                self.board[i], s = self.slide_row_left(self.board[i])
                self.board[i] = self.board[i][::-1]
                score_add += s
        elif direction == 2:
            self.board = self.board.T
            for i in range(4):
                self.board[i], s = self.slide_row_left(self.board[i])
                score_add += s
            self.board = self.board.T
        elif direction == 3:
            self.board = self.board.T
            for i in range(4):
                self.board[i] = self.board[i][::-1]
                self.board[i], s = self.slide_row_left(self.board[i])
                self.board[i] = self.board[i][::-1]
                score_add += s
            self.board = self.board.T
        
        changed = not np.array_equal(old_board, self.board)
        if changed:
            self.score += score_add
            self.add_random_tile()
            self.check_game_over()
        
        return score_add, changed
    
    def check_game_over(self):
        if 0 in self.board:
            return
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:
                    return
                if self.board[j, i] == self.board[j + 1, i]:
                    return
        self.game_over = True
    
    def valid_actions(self):
        valid = []
        for action in range(4):
            test = Game2048.__new__(Game2048)
            test.board = self.board.copy()
            test.score = self.score
            test.game_over = False
            _, changed = test.move(action)
            if changed:
                valid.append(action)
        return valid if valid else [0]
    
    def max_tile(self):
        return np.max(self.board)
    
    def get_state(self):
        """Simple normalized state: 16 features"""
        state = np.zeros(16, dtype=np.float32)
        for i in range(4):
            for j in range(4):
                val = self.board[i, j]
                if val == 0:
                    state[i * 4 + j] = -1.0
                else:
                    state[i * 4 + j] = (np.log2(val) / 8.0) - 0.5
        return state


# ============== Simple MLP for speed comparison ==============

class SimpleMLP(nn.Module):
    """Simple MLP to compare with KAN speed"""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# ============== DQN Agent ==============

class DQNAgent:
    def __init__(self, state_dim=16, action_dim=4, lr=0.001, gamma=0.99, use_kan=True):
        self.use_kan = use_kan
        
        if use_kan:
            # KAN network: [16, 32, 16, 4] - smaller for speed
            self.policy_net = KAN(
                layers_hidden=[state_dim, 32, 16, action_dim],
                grid_size=5,
                spline_order=3,
            ).to(device)
            
            self.target_net = KAN(
                layers_hidden=[state_dim, 32, 16, action_dim],
                grid_size=5,
                spline_order=3,
            ).to(device)
        else:
            # Simple MLP for comparison
            self.policy_net = SimpleMLP(state_dim, [64, 32], action_dim).to(device)
            self.target_net = SimpleMLP(state_dim, [64, 32], action_dim).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim
    
    def select_action(self, state, game, epsilon):
        valid_actions = game.valid_actions()
        
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_t).cpu().numpy()[0]
        
        best_action = valid_actions[0]
        best_q = q_values[best_action]
        for a in valid_actions:
            if q_values[a] > best_q:
                best_q = q_values[a]
                best_action = a
        return best_action
    
    def train_batch(self, states, actions, rewards, next_states, dones):
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).to(device)
        
        current_q = self.policy_net(states_t)
        current_q_actions = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states_t)
            max_next_q = next_q.max(1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)
        
        loss = nn.MSELoss()(current_q_actions, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ============== Replay Buffer ==============

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


# ============== Training ==============

def train(
    episodes=3000,
    batch_size=32,
    lr=0.001,
    gamma=0.99,
    replay_capacity=50000,
    target_update=100,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.998,
    use_kan=True,
):
    net_type = "KAN" if use_kan else "MLP"
    print("=" * 60)
    print(f"    2048 DQN Training with {net_type}")
    print("=" * 60)
    print(f"Episodes:       {episodes}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {lr}")
    print(f"Network:        {net_type}")
    print()
    
    agent = DQNAgent(state_dim=16, action_dim=4, lr=lr, gamma=gamma, use_kan=use_kan)
    buffer = ReplayBuffer(replay_capacity)
    
    epsilon = epsilon_start
    best_score = 0
    best_tile = 0
    total_steps = 0
    train_steps = 0
    
    scores = []
    
    start_time = time.time()
    
    for ep in range(1, episodes + 1):
        game = Game2048()
        
        while not game.game_over:
            state = game.get_state()
            action = agent.select_action(state, game, epsilon)
            
            score_gained, changed = game.move(action)
            reward = score_gained / 100.0 if changed else -0.5
            
            next_state = game.get_state()
            done = game.game_over
            
            buffer.push(state, action, reward, next_state, float(done))
            total_steps += 1
        
        # Train ONCE per episode (not per step!)
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            agent.train_batch(states, actions, rewards, next_states, dones)
            train_steps += 1
        
        scores.append(game.score)
        if game.score > best_score:
            best_score = game.score
        if game.max_tile() > best_tile:
            best_tile = game.max_tile()
        
        epsilon = max(epsilon * epsilon_decay, epsilon_end)
        
        if ep % target_update == 0:
            agent.update_target()
        
        if ep % 48 == 0 or ep == 1:
            elapsed = time.time() - start_time
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            eps_per_sec = ep / elapsed
            print(f"Ep {ep:5} | Best: {best_score:5} | Tile: {best_tile:4} | "
                  f"Avg: {avg_score:6.1f} | ε: {epsilon:.3f} | {eps_per_sec:.1f} ep/s")
    
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("              TRAINING COMPLETE")
    print("=" * 60)
    print(f"Training time:      {elapsed:.1f}s")
    print(f"Episodes/second:    {episodes / elapsed:.1f}")
    print(f"Best score:         {best_score}")
    print(f"Best tile:          {best_tile}")
    print(f"Average score:      {np.mean(scores):.1f}")
    print()
    
    # Test 100 games
    print("Testing trained agent (100 games):")
    test_scores = []
    test_tiles = []
    
    for _ in range(100):
        game = Game2048()
        while not game.game_over:
            state = game.get_state()
            action = agent.select_action(state, game, 0.0)
            game.move(action)
        test_scores.append(game.score)
        test_tiles.append(game.max_tile())
    
    print(f"Average score:      {np.mean(test_scores):.1f}")
    print(f"Best score:         {max(test_scores)}")
    print(f"Best tile:          {max(test_tiles)}")
    print()
    print("Tile achievements (out of 100 games):")
    print(f"  256+:  {sum(1 for t in test_tiles if t >= 256)} games")
    print(f"  512+:  {sum(1 for t in test_tiles if t >= 512)} games")
    print(f"  1024+: {sum(1 for t in test_tiles if t >= 1024)} games")
    print(f"  2048:  {sum(1 for t in test_tiles if t >= 2048)} games")


if __name__ == "__main__":
    import sys
    use_kan = "--mlp" not in sys.argv
    
    train(
        episodes=3000,
        batch_size=32,
        lr=0.001,
        gamma=0.99,
        replay_capacity=50000,
        target_update=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.998,
        use_kan=use_kan,
    )
