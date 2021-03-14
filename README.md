# Deep_reinforcement_learning_Unity_Tennis
This notebook uses deep reinforcement learning technique - Multuple Agent Deep Deterministic Policy Gradient to solve Unity Tennis environment. The environment is considered solved if the highest score between the two agents (average over 100 episodes) is >0.5. The code is in PyTorch (v0.4) and Python 3.

## Unity Tennis Environment
The environment has two agents. Each agent has an observation space consisting of 8 variables and an action space consisting of two continuous variables (corresponding to moving toward/away from the net and jumping). The agents receive 0.1 reward for hitting the ball over the net and -0.01 reward for letting a ball hit the ground or hitting the ball out of bounds. 

<img src="https://github.com/ccakarolotw/Deep_reinforcement_learning_Unity_Tennis/blob/main/untrained.gif" width ="500">

## Dependencies
Installation of dependencies follow https://github.com/udacity/deep-reinforcement-learning#dependencies
1. Create (and activate) a new environment with Python 3.6.

`conda create --name drlnd python=3.6 
activate drlnd`

2.  Install pytorch 0.4.0

`conda install pytorch=0.4.0 -c pytorch`

3. Download Unity environment and place it in the same folder as the jupyter notebook `Tennis.ipynb`

## Multiple Agent Deep Deterministic Policy Gradient (MADDPG)
In MADDPG, each agent's policy is based on DDPG algorithm. MADDPG adopts the framework of centralized training with decentralized execution. The critics of each agent is trained on the observations and actions of both agents, the actors of each agent is trained only on its own observations. 

### Deep Deterministic Policy Gradient (DDPG) Agent

The agent consists of a critic network and an actor network. The critic network learns to approximate the state(of both agents)-action(of both agents) value function (Q). The actor network learns to choose actions based on input state(of itself) to maximize the expected value given by the critic network. 

The critic network is updated through TD learning. 

Q(state, action) = reward + GAMMA* Q'(next_state,next_action)

Q' is the target Q network. 
The critic and target critic network is represented by a neural network with three fully connected layers.
```x_state = nn.Linear(2*state_dim,hidden_dim)(state)
x_state = F.leaky_relu(x_state)
x = torch.cat(x_state, actions)
x = nn.Linear(hidden_dim+2*action_dim, hidden_dim_1)(x)
x = F.relu(x)
x = nn.Linear(hidden_dim_1, 1)(x)
```

The actor network is represented by a neural network with three fully connected layers.
```x = nn.Linear(state_dim,hidden_dim)(state)
x = F.relu(x)
x = nn.Linear(hidden_dim, hidden_dim_1)(x)
x = F.relu(x)
x = nn.Linear(hidden_dim_1, action_dim)(x)
x = F.tanh(x)
```

### Hyperparameters
- GAMMA=0.99
- hidden_dim = 128
- hidden_dim_1 = 64
- Target critic and target actor network are are updated through soft update (target_ parameter = (1-tau)* target_ parameter + tau* local_ parameter)
- tau = 1E-3
- Batch_size = 128
- Critic optimizer: Adam with learning rate 3E-4
- Actor optimizer: Adam with learning rate 1E-4

## Results
The environment is solved in 3862 episodes.
<img src="https://github.com/ccakarolotw/Deep_reinforcement_learning_Unity_Tennis/blob/main/trained.gif" width ="500">

<img src="https://github.com/ccakarolotw/Deep_reinforcement_learning_Unity_Tennis/blob/main/score.png" width ="500">


