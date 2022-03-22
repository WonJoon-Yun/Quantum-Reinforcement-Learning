import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch.nn as nn
import torchquantum as tq
import gym
import torchquantum.functional as tqf
from torch.optim.lr_scheduler import CosineAnnealingLR
from  torch.distributions import Categorical
from network import QActor, ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

s_dim = 4 
a_dim = 2
gamma = 0.99
n_epochs = 1000
wires_per_block = 2 
a_lr = 1e-4
static = 'store_true'

def _train(epoch, Experience, Pi, PiTarget, optimizer):
    Pi.train()
    PiTarget.eval()
    S, A, R, S_Prime = Experience
    State       = S.reshape(S.shape[0], -1)
    State_Prime = S_Prime.reshape(S_Prime.shape[0], -1)
    Q             = Pi(State).gather(-1, A)
    Qtarget       = PiTarget(State_Prime).max(-1).values
    targets       = R + gamma * Qtarget 
    td_error      = targets.detach() - Q
    loss          = torch.sqrt((td_error ** 2).mean())
    print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(Pi.parameters(), 10)
    optimizer.step()
    if epoch > 0 and epoch % 10 == 0:
        PiTarget.load_state_dict(Pi.state_dict())
        
def train():    
    env =gym.make('CartPole-v1')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    actor = QActor().to(device)
    actor_target = copy.deepcopy(actor)
    actor_target.load_state_dict(actor.state_dict())
    replay     = ReplayBuffer(device)
    Qoptimizer = optim.Adam(actor.parameters(), lr=a_lr)
    Scheduler  = CosineAnnealingLR(Qoptimizer, T_max=n_epochs)

    if static:
        actor.q_layer.static_on(wires_per_block=wires_per_block)
    epsilon = 0.5
    for epoch in range(1, n_epochs + 1):
        
        s = env.reset()
        s = np.array(s)
        s_prime = np.copy(s)
        done = False
        score = 0 
        while not done:
            q_value  = actor(torch.from_numpy(s).to(device, dtype=torch.float))
            if np.random.rand() < epsilon:
                a = np.random.choice([0,1])
            else: 
                action_dist = F.softmax(q_value.squeeze(), dim=-1)
                a = Categorical(action_dist).sample().item()
            s_prime, r, done, _ = env.step(a)
            score += r
            s_prime = np.array(s_prime)
            transition = [s,a,r,s_prime]
            replay.put_data(transition)
            s = s_prime
        
        print(f'[Epoch {epoch}] Reward : {score}, Epsilon: {epsilon}')
        epsilon = max(0, epsilon-0.001)
        experience = replay.make_batch()        
        _train(epoch = epoch,
               Experience=experience, 
               Pi= actor, 
               PiTarget= actor_target, 
               optimizer= Qoptimizer)
        
        Scheduler.step()
        torch.save(actor.state_dict(),'./Qagent.pkl')
            
