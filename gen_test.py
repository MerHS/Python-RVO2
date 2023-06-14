import random
from pathlib import Path

from vae import VanillaVAE
import numpy as np
import rvo2
import torch
import cv2
from tqdm import tqdm

VAL_SIZE = 100
TRAIN_SIZE = 3000

WIDTH = 10
HEIGHT = 10
RADIUS = 0.1
VIS_RANGE = 2
TIMESTEP = 1/30.

# val_path = Path('data/val')
# train_path = Path('data/train')

# if not train_path.exists():
#     train_path.mkdir()
# if not val_path.exists():
#     val_path.mkdir()

ckpt = torch.load('last.ckpt')
model = VanillaVAE(1, 24)
model.load_state_dict({k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()})
model.eval()
model.cuda()

class FakeAgent:
    def __init__(self, pos=None, vel=None, target=None, timestep=TIMESTEP):
        self.pos = np.array(pos) if pos is not None else np.zeros(2)
        self.vel = np.array(vel) if vel is not None else np.zeros(2)
        self.target = np.array(target) if target is not None else np.zeros(2)
        self.end = False
        self.timestep = timestep

    def update(self):
        if self.end:
            return
        self.pos += self.vel * self.timestep
        if np.linalg.norm(self.pos - self.target) < 0.05:
            self.end = True


def gauss(len, mu, sigma):
    x, y = np.meshgrid(np.linspace(-1,1,len), np.linspace(-1,1,len))
    d = np.sqrt(x*x+y*y)
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return g

gauss_200 = gauss(200, 0, 0.05)
gauss_100 = gauss_200[50:150, 50:150]

def gen_observation(n_agents, n_fakes=0):
    sim = rvo2.PyRVOSimulator(TIMESTEP, 1.5, 5, 1.5, 2, RADIUS, 2)

    len_agents = n_agents + n_fakes
    agents = []
    targets = []
    obs = [[] for _ in range(len_agents)] # (position, latent, action, reward)

    old_pos = []
    old_done = []

    for i in range(len_agents):
        target_done = True
        while target_done:
            target_done = False

            pos = np.array([random.uniform(1, WIDTH - 1), random.uniform(1, HEIGHT - 1)])
            target = np.array([WIDTH - pos[0] + random.uniform(-1, 1), HEIGHT - pos[1] + random.uniform(-1, 1)])

            for curr_pos in old_pos:
                if np.linalg.norm(pos - curr_pos) < 2 * RADIUS:
                    target_done = True
                    break
            
            for curr_target in targets:
                if np.linalg.norm(target - curr_target) < 2 * RADIUS:
                    target_done = True
                    break
            
        targets.append(target)
        old_pos.append(pos)
        old_done.append(False)
        

        vel = np.array(target) - np.array(pos)
        vel = vel / (np.linalg.norm(vel) + 1e-6)

        if i < n_agents:
            agent = sim.addAgent(tuple(pos.tolist()))
            sim.setAgentPrefVelocity(agent, tuple(vel.tolist()))
            sim.setAgentTarget(agent, tuple(target.tolist()))
        else:
            agent = FakeAgent(pos, vel, target)

        agents.append(agent)

    for time in range(30 * 10): # 10 sec timeout
        sim.doStep()

        new_pos = []
        vel_list = []
        done_list = []

        plane_w = int((WIDTH + 2 * VIS_RANGE) / VIS_RANGE * 100)
        plane_h = int((HEIGHT + 2 * VIS_RANGE) / VIS_RANGE * 100)
        plane = np.zeros((plane_w, plane_h))
        
        for agent in agents[:n_agents]:
            pos = np.array(sim.getAgentPosition(agent))
            vel = np.array(sim.getAgentVelocity(agent))
            done = sim.getAgentEnd(agent)

            new_pos.append(pos)
            vel_list.append(vel)
            done_list.append(done)
        
        for agent in agents[n_agents:]:
            pos = agent.pos
            vel = agent.vel
            done = agent.end

            new_pos.append(pos)
            vel_list.append(vel)
            done_list.append(done)

        planes = [plane.copy() for _ in range(len_agents)]
        for i, plane in enumerate(planes):
            if old_done[i]:
                continue
            for j, pos in enumerate(old_pos):
                if i == j:
                    continue
                pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
                pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)

                pos_x = int(np.clip(pos_x, 100, plane_w - 100))
                pos_y = int(np.clip(pos_y, 100, plane_h - 100))
                # print(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50].shape, pos, pos_x, pos_y)
                plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50] = np.maximum(plane[pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50], gauss_100)

        # img = Image.fromarray(np.uint8(plane[100:-100, 100:-100] * 255))
        # img.save('test.png')
        # break

        # cv2.imwrite('test.png', plane * 255)
        
        data = []
        tensors = []
        for i in range(len_agents):
            if old_done[i]:
                continue
            
            dist_diff = np.linalg.norm(old_pos[i] - targets[i]) - np.linalg.norm(new_pos[i] - targets[i])
            
            reward = dist_diff * 0.3 # max 0.3

            coll_reward = 0
            for j in range(n_agents):
                if i == j:
                    continue
                if np.linalg.norm(new_pos[i] - new_pos[j]) < 2 * RADIUS:
                    coll_reward = -1 # collision
                    break
            
            reward += coll_reward
            pos = old_pos[i]
            vel = vel_list[i]
            
            pos_x = int((pos[0] + VIS_RANGE) / VIS_RANGE * 100)
            pos_y = int((pos[1] + VIS_RANGE) / VIS_RANGE * 100)
            pos_x = int(np.clip(pos_x, 100, plane_w - 100))
            pos_y = int(np.clip(pos_y, 100, plane_h - 100))
            view = planes[i][pos_x - 50:pos_x + 50, pos_y - 50:pos_y + 50]

            view = cv2.resize(view, (64, 64))
            view_tensor = torch.from_numpy(view).unsqueeze(0).unsqueeze(0).float().cuda()
        
            # obs[i].append((pos, latent, vel, reward))
            tensors.append(view_tensor)
            data.append((i, pos, vel, reward, new_pos[i], done_list[i]))

        with torch.no_grad():
            view_tensor = torch.cat(tensors, dim=0)
            latents = model.reparameterize(*model.encode(view_tensor)).cpu().detach().numpy()

        for l_pos, (i, pos, vel, reward, next_pos, done) in enumerate(data):
            obs[i].append((pos, latents[l_pos], vel, reward, next_pos, done))

        old_pos = new_pos
        old_done = done_list

        if all(done_list):
            break
    
    return obs, targets

if __name__ == '__main__':
    pareto = np.clip(np.int32((np.random.pareto(2, TRAIN_SIZE // 3) + 1) * 3.2), 0, 20)
    total_n = 0

    val_file = 'val.pkl'
    train_file = 'train.pkl'

    train_obs = []
    val_obs = []

    for n_agents in tqdm(pareto):
        # n_fake = np.random.randint(0, max(n_agents // 3, 5))
        n_fake = 0
        obs, targets = gen_observation(n_agents, n_fake)

        train_obs.append((obs, targets))

        total_n += n_agents
        if total_n > TRAIN_SIZE:
            break

    torch.save(train_obs, train_file)

    total_n = 0
    for n_agents in tqdm(pareto):
        n_fake = 0
        # n_fake = np.random.randint(0, max(n_agents // 3, 5))
        obs, targets = gen_observation(n_agents, n_fake)

        val_obs.append((obs, targets))

        total_n += n_agents
        if total_n > VAL_SIZE:
            break

    torch.save(val_obs, val_file)