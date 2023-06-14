#!/usr/bin/env python

from typing import Any
import rvo2
import random
import pygame
import numpy as np
import itertools
# timestep, neighborDist, maxNeighbors, timeHorizon, 
# timeHorizonObst, radius, maxSpeed, velocity
sim = rvo2.PyRVOSimulator(1/60., 0, 1.5, 5, 1.5, 2, 0.1, 2)

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]

agents = []

# pos = (2, -2)
# vel = (-1, 1)
# a = sim.addAgent(pos)
# sim.setAgentPrefVelocity(a, vel)
# agents.append(a)

class FakeAgent:
    def __init__(self, pos=None, vel=None, target=None, timestep=1/60.):
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

fake_agents = []

for i in range(20):
    pos = (random.uniform(-4, 4), random.uniform(-4, 4))
    target = (-pos[0] + random.uniform(-1, 1), -pos[1] + random.uniform(-1, 1))
    a = sim.addAgent(pos)
    vel = np.array(target) - np.array(pos)
    vel = vel / np.linalg.norm(vel)
    sim.setAgentPrefVelocity(a, tuple(vel.tolist()))
    sim.setAgentTarget(a, target)

    # print(i, sim.getAgentPeturb(a))
    # if i % 10 == 0:
    #     sim.setAgentPeturb(a, 2.5)

    agents.append(a)

# for i in range(10):
#     pos = (random.uniform(-4, 4), random.uniform(-4, 4))
#     target = (-pos[0] + random.uniform(-1, 1), -pos[1] + random.uniform(-1, 1))
#     vel = np.array(target) - np.array(pos)
#     vel = vel / np.linalg.norm(vel)
    
#     a = FakeAgent(pos, vel, target, 1/60.)

#     # print(i, sim.getAgentPeturb(a))
#     # if i % 10 == 0:
#     #     sim.setAgentPeturb(a, 2.5)

#     fake_agents.append(a)

# a0 = sim.addAgent((0, 0))
# sim.setAgentTarget(a0, (0, 0))
# agents.append(a0)

# b = sim.addAgent((2, 2))
# sim.setAgentTarget(b, (-1, -1))
# sim.setAgentPrefVelocity(b, (-1, -1))
# sim.setAgentPeturb(b, 0.5)
# agents.append(b)


# Pass either just the position (the other parameters then use
# the default values passed to the PyRVOSimulator constructor),
# or pass all available parameters.
# a0 = sim.addAgent((-2, -2))
# a1 = sim.addAgent((2, 0))
# a2 = sim.addAgent((2, 2))

# # position neighborDist maxNeighbors timeHorizon 
# # timeHorizonObst radius maxSpeed velocity
# a3 = sim.addAgent((0, 2), 1.5, 5, 1.5, 2, 0.1, 2, (0, 0))

# # Obstacles are also supported.


# sim.setAgentPrefVelocity(a0, (0.5, 0.5))
# sim.setAgentPrefVelocity(a1, (-0.5, 0.5))
# sim.setAgentPrefVelocity(a2, (-0.5, -0.5))
# sim.setAgentPrefVelocity(a3, (0.5, -0.5))

# sim.setAgentTarget(a0, (2, 2))
# sim.setAgentTarget(a1, (0, 2))
# sim.setAgentTarget(a2, (0, 0))
# sim.setAgentTarget(a3, (2, 0))


obs_pos = [(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.2)]
# o1 = sim.addObstacle(obs_pos)
# sim.processObstacles()
# Make agent 0 much less collaborative (nominally does 0.5 of the avoidance)
# sim.setAgentCollabCoeff(a0, 0.1)

print('Simulation has %i agents and %i obstacle vertices in it.' %
      (sim.getNumAgents(), sim.getNumObstacleVertices()))

print('Running simulation')

# for step in range(100):
#     sim.doStep()

#     positions = ['(%5.3f, %5.3f)' % sim.getAgentPosition(agent_no)
#                  for agent_no in (a0, a1, a2, a3)]
#     print('step=%2i  t=%.3f  %s' % (step, sim.getGlobalTime(), '  '.join(positions)))

pygame.init()
dim = (640, 480)
origin = np.array(dim) / 2
scale = 6

screen = pygame.display.set_mode(dim)

clock = pygame.time.Clock()

def draw_agent(pos, radius, color):
    pygame.draw.circle(
        screen, 
        color, 
        np.rint(pos * scale + origin).astype(int), 
        int(round(radius * scale)), 0)

def draw_velocity(pos, vel):
    pygame.draw.line(
        screen, 
        pygame.Color(0, 255, 255), 
        np.rint(pos * scale + origin).astype(int), 
        np.rint((pos + vel) * scale + origin).astype(int), 1)
    
def draw_pref_vel(pos, vel):
    pygame.draw.line(
        screen, 
        pygame.Color(255, 0, 255), 
        np.rint(pos * scale + origin).astype(int), 
        np.rint((pos + vel) * scale + origin).astype(int), 1)


while True:
    clock.tick(60)
    sim.doStep()

    for a in fake_agents:
        a.update()

    screen.fill(pygame.Color(0, 0, 0))

    pygame.draw.polygon(screen, pygame.Color(255, 255, 255), np.array(obs_pos) * 10 * scale + origin, 1)
    # pygame.draw.polygon(screen, pygame.Color(255, 255, 255), [[0, 0], [10, 10], [10, 0]], 1)

    for i, (a, color) in enumerate(zip(agents, itertools.cycle(colors))):
        pos = np.array(sim.getAgentPosition(a)) * 10
        vel = np.array(sim.getAgentVelocity(a)) * 10
        radius = sim.getAgentRadius(a) * 10

        draw_agent(pos, radius, color)
        draw_velocity(pos, vel)
        draw_pref_vel(pos, np.array(sim.getAgentPrefVelocity(a)) * 10)

        # if i % 10 != 0 and not sim.getAgentEnd(a):
        #     print(i)
    
    for a in fake_agents:
        pos = a.pos * 10
        vel = a.vel * 10
        radius = 0.1 * 10

        draw_agent(pos, radius, pygame.Color(255, 255, 255))
        draw_velocity(pos, vel)
        # draw_pref_vel(pos, a.target_vel * 10)

    # print(sim.getAgentEnd(a0))
    # print(sim.getAgentVelocity(a0))

    pygame.display.flip()
    # pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False