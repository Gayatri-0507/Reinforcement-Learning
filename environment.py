# environment.py

import numpy as np
import random

class Highway:
    def __init__(self):
        self.width = 140
        self.height = 700
        self.grid = np.zeros((7, 70))  # Occupancy grid

class Vehicle:
    def __init__(self, lane, speed, position):
        self.lane = lane
        self.speed = speed
        self.position = position
        self.top_speed = 80

    def move(self):
        self.position[1] += self.speed

class Simulation:
    def __init__(self, highway):
        self.highway = highway
        self.vehicles = [Vehicle(random.randint(0, 6), random.randint(65, 80), [0, random.randint(0, highway.height)]) for _ in range(20)]

    def run_step(self, action):
        ego_vehicle = self.vehicles[0]
        if action == 0 and ego_vehicle.lane > 0:
            ego_vehicle.lane -= 1
        elif action == 2 and ego_vehicle.lane < 6:
            ego_vehicle.lane += 1

        for vehicle in self.vehicles:
            vehicle.move()
            if vehicle.position[1] > self.highway.height:
                vehicle.position[1] = 0
                vehicle.lane = random.randint(0, 6)
                vehicle.speed = random.randint(65, 80)
