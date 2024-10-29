# utils.py

def reset_simulation(simulation):
    for vehicle in simulation.vehicles:
        vehicle.position = [0, random.randint(0, simulation.highway.height)]
        vehicle.speed = random.randint(65, 80)
        vehicle.lane = random.randint(0, 6)
    return simulation
