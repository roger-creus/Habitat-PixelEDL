



import habitat
from habitat.utils.visualizations import maps
import habitat_sim
import numpy as np
import random
import imageio
import os
from os import listdir
import magnum as mn
import math

import matplotlib.pyplot as plt


folder = "trajectories_map2"
test_scene = "../data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb"


#test_scene = "../data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
#folder = "trajectories_experts"

#test_scene = "../data/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
#folder = "trajectories_skok"


sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": False,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        }
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown



skill_positions = []
#for g in sorted(os.listdir("goal_states/white_positions/")):
#    skill_positions.append(np.array(np.load("goal_states/white_positions/" + g, allow_pickle=True),dtype="float32"))


skill_positions.append(np.array(np.load("goal_states/white_positions/white_positions4.npy", allow_pickle=True),dtype="float32"))
#skill_positions.append([-3.5, 0.7, -1.2])


height = abs(sim.pathfinder.get_bounds()[0][1])
meters_per_pixel = 0.02

xy_vis_points = convert_points_to_topdown(sim.pathfinder, skill_positions, meters_per_pixel)



######## TOP DOWN MAP #####

#height = abs(sim.pathfinder.get_bounds()[0][1])
#meters_per_pixel = 0.2

#####GET MAP
top_down_map = maps.get_topdown_map(sim.pathfinder, height, meters_per_pixel=meters_per_pixel)
recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
top_down_map = recolor_map[top_down_map]

grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])



fig= plt.figure()
plt.imshow(top_down_map)
for point in xy_vis_points:
    plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)

fig.savefig("top_down_map_skills.png")
plt.close()

#####GET TRAJECTORY POSITIONS
#onlyfiles = [f for f in listdir("../results/"+folder+"_positions/")]


#for t in onlyfiles:
    #traj = list(np.load("../results/" + folder + "_positions/" + t))
    #trajectory = [maps.to_grid(path_point[2],path_point[0],grid_dimensions,pathfinder=sim.pathfinder,) for path_point in traj]
   # grid_tangent = mn.Vector2(trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0])
   # path_initial_tangent = grid_tangent / grid_tangent.length()
   # initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
    # draw the agent and trajectory on the map
    #maps.draw_path(top_down_map, trajectory)
    #maps.draw_agent(top_down_map, trajectory[0], initial_angle, agent_radius_px=8)



#map_filename = os.path.join("../results/", "top_down_map.png")
#imageio.imsave(map_filename, top_down_map)
