import habitat
import habitat_sim
import numpy as np
import random
from IPython import embed
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import matplotlib.pyplot as plt
from habitat_sim.utils.data import ImageExtractor

#map1
#test_scene = "../data/scene_datasets/mp3d/8WUmhLawc2A/8WUmhLawc2A.glb"

#map2
#test_scene = "../data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb"
#folder = "trajectories_map2"

#map3
test_scene = "../data/scene_datasets/mp3d/29hnd4uzFmX/29hnd4uzFmX.glb"
folder = "trajectories_circular"

#white
#test_scene = "../data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
#folder = "trajectories_full"

rgb_sensor = True # @param {type:"boolean"}
depth_sensor = False # @param {type:"boolean"}
semantic_sensor = False  # @param {type:"boolean"}


sim_settings = {
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 100,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]
    # Note: all sensors must have the same resolution
    sensor_specs = []
    color_sensor_spec = habitat_sim.SensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)
    depth_sensor_spec = habitat_sim.SensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)
    semantic_sensor_spec = habitat_sim.SensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.postition = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)
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
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.5)
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


random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])


max_frames = 150
episodes = 500

"""
extractor = ImageExtractor(test_scene, output=['semantic'])
instance_id_to_name = extractor.instance_id_to_name
map_to_class_labels = np.vectorize(lambda x: labels.get(instance_id_to_name.get(x, 0), 0))

"""
for episode in range(episodes):
    # Set agent states
    
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)

    # Get agent state
    agent_state = agent.get_state()

    total_frames = 0
    action_names = list(
        cfg.agents[
            sim_settings["default_agent"]
        ].action_space.keys()
    )

    trajectory = []
    agent_positions = []
    while total_frames < max_frames:
        action = random.choice(action_names)
        observations = sim.step(action)
        rgb = observations["color_sensor"][:,:,0:3]
        
        
        #semantic = observations["semantic_sensor"]
        #semantic_img = Image.new("P", (semantic.shape[1], semantic.shape[0]))
        #semantic_img.putpalette(d3_40_colors_rgb.flatten())
        #semantic_img.putdata((semantic.flatten() % 40).astype(np.uint8))
        #semantic_img = semantic_img.convert("RGB")
        #semantic = np.array(semantic_img)

        #depth = observations["depth_sensor"]

        #full = np.dstack((rgb, depth, semantic))

        trajectory.append(rgb)
        agent_positions.append(agent.get_state().position)
        total_frames += 1

    agent_positions = np.array(agent_positions)
    trajectory = np.array(trajectory)
    
    
    #map_to_class_labels(extractor[episode]["semantic"])
    with open(f'../results/{folder}/trajectory_{episode}.npy', 'wb') as f:
        np.save(f, trajectory)

    with open(f'../results/{folder}_positions/trajectory_positions_{episode}.npy', 'wb') as fp:
        np.save(fp,agent_positions)
    print("Saved episode: " + str({episode}))
