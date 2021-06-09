# PixelEDL: Unsupervised Skill Discovery from Pixels
#### This project is the Bachelor's Thesis of the Bachelor's Degree in Data Science and Engineering at la Universitat Politècnica de Catalunya (UPC).
Find the full work report in the main directory "./report.pdf"

## Useful links:

https://imatge-upc.github.io/PixelEDL/

https://imatge-upc.github.io/PiCoEDL/

## Installation

1) Create conda environment for habitat-lab and habitat-sim

```
$ conda create -n habitat python=3.6 cmake=3.14.0
$ conda activate habitat
```

2) Clone this repository and install requirements (includes habitat-lab and habitat-baselines):
```
$ git clone https://github.com/yuyecreus/Habitat-PixelEDL.git
$ cd Habitat-PixelEDL
$ pip install -r requirements.txt
```

3) Install habitat-sim. This work used the habitat-sim headless installation: 
```
$ conda install habitat-sim headless -c conda-forge -c aihabitat
```
But if your machine has a display you can install with:
```
$  conda install habitat-sim -c conda-forge -c aihabitat
```

4) Ask for access to the Matterport3D dataset https://niessner.github.io/Matterport/ and install the Matterport-habitat dependencies:
```
$ python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

5) Download the test scenes as indicated in https://github.com/facebookresearch/habitat-lab#installation and extract data folder in zip to Habitat-PixelEDL/data/ where Habitat-PixelEDL/ is the github repository folder

6) Run python examples/example.py to test performance

For more help visit https://github.com/facebookresearch/habitat-lab and https://github.com/facebookresearch/habitat-sim


## Experiments

#### Exploration
1) Create empty folders in Habitat-PixelEDL/results/{folder_name} for the exploration dataset. 
2) Edit script in Habitat-PixelEDL/src/one_scene_exploration.py to set folder_name variable to the chosen {folder_name}. One can set the scene to explore, the number of trajectories, the number of observations in each trajectory and the sensors. 
3) Run Habitat-PixelEDL/src/one_scene_exploration.py

The trajectories can be plotted running Habitat-PixelEDL/src/generate_scene_map.py and setting folder_name variable to {folder_name}

#### Skill Discovery
Configuration files for ATC and VQ-VAE are found in Habitat-PixelEDL/config/curl.yml and Habitat-PixelEDL/config/vqvae.yml respectively.

##### Training
In these config files:

1) Set trajectories: {folder_name} (containing the exploration trajectories)
2) Set data_type to "pixel", "coord" or "pixelcoord" (ATC only supports "pixel")
3) Run training from the Habitat-PixelEDL/src/ directory:
```
python -u -m main.curl_train curl
```
```
python -u -m main.vqvae_train vqvae
```


#### Testing
Only for ATC (curl): Store goal states (K-means clustering of the embedding space) by commenitng construct_map() and running only store_goal_states() in Habitat-PixelEDL/src/main/curl_test.py
Then run:
```
python -u -m main.curl_test curl
```

1) Show maps (index, reward or embed) by setting test/type to "pixel", "reward" or "embed" in the models' config files. Then run:
```
python -u -m main.vqvae_test vqvae
```

For ATC first comment store_goal_states() and uncomment construct_map() in Habitat-PixelEDL/src/main/curl_test.py and then run 
```
python -u -m main.curl_test curl
```

Maps are saved in Habitat-PixelEDL/results/ by default


#### Image goal-driven Navigation

1) Change the variable DATA_PATH in habitat/config/default.py to specify the scene where to perform image goal-driven navigation
2) Configure config/curl_RL.yml to specify the path of the pre-trained weights of the ATC encoder
3) Configure habitat_baselines/config/imagenav/ppo_imagenav_example.py for tuning the PPO model (set path for model checkpoints, total number of steps, directory for saving evaluation videos...)
4) Run the training of the PPO model from the main directory:
```
python -u habitat_baselines/run.py --exp-config habitat_baselines/config/imagenav/ppo_imagenav_example.yaml --run-type train
```
5) After trainining, run the evaluation of the model with: 
```
python -u habitat_baselines/run.py --exp-config habitat_baselines/config/imagenav/ppo_imagenav_example.yaml --run-type eval
```

Feel free to open issues for asking for any feature or reporting a bug!

