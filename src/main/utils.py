import os
import csv
import torch

import numpy as np
import pandas as pd
import seaborn as sns

from plot import *
from os.path import join
from pathlib import Path
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader, Subset
from customLoader import *
from torchvision.transforms import transforms
from sklearn.metrics import pairwise_distances_argmin_min


from IPython import embed


def get_loader(trajectories, transform, conf, shuffle=False, limit=None):
    train, _ = get_train_val_split(trajectories, 1)
    train_dataset = CustomHabitatData(train, transform=transform, delay=False, **conf)

    if not limit == None:
        train_dataset = Subset(train_dataset, list(range(limit)))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=0)
    return train_dataloader

def compute_kmeans(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)


def compute_embeddings_curl(loader, encode):
    #embed()
    print("Computing embeddings")
    return np.array([encode(data[:,0].cuda()).detach().cpu().numpy() for data in loader]).squeeze()


def compute_embeddings(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return np.array([model.compute_embedding(batch, device).detach().cpu().numpy() for batch in loader]).squeeze()



def get_images(loader):
    return torch.cat([data[:,0] for data in loader])

"""
def load_trajectories(trajectories):
    print("Loading trajectories...")

    all_trajectories = []
    files = sorted([x for x in os.listdir(f"./results/{trajectories}/") if 'coords' in x], key=lambda x: int(x.split('.')[1]))
    for file in files:
        with open(f"./results/{trajectories}/{file}") as csv_file:
            trajectory = []
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for i, row in enumerate(csv_reader):
                trajectory.append(row)
            all_trajectories.append(trajectory)
    return np.array(all_trajectories).reshape(-1, 3)
"""
def load_trajectories(trajectories):
    print("Loading trajectories...")

    all_trajectories = []
    files = sorted([x for x in os.listdir(f"../results/"+trajectories+"_positions/")],  key=lambda x: int(x.split('.')[0].split('_')[2]))
    for file in files:
        all_trajectories.append(np.load("../results/"+trajectories+"_positions/"+file))
    #IF EXPERTS:
    #all_trajectories = [y for x in all_trajectories for y in x]
    f = np.array(all_trajectories, dtype=object).reshape(-1,3)
    return f



def store_goal_positions(enc):
    loader = get_loader(
        enc.trajectories,
        enc.transform,
        enc.conf,
        shuffle=enc.shuffle,
        limit=enc.limit)
    trajectories = load_trajectories(enc.trajectories[0])
    embeddings = compute_embeddings(loader, enc.encode)
    values = pd.DataFrame(columns=['x', 'y', 'Code:'])


    """
    for i, (e, p) in enumerate(zip(embeddings, trajectories)):
        x = float(p[2])
        y = float(p[0])
        e = torch.from_numpy(e).cuda()
        k = enc.compute_argmax(e.unsqueeze(dim=0))
        values = values.append({'x': x, 'y': y, 'Code:': int(k)}, ignore_index=True)
    #embed()
    values['Code:'] = values['Code:'].astype('int32')
    
    
    means = values.groupby('Code:').mean()
    means.to_csv('skill_mean_position.csv')

    """

    goals_path = sorted(os.listdir("goal_states/white_DEF/"))
    goals = []
    i = 0
    print("Computing closest embeddings")
    for goal in goals_path:
        g = np.load("goal_states/white_DEF/"+goal)
       
        closest = enc.compute_argmax(torch.from_numpy(g).squeeze().float().cuda(), embeddings)
        #closest, _ = pairwise_distances_argmin_min(g.reshape(1,-1), embeddings)
        print(closest)
        with open(f"goal_states/white_positions"+str(i)+".npy", 'wb') as f:
            np.save(f, trajectories[closest])
        i = i+1
    

def construct_map(enc, model):
    if not enc.limit == None:
        limit = [x*50 for x in range(enc.limit)]
    else: 
        limit = None

    loader = get_loader(
        enc.trajectories,
        enc.transform,
        enc.conf,
        shuffle=enc.shuffle,
        limit=enc.limit)
    trajectories = load_trajectories(enc.trajectories[0])
    
    if model == "vqvae":
        embeddings = compute_embeddings(loader, enc.model)
    elif model == "curl":
        embeddings = compute_embeddings_curl(loader, enc.encode)

    print(trajectories.shape)
    print(embeddings.shape)
    if enc.type == "index":
        index_map(trajectories, embeddings, enc, model)
    elif enc.type == "reward":
        reward_map(trajectories, embeddings, enc, model)
    elif enc.type == "embed":
        images = get_images(loader) + 0.5
        embed_map(embeddings, images, enc.experiment)
    else:
        raise NotImplementedError()

def index_map(trajectories, embeddings, enc, model):

    print("Get index from all data points...")
    values = pd.DataFrame(columns=['x', 'y', 'Code:'])
    for i, (e, p) in enumerate(zip(embeddings, trajectories)):
        x = float(p[2])
        y = float(p[0])
        e = torch.from_numpy(e).cuda()
        k = enc.compute_argmax(e.unsqueeze(dim=0))
        values = values.append({'x': x, 'y': y, 'Code:': int(k)}, ignore_index=True)
    values['Code:'] = values['Code:'].astype('int32')
    palette = sns.color_palette("Paired", n_colors=len(list(values['Code:'].unique())))
    plot_idx_maps(values, palette, "brief", model)


def reward_map(trajectories, embeddings, enc, model):
    print("Get index from all data points...")
    data_list = []
    for g in range(enc.num_clusters):
        print(f"Comparing data points with goal state {g}", end="\r")
        values = pd.DataFrame(columns=['x', 'y', 'reward'])
        for i, (e, p) in enumerate(zip(embeddings, trajectories)):
            x = float(p[2])
            y = float(p[0])
            e = torch.from_numpy(e).cuda()

            coord = None
            if not enc.conf["data_type"] == "pixel":
                coord = np.array(p, dtype=np.float32)
                mu = loader.dataset.coord_mean
                std = loader.dataset.coord_std
                coord = (coord-mu)/std


            r = enc.compute_reward(e.unsqueeze(dim=0), g, coord)


            values = values.append({'x': x, 'y': y, 'reward': r}, ignore_index=True)


        data_list.append(values)
    
    #experiment = enc.test['path_weights'].split('/')[0]
    
    plot_reward_maps(data_list, "vqvae")


"""
##### CURL SPARSE 
def reward_map(trajectories, embeddings, enc, model):
    print("Get index from all data points...REWARD")
    data_list = []
    for g in range(enc.num_clusters):
        print(f"Comparing data points with goal state {g}", end="\r")
        values = pd.DataFrame(columns=['x', 'y', 'reward'])
        for i, (e, p) in enumerate(zip(embeddings, trajectories)):
            x = float(p[2])
            y = float(p[0])
            e = torch.from_numpy(e).cuda()
            logits = enc.compute_logits(e.unsqueeze(dim=0))
            r = 0
            if k == g:
                r = 1                  
                #embed()
                max = logits[0][k].detach().cpu().item()
                #logits[0][k] = 0
                #k2 = torch.argmax(logits).cpu().item()
                #sec_max = logits[0][k2].detach().cpu().item()
                #r = (max-sec_max)/max
                
            values = values.append({'x': x, 'y': y, 'reward': r}, ignore_index=True)
        data_list.append(values)

    plot_reward_maps(data_list, model)
"""

def embed_map(embeddings, images, exp):
    import tensorflow
    from torch.utils.tensorboard import SummaryWriter
    import tensorboard

    tensorflow.io.gfile = tensorboard.compat.tensorflow_stub.io.gfile
    writer = SummaryWriter(log_dir=os.path.join("./results", exp))
    writer.add_embedding(embeddings, label_img=images)
    writer.close()

def trainValSplit(traj_list, split):
    num_traj = len(traj_list)
    if split == 1:
        return traj_list, []
    else:
        # Since we can mix trajectories from different tasdks, we want to shuffle them
        # e.g: otherwise we could have all treechop trajectories as validation
        shuffle(traj_list)
        return traj_list[:int(split*num_traj)], traj_list[int(split*num_traj):]

def get_train_val_split(t, split):
    path = Path('../results')
    total_t = []
    #items = sorted(os.listdir(path / t[0]))
    items = sorted(os.listdir(path / t[0]), key=lambda x: int(x.split('.')[0].split('_')[1]))
    items = [path / t[0] / x for x in items]
    total_t.extend(items)
    return trainValSplit(total_t, split)
