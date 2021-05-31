import os
import cv2
import torch

import numpy as np
import matplotlib.pylab as plt

from random import choice, randint, shuffle
from os.path import join
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

from IPython import embed


class CustomHabitatData(Dataset):
    def __init__(self, traj_list, transform=None, path='../results', delay=False, **kwargs) -> None:
        self.path = Path(path)
        self.traj_list = traj_list
        self.delay = delay
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.transform = transform
        self.k_std = kwargs['k_std']
        self.k_mean = kwargs['k_mean']
        try:
            self.data_type = kwargs['data_type']
        except:
            print("curl")

        self.loadData()

    def getTrajLastIdx(self, idx):
        list_idxs = self.list_idxs
        idx_acc = 0
        for i in list_idxs:
            if idx >= idx_acc and idx < (idx_acc + i):
                return idx_acc + i - 1
            idx_acc += i
        return None

    def load_trajectories(self):
        print("Loading trajectories...")

        trajectories = self.traj_list

        #print(len(trajectories))

        all_trajectories = []
        #files = sorted([x for x in os.listdir(f"../results/trajectories1_positions/")],  key=lambda x: int(x.split('.')[0].split('_')[2]))
        for file in trajectories:
            #embed()
            #../results/trajectories1_positions/trajectory_0.npy
            all_trajectories.append(np.load("../results/" + str(file).split("/")[2] + "_positions/" + "trajectory_positions_" +  str(file).split("/")[3].split(".")[0].split("_")[1] + ".npy" ))
        #IF EXPERTS:
        #all_trajectories = [y for x in all_trajectories for y in x]
        coords = np.array(all_trajectories, dtype=np.float32).reshape(-1,3)
        print(coords.shape)
        self.coord_mean = coords.mean(axis=0)
        self.coord_std = coords.std(axis=0)

        self.coords = (coords - self.coord_mean)/self.coord_std


    def getCoords(self, idx, key_idx):
        return torch.from_numpy(self.coords[idx]), torch.from_numpy(self.coords[key_idx])

    def getImages(self, idx, key_idx):
        
        query = torch.tensor(self.data[idx])
        key = torch.tensor(self.data[key_idx])
        

        #if self.transform is not None:
            #DEPTH
            #key = self.transform(key)
            #query = self.transform(query)
            #key = torch.cat((self.transform(key[:,:,0:3]), self.transform(key[:,:,3:6])))
            #query = torch.cat((self.transform(query[:,:,0:3]), self.transform(query[:,:,3:6])))
        return key, query


    def customLoad(self):
        data, list_idxs = [], []


        for i, traj in enumerate(self.traj_list):
            print(f"\tTraj: {i}", end ='\r')
            obs = np.load(traj, allow_pickle=True)
            data.append(obs)
            list_idxs.append(obs.shape[0])
        #print()
        #IF EXPERTS:
        #data = [y for x in data for y in x]
        #4 IF WE CONSIDER DEPTH
        data = np.array(data).reshape(-1, 256, 256, 1)
        self.data = data
        self.list_idxs = list_idxs

    def expertLoad(self):
        data, list_idxs = [], []

        for i, vid in enumerate(self.traj_list):
            print(f"\tVid: {i}", end ='\r')
            video = []

            vid_path = vid / 'recording.mp4'
            frames = cv2.VideoCapture(str(vid_path))
            ret = True
            fc = 0
            while(frames.isOpened() and ret):
                ret, frame = frames.read()
                if ret and fc % 3 == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video.append(frame)
                fc += 1

            data.append(video)
            list_idxs.append(len(video))
        data = [y for x in data for y in x]
        self.data = np.array(data)
        self.list_idxs = list_idxs

    def loadData(self) -> list:
        print('Loading data...')
        if self.data_type == "pixel":
            print("PIXEL")
            self.customLoad()
        elif self.data_type == "coord":
            self.load_trajectories()
        elif self.data_type == "pixelcoord":
            self.customLoad()
            self.load_trajectories()


    def __len__(self) -> int:
        return len(self.data)

    """
    def __getitem__(self, index):
        # Get query obs
        query = self.data[index]
        if self.delay:
            # Make sure that we pick a frame from the same trajectory
            fin_idx = self.getTrajLastIdx(index)
            key_idx = index + int(np.random.rand()*self.k_std + self.k_mean)
            # Get key obs
            key = self.data[min(key_idx, fin_idx)]
        else:
            key = self.data[index]


        if self.transform is not None:
            key = self.transform(key)
            query = self.transform(query)

        # Stack query and key to return [2,3,64,64]
        return torch.stack((query, key))
    """


    def __getitem__(self, index):
        # Get query obs
        if self.delay:
            # Make sure that we pick a frame from the same trajectory
            fin_idx = self.getTrajLastIdx(index)
            key_idx = index + int(np.random.rand()*self.k_std + self.k_mean)

            # Get key obs
            key_idx = min(key_idx, fin_idx)
        else:
            key_idx = index

        try:

            if self.data_type == "pixel":
                key, query = self.getImages(index, key_idx)
                return torch.stack((key, query))

            elif self.data_type == "coord":
                coord_query, coord_key = self.getCoords(index, key_idx)
                return torch.stack((coord_query, coord_key))

            elif self.data_type == "pixelcoord":
                key, query = self.getImages(index, key_idx)
                coord_query, coord_key = self.getCoords(index, key_idx)
                return torch.stack((query, key)), torch.stack((coord_query, coord_key))

        except:
            #embed()
            key, query = self.getImages(index, key_idx)
            return torch.stack((key, query))



    """
    def __getitem__(self, index):
        # Get query obs
        query = self.data[index]
        #query = self.data[0]
        if self.delay:
            # Make sure that we pick a frame from the same trajectory
            fin_idx = self.getTrajLastIdx(index)
            key_idx = index + int(np.random.rand()*self.k_std + self.k_mean)

            # Get key obs
            key_idx = min(key_idx, fin_idx)
        else:
            key_idx = index

        key = self.data[key_idx]
        #key = self.data[0]

        if self.transform is not None:
            key = self.transform(key)
            query = self.transform(query)

        if self.coords is not None:
            coord_key, coord_query = self.getCoords(index, key_idx)

            coord_query = torch.from_numpy(coord_query)
            coord_key = torch.from_numpy(coord_key)

            # Stack query and key to return [2,3,64,64]
            return torch.stack((query, key)), torch.stack((coord_query, coord_key))
        else:
            return torch.stack((query, key))
    """

class LatentDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_name, transform=None):
        print('Loading latent block data')
        p = Path('../data')
        self.fname= file_name
        file_name = self.fname + '_t.npy'
        self.data_t = np.load(p / 'latent_blocks' / file_name, allow_pickle=True)
        file_name = self.fname + '_b.npy'
        self.data_b = np.load(p / 'latent_blocks' / file_name, allow_pickle=True)
        self.transform = transform

    def __getitem__(self, index):
        top = self.data_t[index]
        bottom = self.data_b[index]
        if self.transform is not None:
            top = self.transform(top)
            bottom = self.transform(bottom)
        label = 0
        return top, bottom, label

    def __len__(self):
        return len(self.data_t)

class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_path, train=True, shape=16, transform=None):
        print('Loading latent block data')
        self.data = np.load(file_path, allow_pickle=True)
        self.transform = transform
        self.shape = shape

    def __getitem__(self, index):
        img = self.data[index]
        img = img.reshape((self.shape, self.shape))
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transform = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                            ])
    # env_list = ['MineRLNavigate-v0', 'MineRLNavigateVectorObf-v0']
    # mrl = MultiMinecraftData(env_list, 'train', 1, False, transform=transform)
    # embed()
    # img = mrl[10]
    # plt.imshow(img)
    # plt.show()
    # c = CustomMinecraftData('CustomTrajectories4', 'train', 0.95, transform=transform)
    # embed()
