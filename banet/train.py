__all__ = ['SampleEpisode', 'ImageSequence', 'get_y_fn', 'open_mat', 'open_mask', 'set_info_df',
           'BCE', 'accuracy', 'dice2d', 'mae', 'train_model','CutoutCombined', 'CustomLoader']

# Cell
import numpy as np
import pandas as pd
import scipy.io as sio

from fastai.vision.all import *
from fastai.data.transforms import Pipeline
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Sampler
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path
from fastai.data.all import *

from .models import BA_Net


# Cell
class SampleEpisode(Sampler):
    def __init__(self, data_source, n_episodes, sequence_len, n_sequences, info_df, nburned=100):
        self.ds, self.epoch_size = data_source, n_episodes
        self.sequence_len, self.n_sequences = sequence_len, n_sequences
        self._epochs = []
        self.df = info_df
        self.nburned = nburned

    def __len__(self):
        return self.epoch_size*self.sequence_len*self.n_sequences

    def __iter__(self): return iter(self.get_epoch())

    def get_epoch(self):
        """Get indices for one epoch of size epoch_size"""
        idx = []
        for n in range(self.epoch_size):
            idx = [*idx, *self.get_batch()]
        return idx

    def get_batch(self):
        """Get indices for one mini-batch"""
        idx = []
        n = 0
        while n < self.n_sequences:
            k = np.random.choice(self.df.loc[self.df.ba>self.nburned].index, size=1, replace=False)[0]
            s = self.random_sample(k)
            if s is not None:
                idx = [*idx, *s]
                n += 1
        return idx

    def random_sample(self, k):
        """
        Build a sequence of sequence_len days centered on sample k.
        Returns indices if complete sequence exists, None otherwise.
        """
        # Create date range centered on k: [k-32, k-31, ..., k, ..., k+31]
        center_time = self.df.loc[k, 'time']
        times = pd.date_range(
            center_time - pd.Timedelta(days=self.sequence_len // 2),
            periods=self.sequence_len,
            freq='D'
        )

        # Find all images in this time window for same spatial location
        condition = (
            (self.df.name == self.df.loc[k, 'name']) &
            (self.df.time.isin(times)) &
            (self.df.r == self.df.loc[k, 'r']) &
            (self.df.c == self.df.loc[k, 'c'])
        )
        where = self.df.loc[condition].sort_values(by='time').index.values

        # Only return if we have exactly sequence_len consecutive days
        return where if len(where) == self.sequence_len else None

class CustomLoader(DataLoader):
    def __init__(self, dataset, sampler, **kwargs):
        self.custom_sampler = sampler
        super().__init__(dataset, **kwargs)

    def get_idxs(self):
        return list(self.custom_sampler)

class ImageSequence(Callback):
    def __init__(self, sequence_len=64, n_sequences=1):
        self.sequence_len = sequence_len
        self.n_sequences = n_sequences

    def before_batch(self):
        x, y = self.learn.xb[0], self.learn.yb[0]
#        print(f"Before reshape: x={x.shape}, y={y.shape}, dtype={x.dtype}, mu_dtype={self.mean.dtype}")  # Debug
        bs, ch, sz1, sz2 = x.shape
        x = x.view(self.sequence_len, self.n_sequences, ch, sz1, sz2).permute(1, 2, 0, 3, 4)
        y = y.view(self.sequence_len, self.n_sequences, 1, sz1, sz2).permute(1, 2, 0, 3, 4)
#        print(f"After reshape: x={x.shape}, y={y.shape}")  # Debug
        self.learn.xb = (x,)
        self.learn.yb = (y,)

# Cell
def get_y_fn(file, satellite='VIIRS750', target_product='MCD64A1C6'):
    f = str(Path(str(file))).replace('images', 'masks')
    f = f.replace(satellite, target_product)
    return f

def open_mat(fn, *args, **kwargs):
    data = sio.loadmat(fn)
    data = np.array([data[r] for r in ['Red', 'NIR', 'MIR', 'FRP']])
    data[np.isnan(data)] = 0
    return torch.from_numpy(data).float()

def open_mask(fn, *args, **kwargs):
    data = sio.loadmat(fn)['bafrac']
    data[np.isnan(data)] = 0
    data = torch.from_numpy(data).float()
    return data.view(1, data.shape[0], data.shape[1])

def set_info_df(items_list, satellite='VIIRS750', target_product='MCD64A1C6'):
    names, dates = [], []
    rs, cs = [], []
    for o in items_list:
        name, date, r,  c = Path(o).stem.split('_')
        date = pd.Timestamp(date)
        names.append(name)
        dates.append(date)
        rs.append(r)
        cs.append(c)
    ba = [open_mask(get_y_fn(str(o), satellite=satellite, target_product=target_product)
                   ).data.sum().item() for o in progress_bar(items_list)]
    return pd.DataFrame({'name': names, 'time': dates, 'r':rs, 'c':cs, 'ba':ba})

def _cutout(x, n_holes=1, length=40):
    "Cut out `n_holes` number of square holes of size `length` in image at random locations."
    h,w = x.shape[1:]
    for n in range(n_holes):
        h_y = np.random.randint(0, h)
        h_x = np.random.randint(0, w)
        y1 = int(np.clip(h_y - length / 2, 0, h))
        y2 = int(np.clip(h_y + length / 2, 0, h))
        x1 = int(np.clip(h_x - length / 2, 0, w))
        x2 = int(np.clip(h_x + length / 2, 0, w))
        #x[:2, y1:y2, x1:x2] = 1
        x[-1, y1:y2, x1:x2] = 0
    return x
  
def _cutout2(x, n_holes=1, length=40):
    "Create `n_holes` number of square batches of size `length` and random value in image at random locations."
    h,w = x.shape[1:]
    h_y = np.random.randint(0, h)
    h_x = np.random.randint(0, w)
    y1 = int(np.clip(h_y - length / 2, 0, h))
    y2 = int(np.clip(h_y + length / 2, 0, h))
    x1 = int(np.clip(h_x - length / 2, 0, w))
    x2 = int(np.clip(h_x + length / 2, 0, w))
    x[0, y1:y2, x1:x2] = torch.rand(1)
    x[1, y1:y2, x1:x2] = torch.rand(1)
    x[2, y1:y2, x1:x2] = torch.rand(1)
    return x

class CutoutCombined(RandTransform):
    """Applies both cutout types: random patches on Red/NIR/MIR and erasing on FRP."""
    order = 20
    def __init__(self, n_holes=(1, 5), length=(5, 50), **kwargs):
        super().__init__(**kwargs)
        self.n_holes = n_holes
        self.length = length

    def encodes(self, x:TensorImage):
        n_holes = random.randint(*self.n_holes) if isinstance(self.n_holes, tuple) else self.n_holes
        length = random.randint(*self.length) if isinstance(self.length, tuple) else self.length

        # Apply cutout2 first (random values on Red, NIR, MIR)
        x = _cutout2(x, n_holes=n_holes, length=length)

        # Then apply cutout (erase FRP channel)
        x = _cutout(x, n_holes=n_holes, length=length)
        return x

class BCE(Module):
    "Binary Cross Entropy loss."
    def forward(self, x, y):
        bce = nn.BCEWithLogitsLoss()
        return 100*bce(x.view(x.size()[0],-1),y.view(y.size()[0], -1))

def accuracy(input:Tensor, targs:Tensor, thr:int=0.5)->Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    input = (input.sigmoid()>thr).long()
    targs = (targs>thr).long()
    return (input==targs).float().mean()

def dice2d(pred, targs, thr=0.5):
    pred = pred.squeeze()
    targs = targs.squeeze().sum(0)
    pred = (pred.sigmoid().sum(0)>thr).float()
    targs = (targs>thr).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def mae(pred, targs, thr=0.5):
    a = pred.squeeze().sigmoid().sum(0)>thr
    pred = pred.squeeze().max(0)[1]
    targs = targs.squeeze().max(0)[1]
    pred = pred[a]
    targs = targs[a]
    return (pred-targs).abs().float().mean()

def train_model(val_year, r_fold, path, model_path, n_epochs=8, lr=1e-2, nburned=10, n_episodes_train=2000,
        n_episodes_valid=100, sequence_len=64, n_sequences=1, do_cutout=True, model_arch=None,
        pretrained_weights=None, satellite='VIIRS750', target_product='MCD64A1C6',
        get_learner=False, save_to=None):
    path_img = path/'images'
    train_files = sorted([f.name for f in path_img.iterdir()])
    times = pd.DatetimeIndex([pd.Timestamp(t.split('_')[1]) for t in train_files])

    train_df = pd.DataFrame({'times': times, 'ID': train_files})

    valid_idx = train_df.loc[train_df.times.dt.year == val_year].index.values

    dblock = DataBlock(
        blocks=(TransformBlock, TransformBlock),
        get_x= lambda x: TensorImage(open_mat(x)),
        get_y= lambda x: TensorBase(open_mask(get_y_fn(str(x), satellite, target_product))),
        splitter=IndexSplitter(valid_idx)
    )

    dsets = dblock.datasets(path_img/train_files)

    info_train = set_info_df(dsets.train.items, satellite=satellite, target_product=target_product)
    info_valid = set_info_df(dsets.valid.items, satellite=satellite, target_product=target_product)

    bs = sequence_len * n_sequences

    train_dl = CustomLoader(dsets.train,
                        sampler=SampleEpisode(dsets.train[0], n_episodes=n_episodes_train,
                                              sequence_len=sequence_len, n_sequences=n_sequences,
                                              info_df=info_train, nburned=nburned), bs=bs)

    valid_dl = CustomLoader(dsets.valid,
                        sampler=SampleEpisode(dsets.valid[0], n_episodes=n_episodes_valid,
                                              sequence_len=sequence_len, n_sequences=n_sequences,
                                              info_df=info_valid, nburned=nburned), bs=bs)

    dls = DataLoaders(train_dl, valid_dl)
    mean = tensor([0.2349, 0.3548, 0.1128, 0.0016])
    std  = tensor([0.1879, 0.1660, 0.0547, 0.0776])

    if do_cutout:
        dls.train.after_batch = Pipeline([
            CutoutCombined(n_holes=(1, 5), length=(5, 50), p=0.5),
            Brightness(max_lighting=0.2, p=0.75),  # Paper: 0.4-0.6 range
            Contrast(max_lighting=0.2, p=0.75),    # Paper: 0.8-1.25 range
            Normalize.from_stats(mean, std)
            ])
    else:
        dls.train.after_batch = Normalize.from_stats(mean, std)

    dls.valid.after_batch = Normalize.from_stats(mean, std)

    model = BA_Net(4, 1, sequence_len) if model_arch is None else model_arch(4, 1, sequence_len)

    if pretrained_weights is not None:
        print(f'Loading pretrained_weights from {pretrained_weights}\n')
        map_loc = None if torch.cuda.is_available() else torch.device('cpu')
        weights = torch.load(pretrained_weights, map_location=map_loc, weights_only=False)
        state_dict = weights['model'] if isinstance(weights, dict) and 'model' in weights else weights
        model.load_state_dict(state_dict)

    learn = Learner(dls, model, cbs=[
        ImageSequence(sequence_len=sequence_len, n_sequences=n_sequences),
        GradientClip(1.0)
        ],
        loss_func=BCE(), wd=1e-2, metrics=[accuracy, dice2d, mae])

    if get_learner: return learn

    print('Starting traning loop\n')
    learn.fit_one_cycle(n_epochs, lr*64/sequence_len)

    model_path.mkdir(exist_ok=True)
    if save_to is None:
        save_to='banet-val{val_year}-fold{r_fold}-v2.pth'
    torch.save(learn.model.state_dict(), model_path/save_to)
    print(f'Completed! {save_to} saved to {model_path}.')
