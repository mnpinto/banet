from fastai.vision import *
from fastai.callbacks import *

__all__ = ['SampleEpisode', 'ImageSequence']


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
        """Random samples are n-way k-shot"""
        idx = []
        condition = ((self.df.name == self.df.loc[k, 'name']) &
            (self.df.time == self.df.loc[k, 'time'] + pd.Timedelta(days=self.sequence_len)) &
            (self.df.r == self.df.loc[k, 'r']) &
            (self.df.c == self.df.loc[k, 'c']))
        where = self.df.loc[condition].index.values
        if len(where) == 0:
            idx = None
        else:
            times = pd.date_range(self.df.loc[k-self.sequence_len//2, 'time'], periods=2*self.sequence_len, freq='D')
            condition = ((self.df.name == self.df.loc[k, 'name']) &
                (self.df.time.isin(times)) &
                (self.df.r == self.df.loc[k, 'r']) &
                (self.df.c == self.df.loc[k, 'c']))
            where = self.df.loc[condition].sort_values(by='time').index.values
            idx = where[:self.sequence_len]
            if len(idx) != self.sequence_len: idx = None
        return idx


class ImageSequence(LearnerCallback):
    def __init__(self, learn, sequence_len=64, n_sequences=1):
        super().__init__(learn)
        self.sequence_len = sequence_len
        self.n_sequences = n_sequences
        
    def on_batch_begin(self, last_input, last_target, epoch, iteration, **kwargs):
        bs, ch, sz1, sz2 = last_input.size()
        last_input = last_input.view(self.sequence_len, self.n_sequences, ch, sz1, sz2).permute(1, 2, 0, 3, 4)
        last_target = last_target.view(self.sequence_len, self.n_sequences, 1, sz1, sz2).permute(1, 2, 0, 3, 4)#.max(2)[0]
        return {'last_input': last_input, 'last_target': last_target}