import os
import glob
import torch
import numpy as np
import pandas as pd
from random import shuffle
from os.path import join, dirname

from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    """
    Wrapper class for the Spain weather dataset
    """
    def __init__(self, n_samples, n_nodes, n_timesteps, features, filename=None, dataset_path=None, force_new=False, discard=False, from_partial=False, normalize=False, normalize_params=None, dset=None, existing_config=None, existing_indices=None, threshold=1):
        """
        Generates the dataset with the given parameters, unless a similar dataset has been generated
            before, in which case it is by default loaded from the file.
        Args:
            n_samples(int): Number of simulations to fetch
            n_nodes(int): Number of atoms in the simulation
            n_timesteps(int): Number of timesteps in each simulation
            features(list[str]): The list of feature names to track at each time step
            filename(str): The name of the file to save to/load from. If the file already exists,
                the data is loaded from it and checked whether it matches the required parameters,
                unless the force_new parameter is set, in which case it is overwritten anyway. If
                the filename is not specified, the generator will default to a predetermined identifier
                format based on the parameters of the generated set.
            force_new(boolean, optional): Force generation of a new set of simulations,
                instead of using already existing ones from a file.
            discard(boolean, optional): Whether to discard the generated data or to save
                it, useful for debugging. Does not apply if filename is specified
            from_partial(boolean,optional): Whether to load a partial file and corresponding list of configurations,
                and keep building on it until it reaches the specified n_samples. filename must be specified. force_new
                obviously cancels this behavior.
            normalize(boolean, optional): Whether to center data at mean 0 and scale to stddev 1. Defaults true
            normalize_params(tuple[tuple[float]]): Means and standard deviations to center with, for each feature. 
                Used to normalize the validation and test sets with the mean and std. of the training set.
            dset(numpy.ndarray): If this value is not None, a foreign numpy array will be used as the dataset, without 
                the need to generate/read
            existing_config(list[list(str)]): List of unique configs tried, in order.
            existing_indices(list[int]): List of integers indexing into existing_config, with their indices corresponding
                to the sample they belong to.
            threshold(int): The maximum number of missing days in consecutive time steps. If missing > threshold, then
                split into different clusters
        """
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.n_timesteps = n_timesteps
        self.features = features
        self.normalize=normalize
        self.__normalize_params = normalize_params
                
        if dset is None: 
            self.generate_dset(filename, dataset_path, force_new, discard, from_partial, threshold)
        else:
            self.dset=dset
            self.configurations=existing_config
            self.config_indices = existing_indices
            
            print(len(self.configurations), len(self.config_indices), len(self.dset))
            
            assert self.configurations is not None \
                and self.config_indices is not None \
                and len(self.config_indices) == len(self.dset) \
                and len(self.configurations) > 0, \
                "Incorrect custom dset, configurations/indices values given to constructor"

        assert self.dset.shape == (self.n_samples, self.n_nodes, self.n_timesteps, len(self.features)), "Dataset dimensions do not match specifications"
            
        if self.normalize:
            if self.__normalize_params is None:
                self.__normalize_params = tuple((self.dset[:,:,:,feature].mean(),
                                                 self.dset[:,:,:,feature].std()) for feature in range(dset[:].shape[-1]))
            self.__normalize_dataset()

        
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        """
        This function defines the indexing behavior of this dataset object. If normalization is not activated, it
            simply passes the given index onto the underlying numpy array. Otherwise, it performs normalization based
            on self.__normalize_params which is set in the constructor (and here again as a sanity check).
        The normalization is performed at the level of the whole dataset, so that meaningful comparisons can be made
            between samples, as well as between different weather stations belonging to the same sample.
        Args:
            idx(index): The index for accessing the underlying dataset.
        """
        if self.normalize and self.__normalize_params is None:
            shape = self.dset.shape
            self.__normalize_params = tuple((self.dset[:,:,:,feature].mean(), self.dset[:,:,:,feature].std()) for feature in range(shape[-1]))
            self.__normalize_dataset()

        return self.dset[idx]
    
    def get_sample_config(self, idx):
        """
        Returns the generating configuration of the sample corresponding to the index given as a parameter.
        Args:
            idx(index): The index for accessing the underlying configuration information.
        """
        if isinstance(idx, int) or isinstance(idx, splice):
            return self.configurations[self.config_indices[idx]] 
        else:
            assert False, "Samples can only be indexed by integers or single splices"
            
    def get_config_list(self):
        """
        Returns the set of unique sample indices used to generate this dataset.
        """
        return self.configurations
        
    def get_complete_config_list(self):
        """
        Returns the entire list of configurations, corresponding to sample indices.
            There will be many repetitions.
        """
        return [self.configurations[ix] for ix in self.config_indices]
            
    def get_normalize_params(self):
        """
        Returns normalization mean and standard deviation for each feature.
        """
        return self.__normalize_params
    
    def __normalize_dataset(self):
        """
        Normalizes the dataset based on the mean and standard deviation for each feature in self.__normalize_params
        """
        assert self.__normalize_params is not None \
            and len(self.__normalize_params) == len(self.features) \
            and len(self.__normalize_params[0]) == 2, \
            "Normalize params incorrect format. Use ((feature1.mean, feature1.std),...,(feature_n.mean, feature_n.std))"
            
        assert self.dset.shape == (
            self.n_samples, self.n_nodes, self.n_timesteps, len(self.features)
        ), "Dataset dimensions do not match specifications"
                
        for feature in range(self.dset.shape[-1]):
            self.dset[:,:,:,feature] = (
                self.dset[:,:,:,feature] - self.__normalize_params[feature][0]
            ) / (self.__normalize_params[feature][1] if self.__normalize_params[feature][1] != 0 else 1)
            
        self.normalize=True

    def __denormalize_dataset(self):
        """
        Denormalizes the dataset based on the mean and standard deviation for each feature in self.__normalize_params
        """
        assert self.__normalize_params is not None \
            and len(self.__normalize_params) == len(self.features) \
            and len(self.__normalize_params[0]) == 2, \
            "Normalize params incorrect format. Use ((feature1.mean, feature1.std),...,(feature_n.mean, feature_n.std))"
            
        assert self.dset.shape == (
            self.n_samples, self.n_nodes, self.n_timesteps, len(self.features)
        ), "Dataset dimensions do not match specifications"
        
        for feature in range(self.dset.shape[-1]):
            self.dset[:,:,:,feature] = (
                self.dset[:,:,:,feature] * (
                    self.__normalize_params[feature][1] if self.__normalize_params[feature][1] != 0 else 1
                )
            ) + self.__normalize_params[feature][0]
            
        self.normalize=False


    def generate_dset(self, filename, dataset_path, force_new, discard, from_partial, threshold):
        """
        Actual meat and potatoes of the data generation process, same params as constructor.
        """
        
        module_dir = dirname(os.path.abspath(__file__))
        data_dir = join(dirname(dirname(module_dir)), "datasets", "weather") if dataset_path is None else dataset_path
                
        # .npy file name convention if filename is not given:
        # #samples_#nodes_#timesteps_#features_#filesOfSameCharacteristics.npy
        
        # Whether the file in "filename" or some unnamed file with similar parameters already exists
        already_exists = False
        
        # This block searches for existing files with a matching filename or dataset generation parameters
        if filename is None:
        
            npy_files = glob.glob(join(data_dir, "[0-9]*.npy"))
            shuffle(npy_files)
            highest_index = -1

            # Check all existing .npy files for matching dataset generation parameters
            for f in npy_files:
                fname = os.path.basename(f)
                if "partial" in fname:
                    continue
                    
                fname = fname.split(".")[0]
                spl = fname.split("_")
                assert len(spl)==5, ".npy filename error"

                if int(spl[0]) >= self.n_samples and int(spl[1]) == self.n_nodes and int(spl[2]) == self.n_timesteps and int(spl[3]) == len(self.features):

                    already_exists = True
                    fpath = f
                    if int(spl[4]) > highest_index:
                        highest_index = int(spl[4])
                
        else:
            if os.path.isfile(join(data_dir, filename)):
                already_exists = True
            fpath = join(data_dir, filename)
            clean_fname = filename.replace("_partial","").replace(".npy","")

        # Whether to create a new file or use an existing one, if available
        if already_exists and not force_new and not from_partial:
            self.dset, self.configurations, self.config_indices = np.load(fpath, allow_pickle=True)
            if self.n_samples < len(self.dset):
                self.dset = self.dset[:self.n_samples]
                self.config_indices = self.config_indices[:self.n_samples]
                self.configurations = self.configurations[:np.where(
                    (np.array(self.configurations)==self.configurations[self.config_indices[-1]]).all(axis=1)
                )[0][0] + 1]
            
            assert self.dset.shape == (self.n_samples, self.n_nodes, self.n_timesteps, len(self.features)), "Given file name contains dataset of a different shape than specified in parameters."
        
        else:

            all_files = glob.glob(join(data_dir, "*.csv"))
            data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
            
            # Add column in datetime format, should standardize to use this everywhere in this file
            # but it is a time-consuming process which doesn't have much benefit outside of elegance.
            if 'date' not in data.columns:
                data['date'] = pd.to_datetime(data[['year','month','day']])

            existing_dset, existing_config, existing_indices = [], [], []

            if from_partial:
                assert filename is not None and already_exists, "Partial file not given or not found"
                existing_dset, existing_config, existing_indices = np.load(fpath, allow_pickle=True)
                
                assert len(existing_dset) < self.n_samples, \
                    "n_samples given is smaller than #samples in file, no need for partial"
                
                assert existing_dset.shape[1] == self.n_nodes \
                    and existing_dset.shape[2] == self.n_timesteps \
                    and existing_dset.shape[3] == len(self.features), \
                    "Incorrect shape for partial file"
                
                print("Progress recovered from partial file, {} samples".format(len(existing_dset)))

            self.save_file_name = join(data_dir, clean_fname) if filename is not None else join(data_dir,
                                                                                        str(self.n_samples) + "_"
                                                                                        + str(self.n_nodes) + "_"
                                                                                        + str(self.n_timesteps) + "_"
                                                                                        + str(len(self.features)) + "_"
                                                                                        + str(highest_index+1))
            print(self.save_file_name)
            
            partial_save_freq = 0 if discard else 1000

            self.dset, self.configurations, self.config_indices = self.sample_configurations(data, self.n_samples, self.n_nodes, self.n_timesteps, self.features, threshold, existing_dset, existing_config, existing_indices, partial_save_freq=partial_save_freq)
            
            if not discard:
                np.save(self.save_file_name, (self.dset, self.configurations, self.config_indices))
        
    
    def export(self):
        assert self.dset.shape == (
            self.n_samples, self.n_nodes, self.n_timesteps, len(self.features)
        ), "Dataset dimensions do not match specifications"
        
        assert self.configurations is not None \
            and self.config_indices is not None \
            and len(self.config_indices) == len(self.dset) \
            and len(self.configurations) > 0, \
            "Dataset object parameters configurations and config_indices invalid"
        
        return self.dset, self.__normalize_params if self.normalize else None, self.configurations, self.config_indices
    
    @classmethod
    def train_valid_test_split(cls, dset, features, spl=[80,10,10], normalize=False, export=False):
        """
        Gives 3 different dataset objects, for train/valid/test DataLoaders
        Args:
            dset(WeatherDataset): original dataset object
            spl(list[int]): train/valid/test split, must add up to 100
        """
        assert len(spl) == 3 and sum(spl) == 100, "Invalid split given, the 3 values must sum to 100"
        assert len(dset[:].shape) == 4, "Dimensionality of given dataset is incorrect"
        
        n_samples, n_nodes, n_timesteps, n_features = dset[:].shape
        n_train = int(len(dset) * (spl[0] / 100))
        n_valid = int(len(dset) * (spl[1] / 100))
        
        # Training set generation
        train_dset = dset[:n_train]
        train_config_indices = dset.config_indices[:n_train]
        cutoff_train_index = np.where(
                    (np.array(dset.configurations)==dset.configurations[train_config_indices[-1]]).all(axis=1)
                )[0][0] + 1
        train_configurations = dset.configurations[:cutoff_train_index]
        
        train_set = cls(n_train, n_nodes, n_timesteps, features, dset=train_dset, existing_config=train_configurations, existing_indices=train_config_indices, normalize=normalize)
        normalize_params = train_set.get_normalize_params()
        
        # Validation set generation
        valid_dset = dset[n_train: n_train + n_valid]
        valid_config_indices = dset.config_indices[n_train: n_train + n_valid]
        cutoff_valid_index = np.where(
                    (np.array(dset.configurations)==dset.configurations[valid_config_indices[-1]]).all(axis=1)
                )[0][0] + 1
        
        if cutoff_valid_index == cutoff_train_index:
            valid_configurations = dset.configurations[cutoff_train_index-1:cutoff_train_index]
        else:
            valid_configurations = dset.configurations[cutoff_train_index:cutoff_valid_index]
            
        # Fixing the indexing after splitting
        valid_config_indices = valid_config_indices - len(train_configurations)
        
        valid_set = cls(n_valid, n_nodes, n_timesteps, features, dset=dset[n_train:n_train+n_valid], existing_config=valid_configurations, existing_indices=valid_config_indices, normalize=normalize, normalize_params=normalize_params)

        
        # Test set generation
        test_config_indices = dset.config_indices[n_train + n_valid:]
        reverse_cutoff_test_index = np.where(
                    (np.array(dset.configurations)==dset.configurations[test_config_indices[0]]).all(axis=1)
                )[0][0]

        # Fixing the indexing after splitting
        test_config_indices = test_config_indices - (len(train_configurations) + len(valid_configurations))
        
        test_set = cls(n_samples - (n_train + n_valid), n_nodes, n_timesteps, features, dset=dset[n_train+n_valid:], existing_config=dset.configurations[reverse_cutoff_test_index:], existing_indices=test_config_indices, normalize=normalize, normalize_params=normalize_params)
        
        if not export:
            return (train_set, valid_set, test_set)
        
        else:    
            train_set, train_norm_params, train_configurations, train_config_indices = train_set.export()
            valid_set, valid_norm_params, valid_configurations, valid_config_indices = valid_set.export()
            test_set, test_norm_params, test_configurations, test_config_indices = test_set.export()

            return dict(
                normalize_params=train_norm_params,
                
                train_set = train_set,
                train_configurations = train_configurations,
                train_config_indices = train_config_indices,
                                
                valid_set = valid_set,
                valid_configurations = valid_configurations,
                valid_config_indices = valid_config_indices,

                test_set = test_set,
                test_configurations = test_configurations,
                test_config_indices = test_config_indices
            )

    
    @staticmethod
    def remove_deep(src, matching_dates):
        """
        For each row in matching_dataframe, this function removes all matching rows in src.
        Assumes that matching_df only contains colums which we want to match against.
        Args:
            src (pandas DataFrame): The source DataFrame that rows will be removed from
            matching_dates (pd.DataFrame, pd.Series or pd.Index): The target containing
                dates that will be removed from src
        """
        
        src.reset_index(inplace=True, drop=True)
        if len(src) == 0:
            return
        
        src_dates = src['date']
        trg_dates = matching_dates['date'].drop_duplicates() if isinstance(matching_dates, pd.DataFrame) else matching_dates.drop_duplicates()
        
        indexNames = pd.Index([])
        
        for row in trg_dates:
            indexNames = indexNames.union(src_dates[src_dates == row].index)
        
        src.drop(indexNames , inplace=True)
        src.reset_index(inplace=True, drop=True)
        
    
    @staticmethod
    def align_dates(df, features, n_nodes):
        """
        Aligns all stations in a given dataframe by date, by removing all entries for dates
            for which at least one weather station's measurements are missing
        Args:
            df (pandas DataFrame): The source DataFrame that rows will be removed from
            features(list(str)): The list of features which should be aligned in the time dimension
            n_nodes(int): The number of nodes in the configuration.
        """        
        # Check if df contains the right number of nodes, return empty df otherwise
        if n_nodes != len(df['station_id'].unique()):
            df = df.iloc[0:0]
            return
        
        all_present = (df.groupby('date').count()[features]==n_nodes).all(axis=1)
        dates_removal = all_present[all_present==False].index

        WeatherDataset.remove_deep(df, dates_removal)

    @staticmethod
    def get_configuration(df, conf, features):
        """
        Fetches a subset of the original dataset, which only contains the given configuration
            of weather stations, for which the entries that are removed are only those which
            create a conflict with this specific configuration of stations.
        Args:
            df (pandas DataFrame): The source DataFrame that contains all of the (dirty) data
            conf(list(str)): The list weather station IDs which constitute the configuration.
        """
        
        
        # Get timesteps for rows matching configuration
        df_conf = df[df['station_id'].isin(conf)].reset_index(drop=True).sort_values(by='date')

        # Remove non-numerical values, turning them into NaNs
        for feature in features:
            df_conf[feature] = pd.to_numeric(df_conf[feature], errors='coerce')
            
        # Interpolate missing values as much as possible to avoid removing too many time steps
        df_conf[features] = df_conf.groupby('station_id')[features].apply(
            lambda group: group.interpolate(axis=0, method='polynomial', order=2, limit=4)
        )
        
        # Sort out NaNs
        missing_rows = df_conf[df_conf[features].isna().any(axis=1)]
        WeatherDataset.remove_deep(df_conf, missing_rows)

        # Remove timesteps for which not all stations have data
        WeatherDataset.align_dates(df_conf, features, len(conf))

        return df_conf

    def sample_configurations(self, df, n_samples, n_nodes, n_timesteps, features, threshold, existing_dset=[], existing_config = [], existing_indices=[], partial_save_freq = 1000):
        """
        Samples different combinations of weather stations from the pool of candidates, and selectively
            applies the propagated deletion operations specified above for each sample, so as to get all
            possible simulations from each.
        Args:
            n_samples(int): Number of simulations to fetch
            n_nodes(int): Number of atoms in the simulation
            n_timesteps(int): Number of timesteps in each simulation
            features(list(str)): The list of feature names to track at each time step
            threshold(int): Tolerance for consecutive missing dates before splitting into separate groups.
        """
        
        candidates = np.array(df['station_id'].unique())
        
        counter = df.groupby('station_id').count()

        # The combinations of stations that we choose to go with
        configurations = existing_config
        config_indices = existing_indices

        sample_df_size = n_timesteps * n_nodes

        partial_save_thresh = 0 if existing_dset == [] else len(existing_dset) - (len(existing_dset)%partial_save_freq)
        partial_save_thresh += partial_save_freq

        # The number of 'simulations' generated so far
        current_samples = 0 if existing_dset == [] else len(existing_dset)

        # Numpy format dataset
        dataset = np.empty((0, n_nodes, n_timesteps, len(features))) if existing_dset == [] else existing_dset
        
        assert len(config_indices) == len(dataset), "Mismatch between dataset and config_indices array"

        while current_samples < n_samples:
            sample = np.random.choice(candidates, size=n_nodes, replace=False)
                        
            # Skip iteration if we come across a duplicate (1/75287520 chance for 5 nodes)
            if WeatherDataset.nested_list_member(configurations, sample):
                continue

            configurations.append(sample)

            df_conf = WeatherDataset.get_configuration(df, sample, features)
                        
            # If all stations made it through the deletion process, and there are enough time steps for >=1 sample
            if len(df_conf['station_id'].unique()) == n_nodes and len(df_conf) >= sample_df_size:
                
                # Sort values by date
                df_conf = df_conf.sort_values(by='date').reset_index(drop=True)
                
                # Turn datetime objects into ordinal values in order to easily find consecutive clusters
                ordinal_dates = df_conf['date'].apply(pd.datetime.toordinal)
                
                # Group values into quasi-consecutive clusters (up to a threshold specified in function param "threshold")
                # Here we do not cluster unique dates, but rows in the dataframe sorted by date, for which there are n_nodes for
                # each unique date. These naturally get clustered into the same bucket
                consecutive = (ordinal_dates.diff() > threshold).astype('int').cumsum()
                grouped = consecutive.groupby(consecutive)
                
                clusters_indices = []
                                
                # Choose consecutive groups which are longer than the number of timesteps required for each sample
                for g in grouped.groups:
                    group_indices = grouped.get_group(g).index
                    if len(group_indices) >= sample_df_size:
                        clusters_indices.append(group_indices)
                                                
                for cluster in clusters_indices:
                    
                    # Determine the number of whole simulations we can use
                    
                    cluster_df = df_conf.loc[cluster].reset_index(drop=True)
                    
                    samples_in_cluster = len(cluster_df) // (sample_df_size)
                    
                    cluster_df = cluster_df.head(samples_in_cluster * sample_df_size)
                    
                    # Group by station id, fetch features
                    grouped = cluster_df.groupby('station_id')[features]

                    # Convert to numpy array with different groups in different dimensions                
                    groups_np = np.array([grouped.get_group(s_id).values for s_id in sample], dtype=np.float32)
                    
                    # Determine whether we need to use less samples than available, due to reaching size limit
                    samples_to_use = samples_in_cluster if current_samples + samples_in_cluster <= n_samples else n_samples - current_samples
                    
                    current_samples += samples_to_use
                    
                    # Split numpy array by timesteps into different 'simulations'
                    groups_np = np.array(np.split(groups_np, samples_in_cluster, axis=1))
                    
                    # Sample simulations randomly if we need to lower the number due to reaching the required amount of samples in the dataset
                    if samples_to_use < samples_in_cluster:
                        sampled_simulations = np.random.choice(np.arange(samples_in_cluster), size=samples_to_use, replace=False)
                        groups_np = groups_np[sampled_simulations]

                    # Append to dataset array
                    dataset = np.append(dataset, groups_np, axis=0)
                    
                    # Append to config indices array
                    config_indices += [len(configurations) - 1] * samples_to_use

                    if current_samples == n_samples:
                        break

                    # Increase threshold by the frequency when it is passed to simulate modulo behavior for nonconsecutive values
                    if current_samples >= partial_save_thresh and partial_save_freq != 0:
                        partial_save_thresh += partial_save_freq
                        np.save(self.save_file_name + "_partial", (dataset, configurations, config_indices))
                        print("{} samples progress saved in partial file {}".format(len(dataset), self.save_file_name + "_partial.npy"))

                print("Progress: {}/{}".format(current_samples, n_samples))
        
        return dataset, configurations, config_indices

    @staticmethod
    def nested_list_member(nested_list, element):
        """
        Utility function used for checking membership of a sample in the nested list of configurations
            used so far. Necessary because the standard Python formulation of (elem in list) fails in nested lists
        Args:
            nested_list(list(list)): List containing list or np.array
            element(list): Check whether this list element is contained in the list of lists.
        """
        for lst in nested_list:
            if (lst==element).all():
                return True
        return False
