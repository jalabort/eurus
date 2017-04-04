from terrarium import Config, config_value


class DataLoaderConfig(Config):
    r"""
    Configuration for :class:`torch.utils.data.DataLoader`.
    """
    @property
    @config_value
    def batch_size(self):
        r"""
        The number of examples per batch.
        
        :rtype: int
        """
        return 16

    @property
    @config_value
    def shuffle(self):
        r"""
        Whether to shuffle the data at every epoch or not.
        
        :rtype: bool
        """
        return False

    @property
    @config_value
    def num_workers(self):
        r"""
        The number of sub-processes to use for data loading. If `0` data will 
        be loaded using the main process
        
        :rtype: int
        """
        return 0

    @property
    @config_value
    def pin_memory(self):
        r"""
        Whether to use pin memory or not.
        
        :rtype: bool
        """
        return True


class CrayonConfig(Config):
    r"""
    """
    @property
    @config_value
    def server_address(self):
        r"""
        

        :rtype: str
        """
        return self.RequiredConfig()

    @property
    @config_value
    def experiment_name(self):
        r"""
        
        
        :rtype: str
        """
        return None


class TrainTrackingModelConfig(Config):
    r"""
    Configuration for :class:`eurus.track.pytorch.train.train_tracking_model`.
    """
    @property
    @config_value
    def dataset_config(self):
        r"""
        Configuration for :class:`eurus.track.pytorch.train.Dataset`.
        
        :rtype: :class:`eurus.track.pytorch.train.TrackingDatasetConfig`
        """
        return self.RequiredConfig()

    @property
    @config_value
    def dataloader_config(self):
        r"""
        Configuration for :class:`torch.utils.data.DataLoader`.
        
        :rtype: :class:`eurus.track.pytorch.train.config.TrackingDatasetConfig`
        """
        return DataLoaderConfig()

    @property
    @config_value
    def tracking_model_config(self):
        r"""
        Configuration for :class:`eurus.track.pytorch.train.TackingModel`.
        
        :rtype: :class:`eurus.track.pytorch.train.TrackingModelConfig`
        """
        return self.RequiredConfig()

    @property
    @config_value
    def crayon_config(self):
        r"""
        

        :rtype: :class:`eurus.track.pytorch.train.config.CrayonConfig` | None
        """
        return self.RequiredConfig()

    @property
    @config_value
    def weights_file(self):
        r"""
        Path to PyTorch weights. 

        :rtype: str
        """
        return self.RequiredConfig()

    @property
    @config_value
    def n_epochs(self):
        r"""
        The number of epoch to run the training for.
        
        :rtype: int
        """
        return 100

    # TODO: Optimizer config?
    @property
    @config_value
    def lr(self):
        r"""
        
        
        :rtype: int
        """
        return 1e-3
