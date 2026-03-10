from data.datamgr import SetDataManager, SimpleDataManager
import os

class BaseLoader():
    def __init__(self, params):
        self.params = params
        self.configures = []
        self.loader_names = []
        self.worker = min(os.cpu_count(), 4)

    def name(self):
        return 'BaseLoader'

    def setup(self):
        """Load networks, create schedulers"""
        for cfg in self.configures:
            self.loader_names.append(cfg['name'])
            loader = self.create_loader(cfg)
            att_name = cfg['name']
            setattr(self, att_name, loader)
            print('create dataset:{} {} '.format(self.params.dataset, att_name))

    @property
    def config_template(self):
        template = {
            'training': True,
            'fewshot': False,
            'image_size': 224,
            'fewshot_params': None,
            'fix': False,
            'pin_memory': False,
            'data_file': '',
            'batch_size': 128,
            'shuffle': True,
            'aug': True,
            'return_gt': False,
            'weighted_samper': False,
            'strong_aug': False
        }
        return template

    def create_config(self, lines):
        configs = []
        for l in lines:
            config = self.config_template
            for k, v in l.items():
                config[k] = v
            configs.append(config)
        return configs

    def create_loader(self, config: dict):
        def fix_loader(loader, length):
            t = []
            for i, data in enumerate(loader):
                if i >= length and i != -1: break
                t.append(data)
            return t

        length = -1
        if config['fewshot']:
            training, fewshot_params = config['training'], config['fewshot_params']
            if fewshot_params is None:
                if training:
                    way, shot, qry, epi = self.params.train_way, self.params.train_shot, self.params.train_query, self.params.train_episode
                else:
                    way, shot, qry, epi = self.params.way, self.params.shot, self.params.query, self.params.test_episode
            else:
                way, shot, qry, epi = fewshot_params
            length = epi
            datamgr = SetDataManager(config['image_size'], way, shot,qry,epi,strong_aug=config['strong_aug'])
            data_loader_params = dict(num_workers=self.worker, pin_memory=config['pin_memory'],persistent_workers=True)
            loader = datamgr.get_data_loader(config['data_file'], aug=config['aug'], data_loader_params=data_loader_params, return_gt=config['return_gt'],weighted_sampler=config['weighted_samper'])

        else:
            datamgr = SimpleDataManager(config['image_size'],batch_size=config['batch_size'])
            data_loader_params = dict(shuffle = config['shuffle'], num_workers=self.worker, pin_memory=config['pin_memory'], persistent_workers=False)
            loader = datamgr.get_data_loader(config['data_file'],config['aug'], data_loader_params)

        if config['fix']:
            return fix_loader(loader, length)
        else:
            return loader