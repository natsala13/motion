'''Moiton utils.'''
import yaml

DEFAULT_CONFIG = 'default'


class EdgePoint(tuple):
    '''Point tupple representation - debug and coding reasons.'''
    def __new__(cls, a, b):
        return super().__new__(cls, [a, b])

    def __repr__(self):
        return f'Edge{super().__repr__()}'


class StaticConfig:
    '''Config file for static motion skeleton'''
    CONFIG_YAML_FILE = 'motion/utils/config.yaml'

    def __getitem__(self, item):
        assert item in self.default_config
        return self.config.get(item, self.default_config[item])

    def __init__(self, character_name: str):
        config = self.load_config(self.CONFIG_YAML_FILE)

        self.character_name = character_name
        self.default_config = config[DEFAULT_CONFIG]
        self.config = config[character_name] if character_name in config else config[DEFAULT_CONFIG]

    @staticmethod
    def load_config(config_path: str):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

        return config
