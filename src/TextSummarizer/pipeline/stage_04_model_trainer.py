import torch
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_trainer import ModelTrainer
from TextSummarizer.logging import logger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):    
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()
        