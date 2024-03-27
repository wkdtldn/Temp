import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

class NLPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass
  
    def configure_optimizers(self):
        pass
  
    def loss_fn(self, output, target):
        pass 
  
    def training_step(self):
        pass
  
    def validation_step(self):
        pass