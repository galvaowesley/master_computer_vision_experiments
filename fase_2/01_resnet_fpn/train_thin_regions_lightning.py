from pathlib import Path
import torch
from torch import optim, nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from lightning.pytorch.utilities import rank_zero_only
import torchtrainer   #https://github.com/chcomin/torchtrainer
import dataset_thin_region
import segmentation_models_pytorch as smp


default_params = {
    # Dataset
    'img_dir': None,                    # Images path
    'label_dir': None,                  # Labels path
    'crop_size': (256, 256),            # Crop size for training
    'train_val_split': 0.1,             # Train/validation split
    'use_transforms': False,            # Use data augmentation
    # Model
    'model_layers': (3, 3, 3),          # Number of residual blocks at each layer of the model
    'model_channels': (16,32,64),       # Number of channels at each layer
    'model_type': 'unet',               # Model to use
    # Training
    'epochs': 1,
    'lr': 0.01,
    'batch_size_train': 8,
    'batch_size_valid': 8, 
    'momentum': 0.9,                    # Momentum for optimizer
    'weight_decay': 0.,
    'seed': 12,                         # Seed for random number generators
    'loss': 'cross_entropy',
    'scheduler_power': 0.9,             # Power por the polynomial scheduler
    'class_weights': (0.367, 0.633),    # Weights to use for cross entropy
    # Efficiency
    'device': 'cuda',
    'num_workers': 3,                   # Number of workers for the dataloader
    'use_amp': True,                    # Mixed precision
    'pin_memory': False,            
    'non_blocking': False,
    # Logging
    'log_dir': 'logs_unet',             # Directory for logging metrics and model checkpoints
    'experiment':'unet_l_3_c_16_32_64', # Experiment tag
    'save_every':1,                     # Number of epochs between checkpoints
    'save_best':True,                   # Save model with best validation loss
    'meta': None,                       # Additional metadata to save
    # Other
    'resume': False,                    # Resume from previous training
}

def process_params(user_params):
    '''Use default value of a parameter in case it was not provided'''

    params = default_params.copy()
    for k, v in user_params.items():
        params[k] = v

    if params['meta'] is None:
        params['meta'] = params.copy()
    else:
        params['meta'] = (params['meta'], params.copy())

    return params

def initial_setup(params):

    torch.set_float32_matmul_precision('high')
    
    # Set deterministic training if a seed is provided
    seed = params['seed']
    if seed is not None:
        # workers=True sets different seeds for each worker.
        pl.seed_everything(seed, workers=True)

    experiment_folder = Path(params['log_dir'])/str(params["experiment"])
    experiment_folder.mkdir(parents=True, exist_ok=True)

    return experiment_folder

class LitSeg(pl.LightningModule):
    def __init__(self, model_layers, model_channels, loss, class_weights, lr, momentum, weight_decay, iters, scheduler_power, 
                 model_type, meta):
        super().__init__()
        self.save_hyperparameters()  # Add __init__ parameters to checkpoint file

        # Define loss function
        if loss=='cross_entropy':
            loss_func = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=self.device), ignore_index=255) 
        elif loss=='label_weighted_cross_entropy':
            loss_func = torchtrainer.perf_funcs.LabelWeightedCrossEntropyLoss()

        # Model
        if model_type=='unet':
            model = torchtrainer.models.resunet.ResUNet(model_layers, model_channels,  in_channels=3)
        elif model_type=='resnetseg':
            model = torchtrainer.models.resnet_seg.ResNetSeg(model_layers, model_channels,  in_channels=3)
        elif model_type=='unet2':
             model = torchtrainer.models.resunet.ResUNetV2(model_layers, model_channels,  in_channels=3)
        elif model_type=='resnet_fpn':
            model = smp.FPN(
                encoder_name="resnet50",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=2,                      # model output channels (number of classes in your dataset)
            )
             
        self.loss_func = loss_func
        self.learnin_rate = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.iters = iters
        self.scheduler_power = scheduler_power
        self.model = model

    def forward(self, x):
        '''Defining this method allows using an instance of this class as litseg(x).'''
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_func(output, y)
        acc = torchtrainer.perf_funcs.segmentation_accuracy(output, y, ('iou', 'prec', 'rec'))

        # In case something goes wrong
        if torch.isnan(loss):
            checkpoint = {
                "model": self.model,
                'batch': batch,
                "batch_idx": batch_idx,
            }
            torch.save(checkpoint, 'error_ckp.pth')

        self.log("val_loss", loss, prog_bar=True)       
        self.log_dict(acc, prog_bar=True)   
        self.log("hp_metric", loss)   # Metric to show with hyperparameters in Tensorboard

    def configure_optimizers(self):
        #optimizer = optim.SGD(self.parameters(), lr=self.learnin_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(self.parameters(), lr=self.learnin_rate, betas=(self.momentum, 0.999), weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.iters, power=self.scheduler_power)
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
        }

        return {'optimizer':optimizer, 'lr_scheduler':lr_scheduler_config}

class MyLogger(Logger):
    '''Simple class for logging performance metrics.'''
    
    def __init__(self):
        self.metrics = {}

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        return 1

    @rank_zero_only
    def log_hyperparams(self, params):
        self.params = params

    @rank_zero_only
    def log_metrics(self, metrics, step):
        saved_metrics = self.metrics
        for k, v, in metrics.items():
            if k in saved_metrics:
                saved_metrics[k].append((step,v))
            else:
                saved_metrics[k] = [(step,v)]

def train(ds_train, ds_valid, experiment_folder, model_layers, model_channels, model_type, loss, class_weights, epochs, lr, batch_size_train, batch_size_valid, momentum=0.9, 
          weight_decay=0., scheduler_power=0.9, num_workers=0, use_amp=False, pin_memory=False, resume=False, save_every=1, save_best=True, seed=None, log_dir='.', 
          experiment='1', meta=None, **kwargs):

    # Mixed precision
    if use_amp:
        precision = '16-mixed'
    else:
        precision = '32-true'

   # Create dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,   # Avoid recreating workers at each epoch
        collate_fn=dataset_thin_region.collate_fn
    )

    data_loader_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=batch_size_valid,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers>0,   # Avoid recreating workers at each epoch
        collate_fn=dataset_thin_region.collate_fn
        
    )
    total_iters = len(data_loader_train)*epochs   # For scheduler
    batch_size_train = batch_size_train
    batches_per_epoch = len(ds_train)//batch_size_train + 1*(len(ds_train)%batch_size_train>0)

    if resume:
        # Resume previous experiment
        checkpoint_file = experiment_folder/'checkpoints/last.ckpt'
        seed = seed
        lit_model = LitSeg.load_from_checkpoint(checkpoint_file) 
        start_epoch = lit_model.current_epoch + 1
        if seed is not None:
            # Seed using the current epoch to avoid using the same seed as in epoch 0 when resuming
            pl.seed_everything(seed+start_epoch, workers=True)
    else:
        checkpoint_file = None
        lit_model = LitSeg(model_layers, model_channels, loss, class_weights, lr, momentum, weight_decay, 
                           total_iters, scheduler_power, model_type, meta)
        start_epoch = 0

    callbacks = [LearningRateMonitor()]
    if save_best:
        # Create callback for saving model with best validation loss
        checkpoint_loss = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min", 
                                            filename="best_val_loss-{epoch:02d}-{val_loss:.2f}")
        callbacks.append(checkpoint_loss)
    # Callback for saving the model at the end of each epoch
    callbacks.append(ModelCheckpoint(save_last=True, every_n_epochs=save_every))

    logger_tb = TensorBoardLogger('.', name=log_dir, version=experiment)
    logger = MyLogger()

    trainer = pl.Trainer(max_epochs=start_epoch+epochs, callbacks=callbacks, precision=precision, logger=[logger_tb, logger], log_every_n_steps=batches_per_epoch)
    trainer.fit(lit_model, data_loader_train, data_loader_valid, ckpt_path=checkpoint_file)

    return trainer, lit_model

def run(user_params):

    params = process_params(user_params)
    experiment_folder = initial_setup(params)

    # Dataset
    ds_train, ds_valid = dataset_thin_region.create_datasets(params['img_dir'], params['label_dir'], params['crop_size'], params['train_val_split'])

    trainer, lit_model = train(ds_train, ds_valid, experiment_folder, **params)

    return trainer, ds_train, ds_valid, lit_model 