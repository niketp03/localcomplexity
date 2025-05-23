#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch as ch
import torchvision
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler, AdamW
from torch.func import jacfwd, jacrev
from torch import vmap
import numpy as np
from torch import nn
import torch as ch
import einops

from attacks import PGD

import ml_collections
from tqdm import tqdm
import os
import time
import logging

import wandb


# # Configuration Settings

# In[4]:


def get_config():
  """hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.optimizer = 'adam'
  config.lr = 1e-4
  config.momentum = 0.01
  config.bias_decay = False

  config.train_batch_size = 64 #256
  config.test_batch_size = 64 #256

  config.num_steps = 1000000                       # number of training steps
  config.weight_decay = 0.01

  config.label_smoothing = 0.0

  config.log_steps = np.unique(
      np.logspace(0,np.log10(config.num_steps),50).astype(int).clip(
          0,config.num_steps
          )
      )
  
  config.seed = 42

  #config.dmax = 1
  #config.dmin = 0

  config.wandb_proj = 'LRLC_study_MNIST_MLP'
  config.wandb_pref = 'MNIST-MLP'

  config.resume_step = 0

  ## mlp params
  config.input_dim = 784
  config.output_dim = 10
  config.hidden_dim = 200
  config.n_layers = 4
  config.input_weights_init_scale = np.sqrt(2) #sets initialization standard deviation to be = (input_weights_init_scale)/sqrt(input_dim) 
  config.output_weights_init_scale = np.sqrt(2)


  ## local complexity approx. parameters
  config.compute_LC = True
  config.n_batches = 2
  config.sigma = 0.01
  config.n_iters_LC = 1

  ## adv robustness parameters
  config.compute_robust = True                   # note that if normalize==True, data is not bounded between [0,1]
  config.atk_eps = 80/255   ## 8/255
  config.atk_alpha = 10/255  ## 2/255
  config.atk_itrs = 10
  config.dmin = -2.0
  config.dmax = 2.0

  ## data rank evaluation
  config.eval_data_rank = True
  config.n_batchs_for_rank = 10

  ## data local rank evaluation
  config.local_rank_eval = True
  config.n_batchs_for_local_rank = 1

  ##
  config.save_model = True
  config.model_save_dir = 'models'

  return config


config = get_config()


# # Data Generation

# In[5]:


from torch.utils.data import Subset
import random

train_dataset = torchvision.datasets.MNIST('../mnist_data',
                                           download=True,
                                           train=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(), # first, convert image to PyTorch tensor
                                               transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                                               lambda x: einops.rearrange(x, 'c h w -> (c h w)') # flatten the input images
                                           ]))
idx = list(range(len(train_dataset)))
random.shuffle(idx)
train_dataset = Subset(train_dataset, idx[:1000])

train_loader = ch.utils.data.DataLoader(train_dataset,
                                        batch_size=config.train_batch_size,
                                        shuffle=True)

test_dataset = torchvision.datasets.MNIST('../mnist_data',
                                           download=True,
                                           train=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(), # first, convert image to PyTorch tensor
                                               transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
                                               lambda x: einops.rearrange(x, 'c h w -> (c h w)') # flatten the input images
                                           ]))
idx = list(range(len(test_dataset)))
random.shuffle(idx)
train_dataset = Subset(test_dataset, idx[:1000])

test_loader = ch.utils.data.DataLoader(train_dataset,
                                        batch_size=config.train_batch_size,
                                        shuffle=True)


# # Model Definition

# In[6]:


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, input_std_init=1, output_std_init=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Define input layer
        self.layers = nn.ModuleList()
        self.initial_weights = []
        
        # Input layer
        input_layer = nn.Linear(input_dim, hidden_dim)
        initial_init = ch.normal(mean=ch.zeros(hidden_dim, input_dim), std=input_std_init / ch.sqrt(ch.tensor(input_dim)))
        input_layer.weight = nn.parameter.Parameter(initial_init.clone())
        input_layer.bias = nn.parameter.Parameter(ch.zeros(hidden_dim))
        self.layers.append(input_layer)
        self.input_layer = input_layer
        self.initial_weights.append(initial_init)

        # Hidden layers
        for _ in range(n_layers - 1):
            hidden_layer = nn.Linear(hidden_dim, hidden_dim)
            fan_in = self.layers[-1].weight.shape[1]
            hidden_init = ch.normal(mean=ch.zeros(hidden_dim, hidden_dim), std=output_std_init / ch.sqrt(ch.tensor(fan_in)))
            hidden_layer.weight = nn.parameter.Parameter(hidden_init.clone())
            hidden_layer.bias = nn.parameter.Parameter(ch.zeros(hidden_dim))
            self.layers.append(hidden_layer)
            self.initial_weights.append(hidden_init)

        # Define output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.output_init = ch.normal(mean=ch.zeros(output_dim, hidden_dim), std=output_std_init / ch.sqrt(ch.tensor(hidden_dim)))
        self.output_layer.weight = nn.parameter.Parameter(self.output_init.clone())
        self.output_layer.bias = nn.parameter.Parameter(ch.zeros(output_dim))
        
        # Activation functions
        self.input_activation = nn.ReLU()
        self.output_activation = nn.Softmax()

    def forward(self, x):
        #x = einops.rearrange(x, "b c u i -> b c (u i)")
        for layer in self.layers:
            x = self.input_activation(layer(x))
        x = self.output_layer(x)
        x = self.output_activation(x)
        #x = einops.rearrange(x, "x 1 n -> x n")
        return x.float()
    
    def forward_to(self, x, n_layer):
        #x = einops.rearrange(x, "b c u i -> b c (u i)")
        for layer in self.layers[:n_layer]:
            x = self.input_activation(layer(x))
        x = self.layers[n_layer](x)
        return x

    def get_dist_to_init(self):
        dists = []
        for layer, init in zip(self.layers, self.initial_weights):
            layer_dif = ch.norm(layer.weight.detach().cpu() - init.detach().cpu()).detach().item()
            dists.append(layer_dif)
        layer_2_dif = ch.norm(self.output_layer.weight.detach().cpu() - self.output_init.detach().cpu()).detach().item()
        dists.append(layer_2_dif)
        return dists


# # Eval Functions

# In[7]:


@ch.no_grad
def evaluate(model, dloader, loss_fn=None):
  
  model.eval()

  acc = 0
  loss = 0
  nsamples = 0
  nbatch = 0
  
  for inputs, targets in dloader:
      
      inputs = inputs.cuda()
      targets = targets.cuda()
      outputs = model(inputs)

      if loss_fn is not None:
        loss += loss_fn(outputs, targets).item()
        nbatch += 1
              
      acc += ch.sum(targets == outputs.argmax(dim=-1)).cpu()
      nsamples += outputs.shape[0]

  model.train()

  return acc/nsamples, loss/nbatch

@ch.no_grad
def evaluate_local_data_rank(model, dloader, layer_number, n_batch):
  
  model.eval()
  
  ranks_all = []
  tvs_all = []
  for inputs, _ in dloader:
     #ims = ch.autograd.Variable(ims.cuda(), requires_grad=True)
     #jacobians_batch = jacfwd(lambda x: model.forward_to(x, layer_number).sum(0))(ims.cuda())
     func = jacfwd(lambda x: model.forward_to(x, layer_number))
     vmaped_func = vmap(func, in_dims = 0, out_dims = 0)

     jacobians_batch = vmaped_func(inputs.cuda())

     #print(jacobians_batch.shape)   
     #jacobians_batch = einops.rearrange(jacobians_batch, 'b 1 1 o 1 x y -> b o (x y)')

     rank_batch = ch.linalg.matrix_rank(jacobians_batch, atol = .01)
     tv_batch = ch.linalg.norm(jacobians_batch, dim = (1,2))

     tvs_all.append(tv_batch)
     ranks_all.append(rank_batch)

     if len(ranks_all) > n_batch:
       break
     
  ranks_all = ch.cat(ranks_all, dim=0).to(ch.float)
  tvs_all = ch.cat(tvs_all, dim=0).to(ch.float)
  rank_average = ranks_all.mean().item()
  tvs_all = tvs_all.mean().item()
  model.train()
  
  return rank_average, tvs_all

@ch.no_grad
def compute_local_complexity(model, dloader, config):
  def add_noise_to_bias(model, sigma):
    """
    Clones the model and adds Gaussian noise with variance sigma to all the bias terms.
    """

    model_clone = MLP(input_dim = config.input_dim, hidden_dim = config.hidden_dim, output_dim = config.output_dim, n_layers=config.n_layers, input_std_init=config.input_weights_init_scale, output_std_init=config.output_weights_init_scale)

    model_clone.load_state_dict(model.state_dict())

    for layer in model_clone.layers:
      if isinstance(layer, nn.Linear):
        noise = ch.randn_like(layer.bias) * sigma
        layer.bias.data += noise
    return model_clone
  
  def normal_distribution_pdf(x, sigma):
    return ch.exp(-((x) ** 2) / (2 * sigma ** 2))

  def evaluate_layer_jac(model, l, x_in, sigma):
    """
    Computes summand of local compelxity estimator for a given layer l
    """
    #ims = ch.autograd.Variable(ims.cuda(), requires_grad=True)
    #jacobians_batch = jacfwd(lambda x: model.forward_to(x, layer_number).sum(0))(ims.cuda())
    #ims = rearrange(ims, 'b h w -> b (h w)')
    func = jacfwd(lambda x: model.forward_to(x, l))
    vmaped_func = vmap(func, in_dims = 0, out_dims = 0)

    jacobians_batch = vmaped_func(x_in)
    dist_from_bias = model.forward_to(x_in, l) - model.layers[l].bias
    bias_term = normal_distribution_pdf(dist_from_bias, sigma = sigma)

    neuron_norms_sum = []

    for i in range(jacobians_batch.shape[1]): #iter over neurons
      neuron_norms = jacobians_batch[:,i].norm(dim=1)
      summand = bias_term[:,i] * neuron_norms
      neuron_norms_sum.append(summand)

    neuron_norms_sum  = ch.stack(neuron_norms_sum)

    return neuron_norms_sum.sum(axis = 0)
   
  # Done with Helpers
  running_mean = []
  for inputs, _ in dloader:
    to_stack_2 = []
    x = inputs.cuda()

    # Iterate over choices for biases
    for _ in range(config.n_iters_LC):
      to_stack = []

      model_copy = add_noise_to_bias(model, config.sigma).to('cuda')

      # Iterate over layers
      for l in range(config.n_layers):
        neuron_norm_sum_per_x = evaluate_layer_jac(model_copy, l, x, config.sigma)
        to_stack.append(neuron_norm_sum_per_x)

      del model_copy

      to_stack = ch.stack(to_stack)
      sum_x = to_stack.sum(axis = 0)
      to_stack_2.append(sum_x)

    to_stack_2 = ch.stack(to_stack_2)

    to_stack_2 = to_stack_2.sum(axis = (0,1)) / (config.n_iters_LC * len(inputs))
    running_mean.append(to_stack_2.item())

  running_mean = ch.tensor(running_mean).mean().item()
  return running_mean

def evaluate_adv(model, dloader, config):

  atk = PGD(model,
          eps=config.atk_eps,
          alpha=config.atk_alpha,
          steps=config.atk_itrs,
          dmin=config.dmin,
          dmax=config.dmax
          )

  acc = 0
  nsamples = 0
  for inputs, targets in tqdm(dloader, desc=f"Computing robust acc for eps:{config.atk_eps:.3f}"):

    inputs = inputs.cuda()
    targets = targets.cuda()

    adv_images = atk(inputs, targets)

    with ch.no_grad():
        adv_pred = model(adv_images).argmax(dim=-1)

    acc += ch.sum(targets == adv_pred).cpu()
    nsamples += adv_pred.shape[0]

  return acc/nsamples


# # Training Loop

# In[8]:


def train(model, loaders, config):
    print('Training....')
    print(f'Logging at steps: {config.log_steps}')

    model.cuda()

    # No Weight Decay on Biases
    decay = dict()
    no_decay = dict()
    for name, param in model.named_parameters():
        print('checking {}'.format(name))
        if 'weight' in name:
            decay[name] = param
        else:
            no_decay[name] = param
        
    print(f"Weight Decay on: {decay.keys()}")
    print(f"Weight Decay off: {no_decay.keys()}")

    # Optimizer Setup
    
    if config.optimizer == 'sgd':
        print('Using SGD optimizer')
        if not config.bias_decay:
            opt = SGD([
                {'params': no_decay.values(), 'weight_decay': 0.0},
                {'params': decay.values(), 'weight_decay': config.weight_decay}
            ],
            lr=config.lr,
            momentum=config.momentum)
        else:
            opt = SGD(model.parameters(),
                  lr=config.lr,
                  momentum=config.momentum,
                  weight_decay=config.weight_decay)

    elif config.optimizer == 'adam':
        if not config.bias_decay:
            opt = AdamW([
                {'params': no_decay.values(), 'weight_decay': 0.0},
                {'params': decay.values(), 'weight_decay': config.weight_decay}
            ],
            lr=config.lr)
        else:   
            opt = AdamW(model.parameters(),
                    lr=config.lr,
                    weight_decay=config.weight_decay)

    else:
        raise NotImplementedError
    
    iters_per_epoch = len(loaders['train'])
    epochs = np.floor(config.num_steps/iters_per_epoch)
    print(f"Training for {epochs} epochs")

    loss_fn = ch.nn.CrossEntropyLoss()

    train_step = 0

    while True:
        if train_step > config.num_steps: break

        for input, targets in loaders['train']:

            model.train()

            input = input.cuda()
            targets = targets.cuda()

            opt.zero_grad()
            out = model(input)

            loss = loss_fn(out, targets)
            loss.backward()
            opt.step()
            train_step += 1

            # Calculate Stats
            print('Logging Evaluations')
            if train_step in config.log_steps:
                model.eval()

                train_acc, train_loss = evaluate(model,
                                                 loaders['train'],
                                                 loss_fn)
                test_acc, test_loss = evaluate(model,
                                                 loaders['test'],
                                                 loss_fn)
                
                l2_norm = 0
                for name, param in model.named_parameters():
                    l2_norm += ch.norm(param)

                if config.compute_LC:
                    print('Computing Local Complexity')
                    train_LC = compute_local_complexity(model, loaders['train'], config)
                    test_LC = compute_local_complexity(model, loaders['test'], config)
                else:
                    train_LC = 0
                    test_LC = 0
                    

                if config.local_rank_eval:
                    print('Computing Local Rank')
                    local_ranks = []
                    for layer_number in range(0, config.n_layers):
                        local_rank, tv = evaluate_local_data_rank(model, loaders['test'], layer_number, config.n_batchs_for_local_rank)
                        local_ranks.append(local_rank)

                if config.compute_robust:
                    print('Computing Robustness')
                    robust_acc = evaluate_adv(model, loaders['test'], config)

                stats_dict = {
                    'iter': train_step,
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'test/loss': test_loss,
                    'test/acc': test_acc,
                    'adv/acc': robust_acc,
                    'l2_norm': l2_norm,
                    'train/LC': train_LC,
                    'test/LC': test_LC,
                    'TV': tv,
                } | {f'train/local_rank_{i}': rank for i, rank in enumerate(local_ranks)}

                wandb.log(stats_dict)

                if config.save_model:
                    ch.save(model.state_dict(), os.path.join(config.model_save_dir, f'model_{train_step}.pth'))

                model.train()


# # Run

# In[12]:


# init wandb
wandb.init(project="mnist-analysis", config=config)

# Create model based on config
def create_model(config):
    model = MLP(
        input_dim=784,  # MNIST images are 28x28 = 784 pixels
        hidden_dim=config.hidden_dim,
        output_dim=10,  # 10 classes for MNIST
        n_layers=config.n_layers
    )
    return model

# Initialize model
model = create_model(config)
print(f"Created MLP with {config.n_layers} layers and {config.hidden_dim} hidden dimensions")

# Create data loaders dictionary
loaders = {
    'train': train_loader,
    'test': test_loader
}

# Train the model
train(model, loaders, config)


# In[ ]:




