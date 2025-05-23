import json, nbformat, re, os, textwrap, pathlib, io

# User-provided corrupted notebook snippet (truncated in their message).
# We'll treat it as plain text and extract code/markdown cell sources from it.
corrupted_snippet = r"""
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch as ch\n",
    "import torchvision\n",
    "from torchvision import transforms\n",
    "from torch.nn import CrossEntropyLoss\n",
    "from torch.optim import SGD, lr_scheduler, AdamW\n",
    "from torch.func import jacfwd, jacrev\n",
    "from torch import vmap\n",
    "import numpy as np\n",
    "from torch import nn\n",
    "import torch as ch\n",
    "import einops\n",
    "\n",
    "from attacks import PGD\n",
    "\n",
    "import ml_collections\n",
    "from tqdm import tqdm\n",
    "import os\n",
    "import time\n",
    "import logging\n",
    "\n",
    "import wandb"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Configuration Settings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_config():\n",
    "  \"\"\"hyperparameter configuration.\"\"\"\n",
    "  config = ml_collections.ConfigDict()\n",
    "\n",
    "  config.optimizer = 'adam'\n",
    "  config.lr = 1e-4\n",
    "  config.momentum = 0.01\n",
    "  config.bias_decay = False\n",
    "\n",
    "  config.train_batch_size = 64 #256\n",
    "  config.test_batch_size = 64 #256\n",
    "\n",
    "  config.num_steps = 1000000                       # number of training steps\n",
    "  config.weight_decay = 0.01\n",
    "\n",
    "  config.label_smoothing = 0.0\n",
    "\n",
    "  config.log_steps = np.unique(\n",
    "      np.logspace(0,np.log10(config.num_steps),50).astype(int).clip(\n",
    "          0,config.num_steps\n",
    "          )\n",
    "      )\n",
    "  \n",
    "  config.seed = 42\n",
    "\n",
    "  #config.dmax = 1\n",
    "  #config.dmin = 0\n",
    "\n",
    "  config.wandb_proj = 'LRLC_study_MNIST_MLP'\n",
    "  config.wandb_pref = 'MNIST-MLP'\n",
    "\n",
    "  config.resume_step = 0\n",
    "\n",
    "  ## mlp params\n",
    "  config.input_dim = 784\n",
    "  config.output_dim = 10\n",
    "  config.hidden_dim = 200\n",
    "  config.n_layers = 4\n",
    "  config.input_weights_init_scale = np.sqrt(2) #sets initialization standard deviation to be = (input_weights_init_scale)/sqrt(input_dim) \n",
    "  config.output_weights_init_scale = np.sqrt(2)\n",
    "\n",
    "\n",
    "  ## local complexity approx. parameters\n",
    "  config.compute_LC = True\n",
    "  config.n_batches = 2\n",
    "  config.sigma = 0.01\n",
    "  config.n_iters_LC = 1\n",
    "\n",
    "  ## adv robustness parameters\n",
    "  config.compute_robust = True                   # note that if normalize==True, data is not bounded between [0,1]\n",
    "  config.atk_eps = 80/255   ## 8/255\n",
    "  config.atk_alpha = 10/255  ## 2/255\n",
    "  config.atk_itrs = 10\n",
    "  config.dmin = -2.0\n",
    "  config.dmax = 2.0\n",
    "\n",
    "  ## data rank evaluation\n",
    "  config.eval_data_rank = True\n",
    "  config.n_batchs_for_rank = 10\n",
    "\n",
    "  ## data local rank evaluation\n",
    "  config.local_rank_eval = True\n",
    "  config.n_batchs_for_local_rank = 1\n",
    "\n",
    "  ##\n",
    "  config.save_model = True\n",
    "  config.model_save_dir = 'models'\n",
    "\n",
    "  return config\n",
    "\n",
    "\n",
    "config = get_config()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Generation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch.utils.data import Subset\n",
    "import random\n",
    "\n",
    "train_dataset = torchvision.datasets.MNIST('../mnist_data',\n",
    "                                           download=True,\n",
    "                                           train=True,\n",
    "                                           transform=transforms.Compose([\n",
    "                                               transforms.ToTensor(), # first, convert image to PyTorch tensor\n",
    "                                               transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs\n",
    "                                               lambda x: einops.rearrange(x, 'c h w -> (c h w)') # flatten the input images\n",
    "                                           ]))\n",
    "idx = list(range(len(train_dataset)))\n",
    "random.shuffle(idx)\n",
    "train_dataset = Subset(train_dataset, idx[:1000])\n",
    "\n",
    "train_loader = ch.utils.data.DataLoader(train_dataset,\n",
    "                                        batch_size=config.train_batch_size,\n",
    "                                        shuffle=True)\n",
    "\n",
    "test_dataset = torchvision.datasets.MNIST('../mnist_data',\n",
    "                                           download=True,\n",
    "                                           train=False,\n",
    "                                           transform=transforms.Compose([\n",
    "                                               transforms.ToTensor(), # first, convert image to PyTorch tensor\n",
    "                                               transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs\n",
    "                                               lambda x: einops.rearrange(x, 'c h w -> (c h w)') # flatten the input images\n",
    "                                           ]))\n",
    "idx = list(range(len(test_dataset)))\n",
    "random.shuffle(idx)\n",
    "train_dataset = Subset(test_dataset, idx[:1000])\n",
    "\n",
    "test_loader = ch.utils.data.DataLoader(train_dataset,\n",
    "                                        batch_size=config.train_batch_size,\n",
    "                                        shuffle=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Definition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "class MLP(nn.Module):\n",
    "    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, input_std_init=1, output_std_init=1):\n",
    "        super(MLP, self).__init__()\n",
    "        self.input_dim = input_dim\n",
    "        self.output_dim = output_dim\n",
    "        self.n_layers = n_layers\n",
    "\n",
    "        # Define input layer\n",
    "        self.layers = nn.ModuleList()\n",
    "        self.initial_weights = []\n",
    "        \n",
    "        # Input layer\n",
    "        input_layer = nn.Linear(input_dim, hidden_dim)\n",
    "        initial_init = ch.normal(mean=ch.zeros(hidden_dim, input_dim), std=input_std_init / ch.sqrt(ch.tensor(input_dim)))\n",
    "        input_layer.weight = nn.parameter.Parameter(initial_init.clone())\n",
    "        input_layer.bias = nn.parameter.Parameter(ch.zeros(hidden_dim))\n",
    "        self.layers.append(input_layer)\n",
    "        self.input_layer = input_layer\n",
    "        self.initial_weights.append(initial_init)\n",
    "\n",
    "        # Hidden layers\n",
    "        for _ in range(n_layers - 1):\n",
    "            hidden_layer = nn.Linear(hidden_dim, hidden_dim)\n",
    "            fan_in = self.layers[-1].weight.shape[1]\n",
    "            hidden_init = ch.normal(mean=ch.zeros(hidden_dim, hidden_dim), std=output_std_init / ch.sqrt(ch.tensor(fan_in)))\n",
    "            hidden_layer.weight = nn.parameter.Parameter(hidden_init.clone())\n",
    "            hidden_layer.bias = nn.parameter.Parameter(ch.zeros(hidden_dim))\n",
    "            self.layers.append(hidden_layer)\n",
    "            self.initial_weights.append(hidden_init)\n",
    "\n",
    "        # Define output layer\n",
    "        self.output_layer = nn.Linear(hidden_dim, output_dim)\n",
    "        self.output_init = ch.normal(mean=ch.zeros(output_dim, hidden_dim), std=output_std_init / ch.sqrt(ch.tensor(hidden_dim)))\n",
    "        self.output_layer.weight = nn.parameter.Parameter(self.output_init.clone())\n",
    "        self.output_layer.bias = nn.parameter.Parameter(ch.zeros(output_dim))\n",
    "        \n",
    "        # Activation functions\n",
    "        self.input_activation = nn.ReLU()\n",
    "        self.output_activation = nn.Softmax()\n",
    "\n",
    "    def forward(self, x):\n",
    "        #x = einops.rearrange(x, \"b c u i -> b c (u i)\")\n",
    "        for layer in self.layers:\n",
    "            x = self.input_activation(layer(x))\n",
    "        x = self.output_layer(x)\n",
    "        x = self.output_activation(x)\n",
    "        #x = einops.rearrange(x, \"x 1 n -> x n\")\n",
    "        return x.float()\n",
    "    \n",
    "    def forward_to(self, x, n_layer):\n",
    "        #x = einops.rearrange(x, \"b c u i -> b c (u i)\")\n",
    "        for layer in self.layers[:n_layer]:\n",
    "            x = self.input_activation(layer(x))\n",
    "        x = self.layers[n_layer](x)\n",
    "        return x\n",
    "\n",
    "    def get_dist_to_init(self):\n",
    "        dists = []\n",
    "        for layer, init in zip(self.layers, self.initial_weights):\n",
    "            layer_dif = ch.norm(layer.weight.detach().cpu() - init.detach().cpu()).detach().item()\n",
    "            dists.append(layer_dif)\n",
    "        layer_2_dif = ch.norm(self.output_layer.weight.detach().cpu() - self.output_init.detach().cpu()).detach().item()\n",
    "        dists.append(layer_2_dif)\n",
    "        return dists"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Eval Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "@ch.no_grad\n",
    "def evaluate(model, dloader, loss_fn=None):\n",
    "  \n",
    "  model.eval()\n",
    "\n",
    "  acc = 0\n",
    "  loss = 0\n",
    "  nsamples = 0\n",
    "  nbatch = 0\n",
    "  \n",
    "  for inputs, targets in dloader:\n",
    "      \n",
    "      inputs = inputs.cuda()\n",
    "      targets = targets.cuda()\n",
    "      outputs = model(inputs)\n",
    "\n",
    "      if loss_fn is not None:\n",
    "        loss += loss_fn(outputs, targets).item()\n",
    "        nbatch += 1\n",
    "              \n",
    "      acc += ch.sum(targets == outputs.argmax(dim=-1)).cpu()\n",
    "      nsamples += outputs.shape[0]\n",
    "\n",
    "  model.train()\n",
    "\n",
    "  return acc/nsamples, loss/nbatch\n",
    "\n",
    "@ch.no_grad\n",
    "def evaluate_local_data_rank(model, dloader, layer_number, n_batch):\n",
    "  \n",
    "  model.eval()\n",
    "  \n",
    "  ranks_all = []\n",
    "  tvs_all = []\n",
    "  for inputs, _ in dloader:\n",
    "     #ims = ch.autograd.Variable(ims.cuda(), requires_grad=True)\n",
    "     #jacobians_batch = jacfwd(lambda x: model.forward_to(x, layer_number).sum(0))(ims.cuda())\n",
    "     func = jacfwd(lambda x: model.forward_to(x, layer_number))\n",
    "     vmaped_func = vmap(func, in_dims = 0, out_dims = 0)\n",
    "\n",
    "     jacobians_batch = vmaped_func(inputs.cuda())\n",
    "\n",
    "     #print(jacobians_batch.shape)   \n",
    "     #jacobians_batch = einops.rearrange(jacobians_batch, 'b 1 1 o 1 x y -> b o (x y)')\n",
    "\n",
    "     rank_batch = ch.linalg.matrix_rank(jacobians_batch, atol = .01)\n",
    "     tv_batch = ch.linalg.norm(jacobians_batch, dim = (1,2))\n",
    "\n",
    "     tvs_all.append(tv_batch)\n",
    "     ranks_all.append(rank_batch)\n",
    "\n",
    "     if len(ranks_all) > n_batch:\n",
    "       break\n",
    "     \n",
    "  ranks_all = ch.cat(ranks_all, dim=0).to(ch.float)\n",
    "  tvs_all = ch.cat(tvs_all, dim=0).to(ch.float)\n",
    "  rank_average = ranks_all.mean().item()\n",
    "  tvs_all = tvs_all.mean().item()\n",
    "  model.train()\n",
    "  \n",
    "  return rank_average, tvs_all\n",
    "\n",
    "@ch.no_grad\n",
    "def compute_local_complexity(model, dloader, config):\n",
    "  def add_noise_to_bias(model, sigma):\n",
    "    \"\"\"\n",
    "    Clones the model and adds Gaussian noise with variance sigma to all the bias terms.\n",
    "    \"\"\"\n",
    "\n",
    "    model_clone = MLP(input_dim = config.input_dim, hidden_dim = config.hidden_dim, output_dim = config.output_dim, n_layers=config.n_layers, input_std_init=config.input_weights_init_scale, output_std_init=config.output_weights_init_scale)\n",
    "\n",
    "    model_clone.load_state_dict(model.state_dict())\n",
    "\n",
    "    for layer in model_clone.layers:\n",
    "      if isinstance(layer, nn.Linear):\n",
    "        noise = ch.randn_like(layer.bias) * sigma\n",
    "        layer.bias.data += noise\n",
    "    return model_clone\n",
    "  \n",
    "  def normal_distribution_pdf(x, sigma):\n",
    "    return ch.exp(-((x) ** 2) / (2 * sigma ** 2))\n",
    "\n",
    "  def evaluate_layer_jac(model, l, x_in, sigma):\n",
    "    \"\"\"\n",
    "    Computes summand of local compelxity estimator for a given layer l\n",
    "    \"\"\"\n",
    "    #ims = ch.autograd.Variable(ims.cuda(), requires_grad=True)\n",
    "    #jacobians_batch = jacfwd(lambda x: model.forward_to(x, layer_number).sum(0))(ims.cuda())\n",
    "    #ims = rearrange(ims, 'b h w -> b (h w)')\n",
    "    func = jacfwd(lambda x: model.forward_to(x, l))\n",
    "    vmaped_func = vmap(func, in_dims = 0, out_dims = 0)\n",
    "\n",
    "    jacobians_batch = vmaped_func(x_in)\n",
    "    dist_from_bias = model.forward_to(x_in, l) - model.layers[l].bias\n",
    "    bias_term = normal_distribution_pdf(dist_from_bias, sigma = sigma)\n",
    "\n",
    "    neuron_norms_sum = []\n",
    "\n",
    "    for i in range(jacobians_batch.shape[1]): #iter over neurons\n",
    "      neuron_norms = jacobians_batch[:,i].norm(dim=1)\n",
    "      summand = bias_term[:,i] * neuron_norms\n",
    "      neuron_norms_sum.append(summand)\n",
    "\n",
    "    neuron_norms_sum  = ch.stack(neuron_norms_sum)\n",
    "\n",
    "    return neuron_norms_sum.sum(axis = 0)\n",
    "   \n",
    "  # Done with Helpers\n",
    "  running_mean = []\n",
    "  for inputs, _ in dloader:\n",
    "    to_stack_2 = []\n",
    "    x = inputs.cuda()\n",
    "\n",
    "    # Iterate over choices for biases\n",
    "    for _ in range(config.n_iters_LC):\n",
    "      to_stack = []\n",
    "\n",
    "      model_copy = add_noise_to_bias(model, config.sigma).to('cuda')\n",
    "\n",
    "      # Iterate over layers\n",
    "      for l in range(config.n_layers):\n",
    "        neuron_norm_sum_per_x = evaluate_layer_jac(model_copy, l, x, config.sigma)\n",
    "        to_stack.append(neuron_norm_sum_per_x)\n",
    "\n",
    "      del model_copy\n",
    "\n",
    "      to_stack = ch.stack(to_stack)\n",
    "      sum_x = to_stack.sum(axis = 0)\n",
    "      to_stack_2.append(sum_x)\n",
    "\n",
    "    to_stack_2 = ch.stack(to_stack_2)\n",
    "\n",
    "    to_stack_2 = to_stack_2.sum(axis = (0,1)) / (config.n_iters_LC * len(inputs))\n",
    "    running_mean.append(to_stack_2.item())\n",
    "\n",
    "  running_mean = ch.tensor(running_mean).mean().item()\n",
    "  return running_mean\n",
    "\n",
    "def evaluate_adv(model, dloader, config):\n",
    "\n",
    "  atk = PGD(model,\n",
    "          eps=config.atk_eps,\n",
    "          alpha=config.atk_alpha,\n",
    "          steps=config.atk_itrs,\n",
    "          dmin=config.dmin,\n",
    "          dmax=config.dmax\n",
    "          )\n",
    "\n",
    "  acc = 0\n",
    "  nsamples = 0\n",
    "  for inputs, targets in tqdm(dloader, desc=f\"Computing robust acc for eps:{config.atk_eps:.3f}\"):\n",
    "\n",
    "    inputs = inputs.cuda()\n",
    "    targets = targets.cuda()\n",
    "\n",
    "    adv_images = atk(inputs, targets)\n",
    "\n",
    "    with ch.no_grad():\n",
    "        adv_pred = model(adv_images).argmax(dim=-1)\n",
    "\n",
    "    acc += ch.sum(targets == adv_pred).cpu()\n",
    "    nsamples += adv_pred.shape[0]\n",
    "\n",
    "  return acc/nsamples"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Training Loop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(model, loaders, config):\n",
    "    print('Training....')\n",
    "    print(f'Logging at steps: {config.log_steps}')\n",
    "\n",
    "    model.cuda()\n",
    "\n",
    "    # No Weight Decay on Biases\n",
    "    decay = dict()\n",
    "    no_decay = dict()\n",
    "    for name, param in model.named_parameters():\n",
    "        print('checking {}'.format(name))\n",
    "        if 'weight' in name:\n",
    "            decay[name] = param\n",
    "        else:\n",
    "            no_decay[name] = param\n",
    "        \n",
    "    print(f\"Weight Decay on: {decay.keys()}\")\n",
    "    print(f\"Weight Decay off: {no_decay.keys()}\")\n",
    "\n",
    "    # Optimizer Setup\n",
    "    \n",
    "    if config.optimizer == 'sgd':\n",
    "        print('Using SGD optimizer')\n",
    "        if not config.bias_decay:\n",
    "            opt = SGD([\n",
    "                {'params': no_decay.values(), 'weight_decay': 0.0},\n",
    "                {'params': decay.values(), 'weight_decay': config.weight_decay}\n",
    "            ],\n",
    "            lr=config.lr,\n",
    "            momentum=config.momentum)\n",
    "        else:\n",
    "            opt = SGD(model.parameters(),\n",
    "                  lr=config.lr,\n",
    "                  momentum=config.momentum,\n",
    "                  weight_decay=config.weight_decay)\n",
    "\n",
    "    elif config.optimizer == 'adam':\n",
    "        if not config.bias_decay:\n",
    "            opt = AdamW([\n",
    "                {'params': no_decay.values(), 'weight_decay': 0.0},\n",
    "                {'params': decay.values(), 'weight_decay': config.weight_decay}\n",
    "            ],\n",
    "            lr=config.lr)\n",
    "        else:   \n",
    "            opt = AdamW(model.parameters(),\n",
    "                    lr=config.lr,\n",
    "                    weight_decay=config.weight_decay)\n",
    "\n",
    "    else:\n",
    "        raise NotImplementedError\n",
    "    \n",
    "    iters_per_epoch = len(loaders['train'])\n",
    "    epochs = np.floor(config.num_steps/iters_per_epoch)\n",
    "    print(f\"Training for {epochs} epochs\")\n",
    "\n",
    "    loss_fn = ch.nn.CrossEntropyLoss()\n",
    "\n",
    "    train_step = 0\n",
    "\n",
    "    while True:\n",
    "        if train_step > config.num_steps: break\n",
    "\n",
    "        for input, targets in loaders['train']:\n",
    "\n",
    "            model.train()\n",
    "\n",
    "            input = input.cuda()\n",
    "            targets = targets.cuda()\n",
    "\n",
    "            opt.zero_grad()\n",
    "            out = model(input)\n",
    "\n",
    "            loss = loss_fn(out, targets)\n",
    "            loss.backward()\n",
    "            opt.step()\n",
    "            train_step += 1\n",
    "\n",
    "            # Calculate Stats\n",
    "            print('Logging Evaluations')\n",
    "            if train_step in config.log_steps:\n",
    "                model.eval()\n",
    "\n",
    "                train_acc, train_loss = evaluate(model,\n",
    "                                                 loaders['train'],\n",
    "                                                 loss_fn)\n",
    "                test_acc, test_loss = evaluate(model,\n",
    "                                                 loaders['test'],\n",
    "                                                 loss_fn)\n",
    "                \n",
    "                l2_norm = 0\n",
    "                for name, param in model.named_parameters():\n",
    "                    l2_norm += ch.norm(param)\n",
    "\n",
    "                if config.compute_LC:\n",
    "                    print('Computing Local Complexity')\n",
    "                    train_LC = compute_local_complexity(model, loaders['train'], config)\n",
    "                    test_LC = compute_local_complexity(model, loaders['test'], config)\n",
    "                else:\n",
    "                    train_LC = 0\n",
    "                    test_LC = 0\n",
    "                    \n",
    "\n",
    "                if config.local_rank_eval:\n",
    "                    print('Computing Local Rank')\n",
    "                    local_ranks = []\n",
    "                    for layer_number in range(0, config.n_layers):\n",
    "                        local_rank, tv = evaluate_local_data_rank(model, loaders['test'], layer_number, config.n_batchs_for_local_rank)\n",
    "                        local_ranks.append(local_rank)\n",
    "\n",
    "                if config.compute_robust:\n",
    "                    print('Computing Robustness')\n",
    "                    robust_acc = evaluate_adv(model, loaders['test'], config)\n",
    "\n",
    "                stats_dict = {\n",
    "                    'iter': train_step,\n",
    "                    'train/loss': train_loss,\n",
    "                    'train/acc': train_acc,\n",
    "                    'test/loss': test_loss,\n",
    "                    'test/acc': test_acc,\n",
    "                    'adv/acc': robust_acc,\n",
    "                    'l2_norm': l2_norm,\n",
    "                    'train/LC': train_LC,\n",
    "                    'test/LC': test_LC,\n",
    "                    'TV': tv,\n",
    "                } | {f'train/local_rank_{i}': rank for i, rank in enumerate(local_ranks)}\n",
    "\n",
    "                wandb.log(stats_dict)\n",
    "\n",
    "                if config.save_model:\n",
    "                    ch.save(model.state_dict(), os.path.join(config.model_save_dir, f'model_{train_step}.pth'))\n",
    "\n",
    "                model.train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Run"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "Finishing last run (ID:9qi84gnn) before initializing another..."
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "d3bcbe437aeb4481a3b08f756b326914",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.004 MB of 0.004 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>TV</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▂▂▂▃█▃</td></tr><tr><td>adv/acc</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▂▂▄▄▄▄▅▄█▃</td></tr><tr><td>iter</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂▃▃▄▅█</td></tr><tr><td>l2_norm</td><td>███████████████████████████▇▇▇▆▆▅▄▃▂▁▁▁▁</td></tr><tr><td>test/LC</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>test/acc</td><td>▁▁▁▁▁▁▂▂▂▃▄▅▆▆▆▆▆▆▆▇▇▇▇▇▇███████████████</td></tr><tr><td>test/loss</td><td>███████▇▇▇▅▄▄▃▃▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>train/LC</td><td>▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>train/acc</td><td>▁▁▁▁▁▁▂▂▂▃▄▅▅▆▆▆▆▆▆▇▇▇▇▇▇███████████████</td></tr><tr><td>train/local_rank_0</td><td>███████████████████████████████████████▁</td></tr><tr><td>train/local_rank_1</td><td>▅▅▅▅▅▅▅▅▅▆▆▇▇▇▇███████████▇▇▇▆▆▆▇▇▆▄▆▄▃▁</td></tr><tr><td>train/local_rank_2</td><td>▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇██████████▇█▇▇▆▆▇▇▇▄▃▁▁▁</td></tr><tr><td>train/local_rank_3</td><td>▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇████████▇█▇▇▇▇▇▇▆▃▁▁▁▁</td></tr><tr><td>train/local_rank_4</td><td>▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇██████████▇█▇▇▇▇▇▇▆▂▁▁▁▁</td></tr><tr><td>train/loss</td><td>███████▇▇▇▅▄▄▃▃▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>TV</td><td>83.71045</td></tr><tr><td>adv/acc</td><td>0.147</td></tr><tr><td>iter</td><td>1000000</td></tr><tr><td>l2_norm</td><td>38.60926</td></tr><tr><td>test/LC</td><td>0</td></tr><tr><td>test/acc</td><td>0.902</td></tr><tr><td>test/loss</td><td>1.55936</td></tr><tr><td>train/LC</td><td>0</td></tr><tr><td>train/acc</td><td>1.0</td></tr><tr><td>train/local_rank_0</td><td>198.0</td></tr><tr><td>train/local_rank_1</td><td>36.00781</td></tr><tr><td>train/local_rank_2</td><td>10.24219</td></tr><tr><td>train/local_rank_3</td><td>7.95312</td></tr><tr><td>train/local_rank_4</td><td>6.96875</td></tr><tr><td>train/loss</td><td>1.46115</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">MNIST-MLP-Fri_Sep_20_14:05:07_2024</strong> at: <a href='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer/runs/9qi84gnn' target=\"_blank\">https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer/runs/9qi84gnn</a><br/> View project at: <a href='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer' target=\"_blank\">https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer</a><br/>Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20240920_140508-9qi84gnn/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Successfully finished last run (ID:9qi84gnn). Initializing new run:<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3b9dc457471a4f36883f91bb669f2bf4",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='Waiting for wandb.init()...\\r'), FloatProgress(value=0.011114381855845245, max=1.0…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.18.1 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.17.0"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/home/niket/local_complexity/wandb/run-20240921_022043-55uqmz81</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer/runs/55uqmz81' target=\"_blank\">MNIST-MLP-Sat_Sep_21_02:20:43_2024</a></strong> to <a href='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer' target=\"_blank\">https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer/runs/55uqmz81' target=\"_blank\">https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer/runs/55uqmz81</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<button onClick=\"this.nextSibling.style.display='block';this.style.display='none';\">Display W&B run</button><iframe src='https://wandb.ai/niket/LRLC_study_MNIST_MLP_OneLayer/runs/55uqmz81?jupyter=true' style='border:none;width:100%;height:420px;display:none;'></iframe>"
      ],
      "text/plain": [
       "<wandb.sdk.wandb_run.Run at 0x7f84a46ed040>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wandb_project = config.wandb_proj\n",
    "timestamp = time.ctime().replace(' ','_')\n",
    "wandb_run_name = f\"{config.wandb_pref}-{timestamp}\"\n",
    "wandb.init(project=wandb_project, name=wandb_run_name, config=config)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training....\n",
      "Logging at steps: [      1       2       3       4       5       7       9      12      16\n",
      "      22      29      39      51      68      91     120     159     212\n",
      "     281     372     494     655     868    1151    1526    2023    2682\n",
      "    3556    4714    6250    8286   10985   14563   19306   25595   33932\n",
      "   44984   59636   79060  104811  138949  184206  244205  323745  429193\n",
      "  568986  754312 1000000]\n",
      "checking layers.0.weight\n",
      "checking layers.0.bias\n",
      "checking layers.1.weight\n",
      "checking layers.1.bias\n",
      "checking layers.2.weight\n",
      "checking layers.2.bias\n",
      "checking layers.3.weight\n",
      "checking layers.3.bias\n",
      "checking layers.4.weight\n",
      "checking layers.4.bias\n",
      "checking output_layer.weight\n",
      "checking output_layer.bias\n",
      "Weight Decay on: dict_keys(['layers.0.weight', 'layers.1.weight', 'layers.2.weight', 'layers.3.weight', 'layers.4.weight', 'output_layer.weight'])\n",
      "Weight Decay off: dict_keys(['layers.0.bias', 'layers.1.bias', 'layers.2.bias', 'layers.3.bias', 'layers.4.bias', 'output_layer.bias'])\n",
      "Training for 62500.0 epochs\n",
      "Logging Evaluations\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/niket/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py:1532: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.\n",
      "  return self._call_impl(*args, **kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Computing Local Complexity\n",
      "Computing Local Rank\n",
      "Computing Robustness\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing robust acc for eps:0.314: 100%|██████████| 16/16 [00:00<00:00, 56.99it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logging Evaluations\n",
      "Computing Local Complexity\n",
      "Computing Local Rank\n",
      "Computing Robustness\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing robust acc for eps:0.314: 100%|██████████| 16/16 [00:00<00:00, 58.46it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logging Evaluations\n",
      "Computing Local Complexity\n",
      "Computing Local Rank\n",
      "Computing Robustness\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing robust acc for eps:0.314: 100%|██████████| 16/16 [00:00<00:00, 62.64it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logging Evaluations\n",
      "Computing Local Complexity\n",
      "Computing Local Rank\n",
      "Computing Robustness\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing robust acc for eps:0.314: 100%|██████████| 16/16 [00:00<00:00, 65.77it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logging Evaluations\n",
      "Computing Local Complexity\n",
      "Computing Local Rank\n",
      "Computing Robustness\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing robust acc for eps:0.314: 100%|██████████| 16/16 [00:00<00:00, 67.52it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logging Evaluations\n",
      "Logging Evaluations\n",
      "Computing Local Complexity\n",
      "Computing Local Rank\n",
      "Computing Robustness\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Computing robust acc for eps:0.314: 100%|██████████| 16/16 [00:00<00:00, 41.84it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [...
"""

# We'll use a regex to pull out every cell's type and source block from the snippet.
cell_pattern = re.compile(
    r'"cell_type"\s*:\s*"(?P<ctype>[^"]+)"[\s\S]*?"source"\s*:\s*\[(?P<source>[\s\S]*?)\]\s*',
    re.MULTILINE,
)

cells = []
for match in cell_pattern.finditer(corrupted_snippet):
    ctype = match.group('ctype')
    source_block = match.group('source')
    # Extract each line inside quotes
    lines = re.findall(r'"(.*?)"', source_block, re.DOTALL)
    source_text = '\n'.join(lines)
    if ctype == "code":
        cell = nbformat.v4.new_code_cell(source=source_text, execution_count=None, outputs=[])
    else:
        cell = nbformat.v4.new_markdown_cell(source=source_text)
    cells.append(cell)

# Build a new notebook object
nb = nbformat.v4.new_notebook(cells=cells, metadata={})

# Save the fixed notebook
output_path = "fixed_mnist.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

output_path
