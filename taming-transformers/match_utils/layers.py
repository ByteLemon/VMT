#from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
import torch.hub
import time
import pickle
import math

from match_utils import matching, stats, proggan, nethook, dataset, loading, plotting


def get_layers(gan, gan_layers, discr,discr_layers, img, args):
    '''Get a dictionary of the layer dimensions for the GAN and discriminative model.'''
    
   
    #### Forward through GAN and discriminator
    with torch.no_grad():
        out_gan = gan(img)
        if args.model_mode=='clip':
            out_sr = discr.model.encode_image(img)
        else:
            out_sr = discr(img)
        
        
    #### append GAN layer activations for batch
    gan_activs = []
    for layer in gan_layers:
        gan_activation = gan.retained_layer(layer, clear = True).detach()
        gan_activs.append(gan_activation)
        
    #### append discr layer activations for batch
    discr_activs = []
    for layer in discr_layers:
        discr_activation = discr.retained_layer(layer, clear = True).detach()
        discr_activs.append(discr_activation)
        

    #create dict of layers
    all_gan_layers = {}
    for iii, gan_activ in enumerate(gan_activs):
        all_gan_layers[gan_layers[iii]] = gan_activ.shape[1]
        
    all_discr_layers = {}
    for jjj, discr_activ in enumerate(discr_activs):
        all_discr_layers[discr_layers[jjj]] = discr_activ.shape[1]
            
    return all_gan_layers, all_discr_layers

def find_act(act_num, net_dict):
    '''Turn raw unit number into (layer, unit) tuple).'''
    
    layers_list = list(net_dict)
    
    layer = 0
    counter =0
    
    while act_num >= counter:
        layer +=1
        counter += net_dict[layers_list[layer-1]]
        
        
    act = act_num-counter+net_dict[layers_list[layer-1]]
    
    del layers_list
    torch.cuda.empty_cache()
    return (layer-1), act
        
