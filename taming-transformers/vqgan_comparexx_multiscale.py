import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import utils as vutils
from taming.data.augment_custom import load_data
import matplotlib.pyplot as plt
from match_utils import nethook, stats, helpers, loading, visualize_pairwisematch, layers, mae, clip
from omegaconf import OmegaConf
import yaml
# from taming.models.vqgan_top import VQModel
from taming.models.vqgan_multi_scale_load_top_scale import VQModel
from utils import *
import timm
CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None):
    model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

class CompareVQGAN:
    def __init__(self, args):
        # self.vqgan = VQGAN(args).to(device=args.device)
        # self.vqgan.load_checkpoint(args.checkpoint_path)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
       
        config1024 = load_config(args.yaml_path, display=False)
        self.vqgan = load_vqgan(config1024).to(device=args.device)
        sd = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
        self.vqgan.load_state_dict(sd, strict=False)
        self.vqgan = self.vqgan.eval()
        
        self.vqgan_net = self.vqgan
        self.vqgan_layers = []
        print(self.vqgan)
        for name, layer in self.vqgan.named_modules():
            # if "quant_conv" in name and 'post_quant_conv' not in name:
            if ("top.encode_out.1" in name) or ("mid.encode_out.1" in name) or ("bot.encode_out.1" in name):
                self.vqgan_layers.append(name) 
        print('self.vqgan_layers:',self.vqgan_layers)
        
        if args.model_mode in ['resnet50', 'resnet34','resnet18','densenet169','densenet161','mobilenet_v2_050']:
            if args.model_mode == 'resnet50':
                self.source_model = resnet = models.resnet50(pretrained=True).to(device=args.device) 
            elif args.model_mode == 'resnet34':
                self.source_model = resnet = models.resnet34(pretrained=True).to(device=args.device) 
            elif args.model_mode == 'resnet18':
                self.source_model = resnet = models.resnet18(pretrained=True).to(device=args.device) 
            elif args.model_mode == 'densenet169':
                self.source_model = resnet = models.densenet169(pretrained=True).to(device=args.device) 
            elif args.model_mode == 'densenet161':
                self.source_model = resnet = models.densenet161(pretrained=True).to(device=args.device) 
            elif args.model_mode == 'mobilenet_v2_050':
                # self.source_model = resnet = models.mobilenet_v2(pretrained=True).to(device=args.device) 
                self.source_model = timm.create_model('mobilenetv2_050.lamb_in1k', pretrained=True).to(device=args.device) 
                
            self.source_model = self.source_model.eval()
            # self.source_layers = [ "layer1", "layer2", "layer3", "layer4"]
            self.source_layers = []
            output = open(os.path.join(args.save_path, 'layername.txt'), 'w')
            layer_idx = 0
            for name, layer in self.source_model.named_modules():
                if ("conv" in name) or ("downsample" in name) or ("maxpool" in name): #去掉bn和relu层
                    self.source_layers.append(name)
                    output.write('Layer-'+str(layer_idx) + ' ' + name + '\n')
                    layer_idx += 1
            output.close()
           
        elif args.model_mode == 'mae': #input size 224
            self.source_model = mae.load_mae(os.path.join(args.pretrained_dir, 'mae_pretrain_vit_base.pth')).to(device=args.device)
            self.source_layers = []
            output = open(os.path.join(args.save_path, 'layername.txt'), 'w')
            # layer_idx = 0
            # for name, layer in self.source_model.named_modules():
            #     if "mlp.act" in name:
            #         self.source_layers.append(name)
            #         output.write('Layer-'+str(layer_idx) + ' ' + name + '\n')
            #         layer_idx += 1
            self.source_layers = []
            for i in range(12):
                self.source_layers.append(f"blocks.{i}.norm1")
                self.source_layers.append(f"blocks.{i}.attn.qkv")
                self.source_layers.append(f"blocks.{i}.attn.proj")
                self.source_layers.append(f"blocks.{i}.norm2")
                self.source_layers.append(f"blocks.{i}.mlp.fc2")
        elif args.model_mode == "clip":
            self.source_model, _ = clip.load("RN50", device=args.device, jit = False)
            for p in self.source_model.parameters(): 
                p.data = p.data.float()
            # self.source_layers = [ "visual.layer1", "visual.layer2", "visual.layer3", "visual.layer4"]
            self.source_layers = []
            output = open(os.path.join(args.save_path, 'layername.txt'), 'w')
            layer_idx = 0
            for name, layer in self.source_model.named_modules():
                if ("visual" in name) and (("conv" in name) or ("downsample" in name) or ("maxpool" in name)): #去掉bn和relu层
                    self.source_layers.append(name)
                    output.write('Layer-'+str(layer_idx) + ' ' + name + '\n')
                    layer_idx += 1
            output.close()
            
        elif args.model_mode in ['dino_vitb16', 'dino_vitb8']:
            self.source_model = torch.hub.load('facebookresearch/dino:main', args.model_mode).to(args.device)
            for p in self.source_model.parameters(): 
                p.data = p.data.float() 
            # self.source_layers = []
            # for name, layer in self.source_model.named_modules():
            #     # if  "mlp.act" in name:
            #     if  ("mlp" in name) and ("drop" not in name) and (not name.endswith("mlp")):
            #         self.source_layers.append(name)
            # self.source_layers = [f"blocks.{i}" for i in range(12)]
            self.source_layers = []
            for i in range(12):
                self.source_layers.append(f"blocks.{i}.norm1")
                self.source_layers.append(f"blocks.{i}.attn.qkv")
                self.source_layers.append(f"blocks.{i}.attn.proj")
                self.source_layers.append(f"blocks.{i}.norm2")
                self.source_layers.append(f"blocks.{i}.mlp.fc2")

        elif args.model_mode == "dino":
            self.source_model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').to(args.device)
            for p in self.source_model.parameters(): 
                p.data = p.data.float() 
            # self.source_layers = [ "layer1", "layer2", "layer3", "layer4"]
            self.source_layers = []
            output = open(os.path.join(args.save_path, 'layername.txt'), 'w')
            layer_idx = 0
            for name, layer in self.source_model.named_modules():
                if ("conv" in name) or ("downsample" in name) or ("maxpool" in name): #去掉bn和relu层
                    self.source_layers.append(name)
                    output.write('Layer-'+str(layer_idx) + ' ' + name + '\n')
                    layer_idx += 1
            output.close()
        
            
        print('self.source_model:',self.source_model)
        print('self.source_layers:',self.source_layers)
        
        #### hook layers for source pretrained model
        self.source_model = nethook.InstrumentedModel(self.source_model)
        self.source_model.retain_layers(self.source_layers)

        #### hook layers for vqgan
        self.vqgan = nethook.InstrumentedModel(self.vqgan)
        self.vqgan.retain_layers(self.vqgan_layers)
        
        
        # self.compare(args)
        # self.visualize(args)
        self.reconstruction(args)
    
    def store_activs(self, model, layernames):
        '''Store the activations in a list.'''
        activs = []
        for layer in layernames:
            activation = model.retained_layer(layer, clear = True)
            activs.append(activation)
            
        return activs
    
    def dict_layers(self, activs):
        '''Return dictionary of layer sizes.'''
        all_layers = {}
        for iii, activ in enumerate(activs):
            all_layers[activs[iii]] = activ.shape[1]
        return all_layers
    
    def normalize(self, activation, stats_table):
        '''Normalize activations based on statistic from dataset.'''
        eps = 0.00001
        norm_input = (activation- stats_table[0])/(stats_table[1]+eps)
        
        return norm_input
    
    def create_final_table(self, all_match_table, model1_dict, model2_dict, device ):
        num_activs1 = sum(model1_dict.values())
        num_activs2 = sum(model2_dict.values())
        final_match_table = torch.zeros((num_activs1, num_activs2)).to(device)
        
        model1_activ_count = 0 
        for ii in range(len(all_match_table)):
            model2_activ_count = 0
            for jj in range(len(all_match_table[ii])):
                num_model1activs = all_match_table[ii][0].shape[0]
                num_model2activs = all_match_table[0][jj].shape[1]
                final_match_table[model1_activ_count: model1_activ_count+num_model1activs, \
                                model2_activ_count:model2_activ_count+num_model2activs] = all_match_table[ii][jj]
                model2_activ_count += num_model2activs
            model1_activ_count += num_model1activs
        return final_match_table

    def compare(self, args):
        train_dataset = load_data(args)
        
        #get dataset stats  遍历整个数据集
        # num = len(train_dataset)
        num = 500
        if not os.path.exists(os.path.join(args.save_path, "gan_stats.pkl")):
            gan_stats_table, discr_stats_table = stats.get_mean_std(self.vqgan, self.vqgan_layers, self.source_model, self.source_layers, train_dataset, args.device, num, args)
            helpers.save_array(gan_stats_table, os.path.join(args.save_path, "gan_stats.pkl"))
            helpers.save_array(discr_stats_table, os.path.join(args.save_path, "discr_stats.pkl"))
        else:
            gan_stats_table, discr_stats_table = loading.load_stats(args.save_path, args.device)
        
        print("Done")
        print("Starting Activation Matching")
        with torch.no_grad():
            with tqdm(range(num)) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    #### append GAN layer activations for batch
                    out_gan = self.vqgan(imgs)
                    gan_activs = self.store_activs(self.vqgan, self.vqgan_layers)
                    
                    if args.model_mode=='clip':
                        out_sr = self.source_model.model.encode_image(imgs)
                    else:
                        out_sr = self.source_model(imgs)
                    discr_activs =  self.store_activs(self.source_model, self.source_layers)
                    
                    #create dict of layers with number of activations
                    all_gan_layers = self.dict_layers(gan_activs) #[1,256,16,16]
                    all_discr_layers = self.dict_layers(discr_activs)#[1,256,64,64],[1,512,32,32],[1,1024,16,16],[1,2048,8,8]
                    
                    if i == 0:
                        num_gan_activs = sum(all_gan_layers.values())
                        num_discr_activs = sum(all_discr_layers.values())
                        final_match_table = torch.zeros((num_gan_activs, num_discr_activs)).to(args.device) #[256,3840]
                        
                    ##### Matching
                    all_match_table = []
                    print('len(gan_activs):',len(gan_activs))
                    for ii, gan_activ in enumerate(gan_activs):
                        match_table = []
                        print('gan_activ.shape:', gan_activ.shape)
                        gan_activ = self.normalize(gan_activ, gan_stats_table[ii])
                        gan_activ_shape = gan_activ.shape

                        for jj, discr_activ in enumerate(discr_activs):
                            discr_activ_new = self.normalize(discr_activ, discr_stats_table[jj]) 
                            #### Get maps to same size
                            map_size = max((gan_activ_shape[2], discr_activ.shape[2]))
                            gan_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_activ)
                            discr_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(discr_activ_new)            
                            #Pearson Correlation 
                            print('gan_activ_new.shape,discr_activ_new.shape:',gan_activ_new.shape,discr_activ_new.shape)
                            prod = torch.einsum('aixy,ajxy->ij', gan_activ_new,discr_activ_new) 
                            div1 = torch.einsum('aixy->i', gan_activ_new**2)
                            div2 = torch.einsum('ajxy->j', discr_activ_new**2)
                            div = torch.einsum('i,j->ij', div1,div2)
                            scores = prod/torch.sqrt(div)
                            nans = torch.isnan(scores)
                            scores[nans] = 0
                            scores = scores.cpu()
                            
                            match_table.append(scores)
                            del gan_activ_new
                            del discr_activ_new
                            del scores

                        all_match_table.append(match_table)
                        del match_table

                    ##create table
                    batch_match_table = self.create_final_table(all_match_table, all_gan_layers, all_discr_layers, args.device)
                    final_match_table += batch_match_table #[256,3840]
                    print('final_match_table.shape:',final_match_table.shape)
                
                    del all_match_table
                    del batch_match_table
                    del gan_activs
                    del discr_activs
                    torch.cuda.empty_cache()
            
                    # source_features = self.source_model(imgs)
                    # source_features = torch.nn.MaxPool2d(kernel_size=4, stride=4)(source_features)
                    
                    # vqgan_feature = self.vqgan.encoder(imgs)
                    # vqgan_feature = self.vqgan.quant_conv(vqgan_feature) # target feature
                    
                    # codebook_mapping, codebook_indices, q_loss = self.vqgan.codebook(vqgan_feature)
                    # # codebook_mapping, codebook_indices, q_loss = self.vqgan.codebook(matching_features)
                    # post_quant_conv_mapping = self.vqgan.post_quant_conv(codebook_mapping)
                    # decoded_images = self.vqgan.decoder(post_quant_conv_mapping)

                    # if i % 1 == 0:
                    #     with torch.no_grad():
                    #         # real_fake_images = torch.cat((imgs[:1], decoded_images.add(1).mul(0.5)[:1]))
                    #         real_fake_images = torch.cat((imgs[:1], decoded_images[:1]))
                    #         vutils.save_image(real_fake_images, os.path.join(args.save_path, f"{i}.jpg"), nrow=4)

                    if i==num-1:
                        break
                
                #average and save
                final_match_table /= num
                helpers.save_array(final_match_table, os.path.join(args.save_path, "table.pkl"))
                
    def visualize(self, args):
        table, gan_stats, discr_stats = loading.load_stats_table(args.save_path, args.device)
        
        # best buddies
        match_scores,_ = torch.max(table,1)
        
        n = 5
        _,gan_matches1 = torch.topk(table,k=1,dim=1)
        _,discr_matches1 = torch.topk(table,k=n, dim=0)
        
        gan_matches = torch.argmax(table,1)
        dino_matches = torch.argmax(table,0)
        
        perfect_matches = []
        perfect_match_scores1= []
        discr_perfect_matches1 = []

        num_kmatches = 0 
        for i in range(table.shape[0]):
            gan_match = gan_matches1[i].item()
            discr_matches = discr_matches1[:, gan_match]
            
            for unit in discr_matches:
                if unit == i: #“best-buddies” pairs: pairs that are mutual nearest neighbors
                    num_kmatches += 1
                    perfect_matches.append(i)
                    discr_perfect_matches1.append(gan_match)
                    perfect_match_scores1.append(table[i, gan_match])
                    break
        print('len(perfect_matches):',len(perfect_matches)) # 51

        # sort units according to scores
        gan_match_units = [match for _,match in sorted(zip(perfect_match_scores1, perfect_matches), reverse = True)] #降序
        scores = sorted(perfect_match_scores1, reverse = True)
        scores = [score.item() for score in scores]
        
        #visualize matches over sample images
        train_dataset = load_data(args)
        num = 5
        with torch.no_grad():
            with tqdm(range(num)) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    # if not (i==29 or i==11 or i==4):
                    #     continue
                    imgs = imgs.to(device=args.device)
                    imgs_vis = imgs.detach().cpu()
                    imgs_vis = torch.permute(imgs_vis[0], (1,2,0))
                    plt.imshow(imgs_vis)
                    plt.axis('off')
                    plt.savefig(os.path.join(args.save_path, f"{i}_img.png"))
                    
                    gan_match_units.sort()
                    save_path_i = os.path.join(args.save_path, f"{i}_vis.png")
                    ganlayers, discrlayers = layers.get_layers(self.vqgan, self.vqgan_layers, self.source_model, self.source_layers, imgs, args)
                    visualize_pairwisematch.viz_matches(table, self.vqgan, self.source_model, imgs, ganlayers, discrlayers, gan_stats, discr_stats, gan_match_units, scores, save_path_i, args )
                    if i==num-1:
                        break
        
                
    def reconstruction(self, args):
        train_dataset = load_data(args)
        table, gan_stats, discr_stats = loading.load_stats_table(args.save_path, args.device)
        print('len(gan_stats):',len(gan_stats)) # 1
        print('len(discr_stats):',len(discr_stats)) # 4
        n = 5
        _,gan_matches1 = torch.topk(table,k=1,dim=1)
        _,discr_matches1 = torch.topk(table,k=n, dim=0)
        match_scores,_ = torch.max(table,1)
        # sort units according to scores
        gan_match_units = [match for _,match in sorted(zip(match_scores, gan_matches1), reverse = True)] #降序
        scores = sorted(match_scores, reverse = True)
        scores = [score.item() for score in scores]
        
        num = 5
        with torch.no_grad():
            with tqdm(range(num)) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    # if not (i==29 or i==11 or i==4):
                    #     continue

                    imgs = imgs.to(device=args.device)
                    out_gan = self.vqgan(imgs)
                    gan_activs = self.store_activs(self.vqgan, self.vqgan_layers)
                    
                    ganlayers, discrlayers = layers.get_layers(self.vqgan, self.vqgan_layers, self.source_model, self.source_layers, imgs, args)
                    print('ganlayers:',ganlayers) #'quant_conv':256, 256, 256
                    print('discrlayers:',discrlayers) #{'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
                    print('table.shape:',table.shape) #[768,34304]
                    
                    matching_scores_list = []
                    matching_features_list = []
                    for idx in range(3):
                        vqgan_feature = gan_activs[idx] #[1,256,28,28],[1,256,56,56],[1,256,112,112]
                        print('idx, vqgan_feature.shape:', idx, vqgan_feature.shape)
                        
                        matching_features = torch.zeros_like(vqgan_feature)
                        
                        for unit_ in range(256):
                            unit = unit_+idx*256
                            
                            if args.model_mode=='clip':
                                out_sr = self.source_model.model.encode_image(imgs)
                            else:
                                out_sr = self.source_model(imgs)
                            match1 = table[unit] #[1,3840]
                            scores, flat_indices = torch.sort(match1, descending = True)
                            print('unit:',unit)
                            assert unit>=0 and unit<=len(gan_match_units)
                            ganidx = layers.find_act(unit, ganlayers) # 0, [228]
                            print('ganidx:', ganidx)
                            gan_act = vqgan_feature
                            gan_act = self.normalize(gan_act, gan_stats[ganidx[0]])
                            
                            ### through discriminator
                            print('flat_indices[0]:',flat_indices[0]) #13
                            discridx = layers.find_act(flat_indices[0], discrlayers)
                            discr_act = self.source_model.retained_layer(list(discrlayers)[discridx[0]], clear = True)
                            discr_act = self.normalize(discr_act, discr_stats[discridx[0]])
                            # print('discridx:',discridx) # 0, [13]
                            # print('discr_act.shape:', discr_act.shape) #[1,256,64,64]
                            
                            ##### resize 
                            map_size = map_size = gan_act.shape[2]
                            discr_act = torch.nn.functional.interpolate(discr_act,size=(map_size,map_size), mode='bilinear')
                            print('discr_act.shape:',discr_act.shape) 
                            #direct matching 
                            matching_features[:, ganidx[1], :, :] = discr_act[:, discridx[1], :, :].float() 
                            # unit_match_score = scores[0]
                            # print('unit_match_score:',unit_match_score.item())
                            # matching_scores_list.append(unit_match_score.item())
                            
                            #preserve best buddies pair
                            gan_match = gan_matches1[unit].item()
                            discr_matches = discr_matches1[:, gan_match]
                            mask = torch.zeros_like(gan_act[:, ganidx[1], :, :])
                            for discr_unit in discr_matches:
                                if discr_unit == unit:
                                    unit_match_score = scores[0]
                                    print('unit_match_score:',unit_match_score.item())
                                    matching_scores_list.append(unit_match_score.item())
                                    mask = torch.ones_like(gan_act[:, ganidx[1], :, :])
                            # # matching_features[:, ganidx[1], :, :] = discr_act[:, discridx[1], :, :].float()*mask    
                            # # matching_features[:, ganidx[1], :, :] = gan_act[:, ganidx[1], :, :].float()*mask

                            
                            
                            # if unit_match_score>0.5:
                            #     mask = torch.ones_like(gan_act[:, ganidx[1], :, :])
                            # else:
                            #     mask = torch.zeros_like(gan_act[:, ganidx[1], :, :])
                            # mask = unit_match_score*torch.ones_like(gan_act[:, ganidx[1], :, :])
                            # matching_features[:, ganidx[1], :, :] = discr_act[:, discridx[1], :, :].float()*mask
                            # # matching_features[:, ganidx[1], :, :] = gan_act[:, ganidx[1], :, :].float()*mask
                            

                        matching_features_list.append(matching_features)
                    mean_matching_score = np.mean(matching_scores_list)
                    print("mean_matching_score",mean_matching_score)
                    
                    def nonlinearity(x):
                        # swish
                        return x*torch.sigmoid(x)
                    print(matching_features_list[0].shape)
                    print(matching_features_list[1].shape)
                    print(matching_features_list[2].shape)
                    x_q_1, emb_loss1, _ = self.vqgan_net.endecoder.top.quantize(matching_features_list[0]) #top
                    x_q_2, emb_loss2, _ = self.vqgan_net.endecoder.top.quantize(matching_features_list[1]) #mid
                    x_q_3, emb_loss3, _ = self.vqgan_net.endecoder.top.quantize(matching_features_list[2]) #bot

                    x_mid_de = self.vqgan_net.endecoder.mid.decode_convin(x_q_2)
                    x_mid_de = self.vqgan_net.endecoder.mid.decode_0(x_mid_de)
                    x_mid_de = self.vqgan_net.endecoder.mid.decode_1(x_mid_de)
                    
                    x_bot_de = self.vqgan_net.endecoder.bot.decode_convin(x_q_3)
                    x_bot_de = self.vqgan_net.endecoder.bot.decode_0(x_bot_de)
                    x_bot_de = self.vqgan_net.endecoder.bot.decode_1(x_bot_de)
                    x_bot_de = self.vqgan_net.endecoder.bot.decode_2(x_bot_de)

                    x_top_de = self.vqgan_net.endecoder.top.decode_convin(x_q_1)
                    x_top_de = self.vqgan_net.endecoder.top.decode_atten_1(x_top_de)
                    x_top_de = self.vqgan_net.endecoder.top.decode_atten_2(x_top_de)
                    x_top_de = self.vqgan_net.endecoder.top.decode_0(x_top_de)
                    x_top_de = self.vqgan_net.endecoder.top.decode_1(x_top_de)

                    x_top_de = x_top_de + x_mid_de
                    x_top_de = self.vqgan_net.endecoder.top.decode_2(x_top_de)
                    x_top_de = x_top_de + x_bot_de
                    x_top_de = self.vqgan_net.endecoder.top.decode_3(x_top_de)
                    x_top_de = self.vqgan_net.endecoder.top.decode_4(x_top_de)

                    out = self.vqgan_net.endecoder.norm_out(x_top_de)
                    x = nonlinearity(out)
                    decoded_images = self.vqgan_net.endecoder.conv_out(out)

                    if i % 1 == 0:
                        with torch.no_grad():
                            # real_fake_images = torch.cat((imgs[:1], decoded_images.add(1).mul(0.5)[:1]))
                            real_fake_images = torch.cat((imgs[:1], decoded_images[:1]))
                            vutils.save_image(real_fake_images, os.path.join(args.save_path, f"{i}_recon_match_score_{mean_matching_score}.jpg"), nrow=4)

                    if i==num-1:
                        break
                    
        
        
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=224, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=1, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc-factor', type=float, default=1., help='')
    parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)

    args = parser.parse_args()
    
    args.pretrained_dir = "/home/xjiangbh/ModelZoo_work/rosetta_neurons/checkpoints/"
    args.txt_file = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/data/train.txt"
    
    # Flower dataset
    args.dataset_path = "/home/xjiangbh/ModelZoo_work/Data/flowers/"
    # single-scale
    # args.checkpoint_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/logs/2024-06-17T13-57-16_single_scale/checkpoints/last.ckpt"
    # args.yaml_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/configs/top_scale_pretrain.yaml"
    # multiple-scale
    args.checkpoint_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/logs/2024-06-20T12-39-39_multi_scale/checkpoints/last.ckpt"
    args.yaml_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/configs/load_top_mul_scale.yaml"
    
    # ISIC2019 dataset
    # args.dataset_path = "/home/xjiangbh/ModelZoo_work/Data/ISIC2019/ISIC_2019_Training_Input/"
    # args.txt_file = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/data/train_ISIC2019.txt"
    # # single-scale
    # # args.checkpoint_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/logs/ISIC2019_checkpt/last.ckpt"
    # # args.yaml_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/configs/top_scale_pretrain_ISIC2019.yaml"
    # # multiple-scale
    # args.checkpoint_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/logs/ISIC2019_checkpt_multiscale/last.ckpt"
    # args.yaml_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/configs/load_top_mul_scale_ISIC2019.yaml"
    
    args.model_mode = "mae"
    args.save_path = "/home/xjiangbh/ModelZoo_work/VQD-SR/taming-transformers/results_compare/multiscalematching_mae/"
    train_vqgan = CompareVQGAN(args)



