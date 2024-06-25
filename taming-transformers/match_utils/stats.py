from match_utils import nethook, dataset
import torch
import torchvision
from tqdm import tqdm

def get_mean_std(gan, gan_layers, discr, discr_layers, dataset, device, num, args):
    '''Get activation statistics over dataset for GAN and discriminative model.'''
    
    print("Collecting Dataset Statistics")
    gan_stats_list = []
    discr_stats_list = []
    with torch.no_grad():
        with tqdm(range(num)) as pbar:
            for iteration, img in zip(pbar, dataset):
                img = img.to(device=device)
                out_gan = gan(img)
                # out_discr = discr(img)
                if args.model_mode=='clip':
                    out_sr = discr.model.encode_image(img)
                else:
                    out_sr = discr(img)
                del img
                
                #### append GAN layer activations for batch
                gan_activs = {}
                for layer in gan_layers:
                    gan_activs[layer] = []    
                    gan_activation = gan.retained_layer(layer, clear = True)
                    gan_activs[layer].append(gan_activation)
                    # print('vqgan layer:',layer)
                    # print('gan_activation.shape:',gan_activation.shape)
                    # [1,256,28,28] [1,256,56,56] [1,256,112,112]
                    
                discr_activs = {}
                for layer in discr_layers:
                    discr_activs[layer] = []
                    discr_activation = discr.retained_layer(layer, clear = True)
                    discr_activs[layer].append(discr_activation)
                    print('source layer:',layer)
                    print('discr_activation.shape:',discr_activation.shape)
                    # mlp.fc2 [1,768,28,28]
                
                

                batch_gan_stats_list = []
                for layer in gan_layers:
                    
                    print('layer:',layer)
                    gan_activs[layer] = torch.cat(gan_activs[layer], 0) #images x channels x m x m
                    gan_activs[layer] = torch.permute(gan_activs[layer], (1,0,2,3)).contiguous() #channels x images x m x m
                    gan_activs[layer] = gan_activs[layer].view(gan_activs[layer].shape[0], -1) 
                    batch_gan_stats_list.append([torch.mean(gan_activs[layer],dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                        torch.std(gan_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])
                del gan_activs
                gan_stats_list.append(batch_gan_stats_list)

                batch_discr_stats_list = []
                for layer in discr_layers:
                    discr_activs[layer] = torch.cat(discr_activs[layer], 0)
                    discr_activs[layer] = torch.permute(discr_activs[layer], (1,0,2,3)).contiguous()
                    discr_activs[layer] = discr_activs[layer].view(discr_activs[layer].shape[0], -1)
                    batch_discr_stats_list.append([torch.mean(discr_activs[layer], dim=-1, dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                        torch.std(discr_activs[layer], dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)])
                del discr_activs
                discr_stats_list.append(batch_discr_stats_list)

                
                torch.cuda.empty_cache()
                if iteration==num-1:
                    break
    
        # print('gan_stats_list:',gan_stats_list)
        # print('discr_stats_list:',discr_stats_list)
        print('len(gan_stats_list):',len(gan_stats_list))
        print('len(discr_stats_list):',len(discr_stats_list))
        print('len(dataset):',len(dataset))
    ####################### After iterating
        print("Finished Iterating for Stats")
        final_discr_stats_list = []

        for iii in range(len(batch_discr_stats_list)):
            means = torch.zeros_like(batch_discr_stats_list[iii][0])
            stds = torch.zeros_like(batch_discr_stats_list[iii][1])
            for jjj in range(num):
                means+=discr_stats_list[jjj][iii][0]
                stds+=discr_stats_list[jjj][iii][1]**2

            final_discr_stats_list.append([means/num, torch.sqrt(stds/num)])



        final_gan_stats_list = []

        for iii in range(len(batch_gan_stats_list)):
            means = torch.zeros_like(batch_gan_stats_list[iii][0])
            stds = torch.zeros_like(batch_gan_stats_list[iii][1])
            for jjj in range(num):
                means+=gan_stats_list[jjj][iii][0]
                stds+=gan_stats_list[jjj][iii][1]**2

            final_gan_stats_list.append([means/num, torch.sqrt(stds/num)])

    return final_gan_stats_list, final_discr_stats_list

