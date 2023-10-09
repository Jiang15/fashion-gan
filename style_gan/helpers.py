import random
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np
import torch
import functools


# generate z from seed
def generateZ_from_seed(seed, G, device = 'cuda'):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    return z

# generate image from z
def generate_image_from_z(z, G, truncation_psi =1, noise_mode = 'none'):
    G = G.float() 
    G.forward = functools.partial(G.forward, force_fp32=True)
    w = G.mapping(z, 0, truncation_psi=truncation_psi)
    img = G.synthesis(w, noise_mode=noise_mode, force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return np.expand_dims(w.cpu().numpy()[0,0],0), img[0].cpu().numpy()

# generate image from w
def generate_image_from_w(w, G, noise_mode = 'none'):
    G = G.float() 
    img = G.synthesis(w, noise_mode=noise_mode, force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return img[0].cpu().numpy()

# generate w from z
def generate_w(z, G, truncation_psi=1):
    G = G.float() 
    return G.mapping(z, 0, truncation_psi=truncation_psi)

# add directions in the specified layers
def E(vectors, vs, rs, ls, w, device = 'cuda'):
    for v,l,r in zip(vs,ls,rs):
        w[0,l]+= vectors[v]*rs
    return torch.from_numpy(w).float().to(device)

# visualize changes in 10 directions and 10 magnititude
def visualize_magnititue(G, vectors,  direction_ind, layers, magnititude = np.arange(-4,5,1)):
    fig, axs = plt.subplots(10,10,figsize=(30,30))
    for ind_x in range(10):
        z = generateZ_from_seed(random.randint(1,100), G, )
        # get original w
        w_origin = generate_w(z, G)
        for ind_y, mag in enumerate(magnititude):
            # get modified w
            w_modify = E(vectors, direction_ind, [mag], layers, w_origin.cpu().numpy())
            img_modify = generate_image_from_w(w_modify, G)
            if mag == 0:
                rect = Rectangle((0,0),255,255,linewidth=2,edgecolor='g',facecolor='none')
                # Add the patch to the Axes
                axs[ind_x,ind_y].add_patch(rect)
            axs[ind_x,ind_y].imshow(img_modify)
            axs[ind_x,ind_y].axis('off')
    plt.show()

# visualize chnages in one direction and one magnititude
def visualize_changes(G, vectors, direction_ind, magnititude, layers, device):
    fig, axs = plt.subplots(2,10,figsize=(30,8))
    for ind, seed in enumerate(np.arange(10,20,1)):
        z = generateZ_from_seed(seed, G, device = device)
        w_origin = generate_w(z, G)
        w_modify = E(vectors, direction_ind, magnititude, layers, w_origin.cpu().numpy())
        img_ori = generate_image_from_w(w_origin, G)
        img_modify = generate_image_from_w(w_modify, G)
        axs[0,ind].imshow(img_ori)
        axs[0,ind].axis('off')
        axs[1,ind].imshow(img_modify)
        axs[1,ind].axis('off')
    plt.show()



# visualize the change of one image in ten directions
def visualize_img_changes(G, image, vectors, layers, noise_mode = 'none', num_steps = 1000):
    fig, axs = plt.subplots(10, 11, figsize=(30,35))
    w_origin = img_to_w(image, G, num_steps = num_steps)
    print(w_origin.shape)
    for ind_x in range(10):
        for ind_y, mag in enumerate(np.arange(-10,10,2)):
            # get modified w
            w_modify = E(vectors, [ind_x], [mag], layers, w_origin.cpu().numpy())
            img_modify = generate_image_from_w(w_modify, G, noise_mode = noise_mode)
            if mag == 0:
                rect = Rectangle((0,0),255,255,linewidth=2,edgecolor='g',facecolor='none')
                # Add the patch to the Axes
                axs[ind_x,ind_y].add_patch(rect)
            axs[ind_x,ind_y].imshow(img_modify)
            axs[ind_x,ind_y].axis('off')

    plt.show()

def visualize_two_magnititue(G, vectors,  direction_ind, layers, mag, noise_mode = 'none', num_steps = 1000):
    fig, axs = plt.subplots(len(mag), len(mag), figsize=(3*len(mag),3*len(mag)))
    seed = random.randint(1,100)
    z = generateZ_from_seed(seed, G)
    w_origin = generate_w(z, G, num_steps = num_steps)

    for ind_x in range(len(mag)):
        for ind_y in range(len(mag)):
            # get modified w
            w_tmp = E(vectors, direction_ind[0], [mag[ind_x]], layers[0], w_origin.cpu().numpy())
            w_modify = E(vectors, direction_ind[1], [mag[ind_y]], layers[1], w_tmp.cpu().numpy())

            img_modify = generate_image_from_w(w_modify, G)
            if mag[ind_y] == 0 and mag[ind_x] == 0:
                rect = Rectangle((0,0),255,255,linewidth=2,edgecolor='g',facecolor='none')
                # Add the patch to the Axes
                axs[ind_x,ind_y].add_patch(rect)
            axs[ind_x,ind_y].imshow(img_modify)
            axs[ind_x,ind_y].axis('off')
    plt.show()
