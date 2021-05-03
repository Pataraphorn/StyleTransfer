import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_filter(model,model_weights,conv_layers):
    model_children = list(model.children())

    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")

    for weight, conv in zip(model_weights, conv_layers):
        # print(f"WEIGHT: {weight} ====> SHAPE: {weight.shape}")
        print(f"CONV: {conv} ====> SHAPE: {weight.shape}")


def visualizeLayerFilter(convLayer,model_weights):
    plt.figure(figsize=(10, 8))
    for i, filter in enumerate(model_weights[convLayer]):
        plt.subplot(8, 8, i+1)
        plt.imshow(filter[0, :, :].cpu().detach(), cmap='gray')
        plt.axis('off')
        # plt.savefig('../outputs/filter.png')
    plt.show()


def visualImageFilter(img,model_weights,conv_layers):
    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results
    # visualize 64 features from each layer 
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(15, 13))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.cpu().data
        print('layer : ',num_layer,layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        # print(f"Saving layer {num_layer} feature maps...")
        # plt.savefig(f"../outputs/layer_{num_layer}.png")
        plt.show()
        plt.close()