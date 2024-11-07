import torch
import matplotlib.pyplot as plt

kit_green = (0, 150/255, 130/255)
kit_blue = (70/255, 100/255, 170/255)
kit_red = (162/255, 34/255, 35/255)
kit_green_50 = (0.5, 0.7941, 0.7549)
kit_blue_50 = (0.6373, 0.6961, 0.8333)
kit_red_50 = (0.8176, 0.5667, 0.5686)

def sigma_weight_plot(weights, sigmas, base_name):
    # Creates scatter plot for a layer's sigma values and weights
    final_weights = (torch.flatten(weights)).tolist()
    final_sigma = (torch.abs(torch.flatten(sigmas))).tolist()

    x1 = final_sigma

    y1 = final_weights

    plt.figure(figsize=(10, 8))
    plt.scatter(x1, y1, c=kit_blue_50,
                linewidths=2,
                marker="s",
                edgecolor=kit_blue,
                s=50)
    # plt.xlim(0, 1)
    # plt.ylim(-0.5, 0.5)

    plt.xlabel("Sigma Values", fontsize=18)
    plt.ylabel("Weights", fontsize=18)

    file_name = base_name + '-sigma-values.png'
    plt.savefig(file_name)
    #plt.show()