import os
import math
import numpy as np
import argparse
import torch
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib import pyplot as plt
from compgraph import DkmCompGraph
import time 

def dkm(data, args = None, n_pretrain_epochs = 50, n_finetuning_epochs = 10, lambda_ = 1.0, batch_size = 256, validation = False, pretrain = True, annealing = False, seeded = False, cpu = False, n_clusters = 3):
    
    n_samples = data.shape[0]
    input_size = data.shape[1]
    
    # Auto-encoder architecture
    hidden_1_size = 500
    hidden_2_size = 500
    hidden_3_size = 2000
    embedding_size = n_clusters
    
    dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensio
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
    activations = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None, # Encoder layer activations
                torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None] # Decoder layer activations
    names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
            'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names
    
    if type(args) != type(None):
        n_pretrain_epochs = args.p_epochs
        n_finetuning_epochs = args.f_epochs
        lambda_ = args.lambda_
        batch_size = args.batch_size
        validation = args.validation
        pretrain = args.pretrain
        annealing = args.annealing
        seeded = args.seeded
        cpu = args.cpu


    if not annealing and pretrain:
        constant_value = 1
        max_n = 20
        alphas = 1000*np.ones(max_n, dtype=float)
        alphas = alphas/constant_value
    else:
        parser.error("Run with either annealing (-a) or pretraining (-p), but not both.")
        exit()


    if cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

    n_runs = 1
    all_clu_tar = []
    for run in range(n_runs):
        if seeded:
            seed = seeds[run]
            torch.manual_seed(seed)
            np.random.seed(seed)
        print("Run: ", run)
        
        cg = DkmCompGraph([dimensions, activations, names], n_clusters, lambda_)
        
        distances = np.zeros((n_clusters, n_samples))
        
        data = torch.Tensor(data).to(device)
        train_data = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
        
        pre_para = list(cg.encoder.parameters()) + list(cg.decoder.parameters())
        optimizer = torch.optim.Adam(pre_para , lr=0.001)
        if pretrain:
            print('Starting autoencoder pretraining')
            
            embeddings = np.zeros((n_samples, embedding_size), dtype=float)

            for epoch in range(n_pretrain_epochs):
                print("Pretraining epoch: ", epoch)
                
                for i, batch in enumerate(train_data):
                    optimizer.zero_grad()
                    ae_loss, embedding_, output = cg.get_reconstruction_loss(batch)
                    ae_loss.backward()
                    optimizer.step()
                    
                    for j in range(i* len(batch), (i+1)*len(batch)):
                        embeddings[j,:] = embedding_[j,:].detach().numpy()
                    print('ae_loss', ae_loss.item())   
        
        if (len(alphas)> 0):
            print('Starting DKM training ')
        kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++").fit(embeddings)
        cg.cluster_rep = torch.nn.Parameter(torch.tensor(kmeans_model.cluster_centers_, dtype=torch.float32), requires_grad=True)

        
        optimizer = torch.optim.Adam(cg.parameters(), lr=0.001)
        for k in range(len(alphas)):
            print("Training step: alpha[{}]: {}".format(k, alphas[k]))
            
            for epoch in range(n_finetuning_epochs):
                print("Fine-tuning epoch: ", epoch)
                for i, batch in enumerate(train_data):
                    optimizer.zero_grad()
                    loss,stack_dist_, ae_loss, kmean_loss = cg(batch, alphas[k])
                    loss.backward()

                    optimizer.step()
                    
                    for j in range(i* len(batch), (i+1)*len(batch)):
                        distances[:,j] = stack_dist_[:, j].detach().numpy()
                    print('loss', loss.item(), 'ae_loss', ae_loss.item(), 'kmean_loss', kmean_loss.item())

        cluster_assign = np.zeros((n_samples), dtype=float)
        for i in range(n_samples):
            index_closest_cluster = np.argmin(distances[:, i])
            cluster_assign[i] = index_closest_cluster
        cluster_assign = cluster_assign.astype(np.int64)
        print("Cluster assignment: ", cluster_assign)
        # print('Target: ', target )
        # all_clu_tar.append((cluster_assign, target))
    return cluster_assign

    # for parameter in cg.parameters():
    #     if parameter.requires_grad:
    #         print(parameter, parameter.data.size())
    # print('------')
    # for parameter in cg.parameters():
    #     if not parameter.requires_grad:
    #         print(parameter, parameter.size())
    # for name, param in cg.state_dict().items():
    #     print(name, param.size())
    
if __name__ == "__main__":
    demo_train = np.array([[1,3,5,4,2],[1,2,3,5,4],[3,3,3,3,3],[3.1,3.1,3.1,3.1,3.1]])
    demo_train = (demo_train - demo_train.min()) / (demo_train.max() - demo_train.min())
    parser = argparse.ArgumentParser(description="Deep k-means algorithm")
    parser.add_argument("-d", "--dataset", type=str.upper,
                        help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", default="DEMO")
    parser.add_argument("-v", "--validation", help="Split data into validation and test sets", action='store_true')
    parser.add_argument("-p", "--pretrain", help="Pretrain the autoencoder and cluster representatives",
                        action='store_false')
    parser.add_argument("-a", "--annealing",
                        help="Use an annealing scheme for the values of alpha (otherwise a constant is used)",
                        action='store_true')
    parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
    parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_false')
    parser.add_argument("-l", "--lambda", type=float, default=1.0, dest="lambda_",
                        help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
    parser.add_argument("-e", "--p_epochs", type=int, default=50, help="Number of pretraining epochs")
    parser.add_argument("-f", "--f_epochs", type=int, default=10, help="Number of fine-tuning epochs per alpha value")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Size of the minibatches used by the optimizer")
    args = parser.parse_args()
    print(demo_train.shape)
    dkm(data = demo_train , args= args)