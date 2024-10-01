import torch 
import collections
import numpy as np
import pickle
TORCH_FLOAT_TYPE = torch.float32

class DkmCompGraph(torch.nn.Module):
    def __init__(self, ae_specs, n_clusters, val_lambda, input_size, device):
        super(DkmCompGraph, self).__init__() 
        self.ae_specs = ae_specs
        self.input_size = input_size
        self.embedding_size = ae_specs[0][int((len(ae_specs[0])-1)/2)]
        
        # kmeans loss computations
        ## Tensor for cluster representatives
        minval_rep, maxval_rep = -1, 1
        # cluster_rep = (maxval_rep-minval_rep) * torch.rand(n_clusters, self.embedding_size, dtype=TORCH_FLOAT_TYPE) + minval_rep
        cluster_rep = torch.tensor(np.zeros((n_clusters, self.embedding_size)), dtype=TORCH_FLOAT_TYPE, device=device)
        self.cluster_rep = torch.nn.Parameter(cluster_rep, requires_grad=True)
        
        self.n_clusters = n_clusters
        self.val_lambda = val_lambda
        self.encoder, self.decoder = self.build_autoencoder(ae_specs)
        self.encoder.to(device)
        self.decoder.to(device)
        self.device = device
    
    def fc_layers(self, input:int, specs):
        [dimensions, activations, names] = specs
        layers = collections.OrderedDict()
        for dimension, activation, name in zip(dimensions, activations, names):
            layers[name] = torch.nn.Linear(in_features=input, out_features=dimension)
            if activation is not None:
                layers[name+'activation'] = activation
            input = dimension
        return torch.nn.Sequential(layers)
    
    def build_autoencoder(self, specs):
        [dimensions, activations, names] = specs
        mid_ind = int(len(dimensions)/2)

        # Encoder
        embedding = self.fc_layers(self.input_size, [dimensions[:mid_ind], activations[:mid_ind], names[:mid_ind]])
        # Decoder
        output = self.fc_layers(self.embedding_size, [dimensions[mid_ind:], activations[mid_ind:], names[mid_ind:]])
        


        # for name, param in embedding.named_parameters():
        #     print(name, param)
        # for name, param in output.named_parameters():
        #     print(name, param)

        # # initial values with pickle vlaues 
        # for index, name in enumerate(['enc_hidden_1', 'embedding']):
        #     with open('deep-k-means/{}_kernel:0.pkl'.format(name), 'rb') as f:
        #         weight = pickle.load(f)
        #         print(weight)
        #         weight = np.transpose(weight)
        #         embedding[index*2].weight.data = torch.tensor(weight, dtype=TORCH_FLOAT_TYPE)
        #         f.close()
        #     with open('deep-k-means/{}_bias:0.pkl'.format(name), 'rb') as f:
        #         bias = pickle.load(f)
        #         embedding[index*2].bias.data = torch.tensor(bias, dtype=TORCH_FLOAT_TYPE)
        #         f.close()
                
        # for index, name in enumerate(['dec_hidden_1', 'output']):
        #     with open('deep-k-means/{}_kernel:0.pkl'.format(name), 'rb') as f:
        #         weight = pickle.load(f)
        #         weight = np.transpose(weight)
        #         output[index*2].weight.data = torch.tensor(weight, dtype=TORCH_FLOAT_TYPE)
        #         f.close()
        #     with open('deep-k-means/{}_bias:0.pkl'.format(name), 'rb') as f:
        #         bias = pickle.load(f)
        #         output[index*2].bias.data = torch.tensor(bias, dtype=TORCH_FLOAT_TYPE)
        #         f.close()

        
        # for name, param in embedding.named_parameters():
        #     print(name, param)
        # for name, param in output.named_parameters():
        #     print(name, param)
        
        return embedding, output
    
    def autoencoder(self, input):
        embedding = self.encoder(input)
        output = self.decoder(embedding)
        return embedding, output
    
    def f_func(self, x, y):
        return torch.sum(torch.square(x - y), axis=1)

    def g_func(self, x, y):
        return torch.sum(torch.square(x - y), axis=1)
    
    def get_reconstruction_loss(self, input):
        embedding, output = self.autoencoder(input)
        rec_error = self.g_func(input, output)
        ae_loss = torch.mean(rec_error)
        return ae_loss, embedding, output
    
    def forward(self, input, alpha=1):
        list_dist = []
        self.ae_loss,embedding,_ = self.get_reconstruction_loss(input)
        for i in range(0, self.n_clusters):
            dist = self.f_func(embedding, self.cluster_rep[i, :].reshape(1, self.embedding_size))
            list_dist.append(dist)
        self.stack_dist = torch.stack(list_dist)
        
        min_dist = torch.min(self.stack_dist, dim=0).values
        
        list_exp = []
        for i in range(self.n_clusters):
            exp = torch.exp(-1 * alpha * (self.stack_dist[i] - min_dist))
            list_exp.append(exp)
        stack_exp = torch.stack(list_exp) 
        sum_exponentials = torch.sum(stack_exp, dim=0)
        
        list_softmax = []
        list_weighted_dist = []
        for j in range(self.n_clusters):
            softmax = stack_exp[j] / sum_exponentials
            list_softmax.append(softmax)
            weighted_dist = softmax * self.stack_dist[j]
            list_weighted_dist.append(weighted_dist)
        stack_weighted_dist = torch.stack(list_weighted_dist)
        stack_possibility = torch.stack(list_softmax) # this can be used as the possible of cluster assignment
        
        self.kmeans_loss = torch.mean(torch.sum(stack_weighted_dist, axis=0))
        
        loss = self.ae_loss + self.val_lambda *self.kmeans_loss
        
        
        # print(self.cluster_rep)
        
        return loss, self.stack_dist, self.ae_loss, self.kmeans_loss, stack_possibility