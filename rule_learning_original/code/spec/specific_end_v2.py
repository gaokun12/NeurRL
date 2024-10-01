import numpy as np
# from utils import read_list
import scipy.sparse as sp

# get the demo data 


# Fetch the dataset
# dataset = fetch_rcv1(subset="all")
# print("Dataset RCV1 loaded...")
# data = dataset.data
# target = dataset.target

# Get the split between training/test set and validation set
# test_indices = read_list("split/rcv1/test")
# n_test = test_indices.shape[0]
# validation_indices = read_list("split/rcv1/validation")
# n_validation = validation_indices.shape[0]

# # Filter the dataset
# ## Keep only the data points in the test and validation sets
# test_data = data[test_indices]
# test_target = target[test_indices]
# validation_data = data[validation_indices]
# validation_target = target[validation_indices]
# data = sp.vstack([test_data, validation_data])
# target = sp.vstack([test_target, validation_target])
# ## Update test_indices and validation_indices to fit the new data indexing
# test_indices = np.asarray(range(0, n_test)) # Test points come first in filtered dataset
# validation_indices = np.asarray(range(n_test, n_test + n_validation)) # Validation points come after in filtered dataset

# # Pre-process the dataset
# ## Filter words based on tf-idf
# sum_tfidf = np.asarray(sp.spmatrix.sum(data, axis=0))[0] # Sum of tf-idf for all words based on the filtered dataset
# word_indices = np.argpartition(-sum_tfidf, 2000)[:2000] # Keep only the 2000 top words in the vocabulary
# data = data[:, word_indices].toarray() # Switch from sparse matrix to full matrix
# ## Retrieve the unique label (corresponding to one of the specified categories) from target's label vector
# names = dataset.target_names
# category_names = ['CCAT', 'ECAT', 'GCAT', 'MCAT']
# category_indices = [i for i in range(len(names)) if names[i] in category_names]
# dict_category_indices = {j: i for i, j in enumerate(category_indices)} # To rescale the indices between 0 and some K
# filtered_target = []
# for i in range(target.shape[0]): # Loop over data points
#     target_coo = target[i].tocoo().col
#     filtered_target_coo = [t for t in target_coo if t in category_indices]
#     assert len(filtered_target_coo) == 1 # Only one relevant label per document because of pre-filtering
#     filtered_target.append(dict_category_indices[filtered_target_coo[0]])
# target = np.asarray(filtered_target)
# n_samples = data.shape[0] # Number of samples in the dataset
# n_clusters = 4 # Number of clusters to obtain

# # Auto-encoder architecture
# # input_size = data.shape[1]
# hidden_1_size = 500
# hidden_2_size = 500
# hidden_3_size = 2000
# embedding_size = n_clusters
# dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
#               hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
# activations = [torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None, # Encoder layer activations
#                torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.ReLU(), None] # Decoder layer activations
# names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
#          'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names



global data 
data = 'image'
def main():
    global n_clusters, period_length, subsequence_length, stride, val_lambda, hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, pretrain_epoch, finetuing_epoch, finetuing_epoch_outter, high_epoch, alphas, sub_epoch, finetuing_logic_epoch, learning_rate_dkm, learning_rate_ae, learning_rate_logic, pre_training, batch_size, ae_batch_size, image_row, image_column
    if data == 'demo':
        n_clusters = 2 # 5 Number of clusters to obtain
        period_length = 5 # 20
        subsequence_length = 5 #10
        stride = 1
        val_lambda = 1
        hidden_1_size = 500
        hidden_2_size = 500
        hidden_3_size = 2000
        embedding_size = n_clusters
        pretrain_epoch = 50 #50
        finetuing_epoch = 0
        finetuing_epoch_outter = 0
        high_epoch = 2
        alphas = 1000
        sub_epoch = 1
        finetuing_logic_epoch = 1000 #300
        learning_rate_dkm = 0.001
        learning_rate_ae = 0.001
        learning_rate_logic = 0.01
        pre_training=True
        batch_size = 512
        ae_batch_size = 2**18
        image_row = -1
        image_column = -1
        print('load DEMO para done', 'n_cluster', n_clusters,'subsequence length', subsequence_length, 'region', period_length)
    elif 'UCR' in data:
        n_clusters = 4 # 5 Number of clusters to obtain
        period_length = 3 # 10
        # number_period = 100
        subsequence_length = 2 #3
        stride = 1
        val_lambda = 1
        hidden_1_size = 500
        hidden_2_size = 500
        hidden_3_size = 2000
        embedding_size = 3 #n_clusters
        pretrain_epoch = 50 # 50 ! train autoencoder first
        finetuing_epoch = 0
        finetuing_epoch_outter = 0 #! for train dkm epochs
        alphas = 1000
        sub_epoch = 1
        finetuing_logic_epoch = 1000 #!1000 for training dkm and logic at the same time 
        learning_rate_dkm = 0.001
        learning_rate_ae = 0.001
        learning_rate_logic = 1
        pre_training=True # !train autoencoder first 
        batch_size = 64
        ae_batch_size = 2**18
        image_row = -1
        image_column = -1
        print('load UCR para done', 'n_cluster', n_clusters,'subsequence length', subsequence_length, 'region', period_length)
    elif 'image' in data:
        n_clusters = 5 # 5 Number of clusters to obtain
        period_length = 5 # 10
        # number_period = 100
        subsequence_length = 3 #3
        stride = 1
        val_lambda = 1
        hidden_1_size = 500
        hidden_2_size = 500
        hidden_3_size = 2000
        embedding_size = 3 #n_clusters
        pretrain_epoch = 5 # 50 ! train autoencoder first
        finetuing_epoch = 0
        finetuing_epoch_outter = 0 #! for train dkm epochs
        alphas = 1000
        sub_epoch = 1
        finetuing_logic_epoch = 20 #!1000 for training dkm and logic at the same time 
        learning_rate_dkm = 0.001
        learning_rate_ae = 0.001
        learning_rate_logic = 0.1
        pre_training=True # !train autoencoder first 
        batch_size = 128
        ae_batch_size = 2**18
        image_row = 28
        image_column = 28
        print('load Image para done', 'n_cluster', n_clusters,'subsequence length', subsequence_length, 'region', period_length)
if __name__ == '__main__':
    main()
    
    
# # Italy data parameters
#         n_clusters = 4 # 5 Number of clusters to obtain
#         period_length = 3 # 10
#         # number_period = 100
#         subsequence_length = 2 #3
#         stride = 1
#         val_lambda = 1
#         hidden_1_size = 500
#         hidden_2_size = 500
#         hidden_3_size = 2000
#         embedding_size = 3 #n_clusters
#         pretrain_epoch = 50 # 50 ! train autoencoder first
#         finetuing_epoch = 0
#         finetuing_epoch_outter = 0 #! for train dkm epochs
#         alphas = 1000
#         sub_epoch = 1
#         finetuing_logic_epoch = 1000 #!1000 for training dkm and logic at the same time 
#         learning_rate_dkm = 0.001
#         learning_rate_ae = 0.001
#         learning_rate_logic = 0.1
#         pre_training=True # !train autoencoder first 
#         batch_size = 64
#         ae_batch_size = 2**18
#         image_row = -1
#         image_column = -1
#         print('load UCR para done')