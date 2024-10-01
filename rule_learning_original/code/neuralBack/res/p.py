import pickle 
with open('rule_learning_original/code/neuralBack/res/gp.mdweights.pkl', 'rb') as f:
    w = pickle.load(f)
    f.close()