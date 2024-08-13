# Medium-Biosciences-Challenge: Structure-Conditioned Classification of CATH Domain Architecture from Protein Sequence

For a quick overview before the written report: The 3 model checkpoints (Sequence Model, Sequence+Structure Model, GNN(For Struct Embeddings)) are all in the "checkpoints" folder. The final cleaned dataset that was used for training is in the "datasets" folder. The sequence and structure embeddings saved for training the models are in the "embeddings" folder. Finally, the 2 models, sequence_model.ipynb and seq_struct_model.ipynb, along with development.ipynb, which is all the code used for development (data cleaning, handling missing amino acids, homology clustering, training, etc...), are in the "models" folder. The paths are set to work when running in the repository, however, if being run on CoLab or something else, they may need to be changed. These same files are all in the shared Google Drive folder as well. 


**Data Processing**:

All initial data was provided in the Google Drive folder. In order to effectively use the data to train the model, the main columns of interest in the dataset provided were the 'sequences', 'architecture', 'cath_indices' and 'cath_id' columns. First, in order to handle gaps and missing amino acids in the sequences, a script was written to compare the CATH indices for each sequence with the indices retrieved from its corresponding PDB file, and if there were missing indices in the range, these sequences were put into another dataset. Using this dataset of only sequences with missing indices for time efficiency, a script was written to use the PDB ID of the sequence and scrape both UniProt and the RSCB PDB Bank for the full original sequence, and then the CATH indices were applied to the sequence to gather solely the real CATH domain of the sequence. These new sequences then replaced the old sequences in the original dataset, completing a dataset of full CATH domain sequences with no gaps.

In order to create an effective and representative train/validation split, sequence homology was taken into account by clustering using MMSeqs2, a homology-based clustering package, which clustered the sequences at 30% sequence identity, and 80% of the clusters were added to the train set, while the remaining 20% were added to the validation set, ensuring significant structural diversity between the train and test sets and accounting for any potential data leakage.

Finally, after all this initial data-processing was done, duplicate sequences were removed, and the dataset was checked for any NaN or missing values in any column, and if any were found, those rows were subsequently removed.


**Model**:

Sequence Model: 

The solely sequence-based model is designed to classify the CATH architecture of protein sequences based on their ESM-2 embeddings. The architecture consists of the following components:

Self-Attention Layer:

The model utilizes a multihead self-attention mechanism, implemented using nn.MultiheadAttention, with 4 attention heads. The attention layer takes in the protein embedding and produces an output that highlights the most relevant parts of the sequence according to the attention mechanism. The use of multihead attention inherently provides a level of permutation invariance and the ability to capture long-range dependencies within the sequence, and the self-attention mechanism allows the model to focus on different parts of the input sequence and understand the relationships between different elements within the sequence as the Q, K, V (query, key, value) vectors, all on the sequence embedding itself, are updated with each pass through the model. Through the concatenation of the original embedding with the attention outputs, we can weigh different parts of the sequence differently, which is crucial for understanding complex biological data. 


Fully Connected Layers:

Layer 1: Takes the concatenated output of the original embedding and the attention output, and applies a linear transformation followed by a ReLU activation function, reducing the dimension to 1280.
Layer 2: Further reduces the dimensionality to 640 with another linear transformation and ReLU activation.
Layer 3: Continues the reduction to 320 using a linear layer followed by ReLU.
Layer 4: The final layer maps the output to the number of classes (10 in this case), which represents the different possible architectures the protein sequence can be classified into.


Loss: Cross-Entropy Loss

The loss function used in this model was Cross-Entropy Loss, which measures the difference between the predicted probability distribution over classes and the true distribution. The softmax function is applied to the logits at the end of the fully-connected layers to convert them into a probability distribution over the classes, and then loss is calculated between the predicted and true labels.

Optimizer: Adam

This model used the Adam optimizer, which is very commonly used in literature when training complex neural networks. It adapts the learning rate for each parameter individually, which is useful for noisy or sparse gradients, and it can help the model converge faster and more efficiently. Compared to other optimization algorithms, it is also relatively computaionally inexpensive, and its momentum calculation can help gradients converge faster as well.


Performance: Accuracy 

The goal of this model is to make as many correct classifications as possible, given a sequence. Therefore, accuracy was used as the main performancce metric on the validation set of the model, which shows the total number of correct predictions as a percentage of the total number of predictions made.


Structure Model:









