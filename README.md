# Medium-Biosciences-Challenge: Structure-Conditioned Classification of CATH Domain Architecture from Protein Sequence

For a quick overview before the written report: The 3 model checkpoints (Sequence Model, Sequence+Structure Model, GNN(For Struct Embeddings)) are all in the "checkpoints" folder. The final cleaned dataset that was used for training is in the "datasets" folder. The sequence and structure embeddings saved for training the models are in the "embeddings" folder. Finally, the 2 models, sequence_model.ipynb and seq_struct_model.ipynb, along with development.ipynb, which is all the code used for development (data cleaning, handling missing amino acids, homology clustering, training, etc...), are in the "models" folder. The paths are set to work when running in the repository, however, if being run on CoLab or something else, they may need to be changed. These same files are all in the shared Google Drive folder as well. 


**Data Processing**:

All initial data was provided in the Google Drive folder. In order to effectively use the data to train the model, the main columns of interest in the dataset provided were the 'sequences', 'architecture', 'cath_indices' and 'cath_id' columns. First, in order to handle gaps and missing amino acids in the sequences, a script was written to compare the CATH indices for each sequence with the indices retrieved from its corresponding PDB file, and if there were missing indices in the range, these sequences were put into another dataset. Using this dataset of only sequences with missing indices for time efficiency, a script was written to use the PDB ID of the sequence and scrape both UniProt and the RSCB PDB Bank for the full original sequence, and then the CATH indices were applied to the sequence to gather solely the real CATH domain of the sequence. These new sequences then replaced the old sequences in the original dataset, completing a dataset of full CATH domain sequences with no gaps.

In order to create an effective and representative train/validation split, sequence homology was taken into account by clustering using MMSeqs2, a homology-based clustering package, which clustered the sequences at 30% sequence identity, and 80% of the clusters were added to the train set, while the remaining 20% were added to the validation set, ensuring significant structural diversity between the train and test sets and accounting for any potential data leakage.

Finally, after all this initial data-processing was done, duplicate sequences were removed, and the dataset was checked for any NaN or missing values in any column, and if any were found, those rows were subsequently removed.

For structural data, a script was written where all PDBs were processed and the atom coordinates and edges were determined for each file and added into the final dataframe. 


**Model**:

![Seq+Struct Architecture](https://github.com/rishab-pulugurta/medium-bio-challenge/blob/main/architecture.png)

**Sequence Model**: 

The solely sequence-based model is designed to classify the CATH architecture of protein sequences based on their ESM-2 embeddings. The architecture consists of the following       components:

Self-Attention Layer:

The model utilizes a multihead self-attention mechanism, implemented using nn.MultiheadAttention, with 4 attention heads. The attention layer takes in the protein embedding and   produces an output that highlights the most relevant parts of the sequence according to the attention mechanism. The use of multihead attention inherently provides a level of permutation invariance and the ability to capture long-range dependencies within the sequence, and the self-attention mechanism allows the model to focus on different parts of the input sequence and understand the relationships between different elements within the sequence as the Q, K, V (query, key, value) vectors, all on the sequence embedding itself, are updated with each pass through the model. Through the concatenation of the original embedding with the attention outputs, we can weigh different parts of the sequence differently, which is crucial for understanding complex biological data. 


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


**Structure Model**:

The structure-model is a GNN-based model which was designed using Graph Attention Networks (GATs) in order to generate meaningful embeddings from the generated structural data froom the PDBs. While designed as a classifier, the model was used to extract structure-rich embeddings before the final classification layer. The architecture consists of the following components:

Graph Attention Network (GAT) Layers:

GATConv Layers: The model uses three GAT convolutional layers (conv1, conv2, and conv3). These layers apply attention mechanisms over graph nodes, allowing the model to focus on the most relevant neighboring nodes when updating each node's features. The GAT layers allow the model to learn which nodes in the graph are most relevant for determining each node's feature, providing a form of context-dependent feature extraction that is well-suited for graphs and it creates structure-awareness, as the model is explicitly designed to work with graph-structured data, making it aware of the relationships between nodes. After the last GAT layer, the model uses global max pooling to aggregate node features across the entire graph into a single vector for each graph in the batch. This pooled vector is then used as the input to the fully connected layers.

Skip Connections: The model employs skip connections (residual connections) after each GAT layer, helping to preserve information from earlier layers, mitigating the vanishing gradient problem, and improving gradient flow during training.

Fully Connected Layers:

After the GAT layers, the model has three fully connected layers, which take the pooled graph representation and further transform it. These layers reduce the dimensionality step by step until reaching the desired output size.
Layer 1: Takes the output from the global max pooling operation and applies a linear transformation followed by a ReLU activation function, reducing the dimensionality to 320.
Layer 2: Further reduces the dimensionality to 128 with another linear transformation and ReLU activation.
Layer 3: The final layer maps the output to the number of classes, which represents the different possible architectures the structural data can be mapped to. 
However, the output of interest for this purpose is the intermediate representation 'x' obtained after global max pooling and before passing through the fully connected layers, as we only need the embeddings. 


Loss: Cross-Entropy Loss

The loss function used in this model was Cross-Entropy Loss, which measures the difference between the predicted probability distribution over classes and the true distribution. The softmax function is applied to the logits at the end of the fully-connected layers to convert them into a probability distribution over the classes, and then loss is calculated between the predicted and true labels.

Optimizer: Adam

This model used the Adam optimizer, which is very commonly used in literature when training complex neural networks. It adapts the learning rate for each parameter individually, which is useful for noisy or sparse gradients, and it can help the model converge faster and more efficiently. Compared to other optimization algorithms, it is also relatively computaionally inexpensive, and its momentum calculation can help gradients converge faster as well.

Performance: Accuracy 

The goal of this model is to make as many correct classifications as possible, given structural data. Therefore, accuracy was used as the main performancce metric on the validation set of the model, which shows the total number of correct predictions as a percentage of the total number of predictions made. Even though this was not the main purpose of this model, it proves that we are generating meaningful structurally-conditioned embeddings for use. 


**Sequence+Structure Model**:

The combined Sequence-And-Structure-based model is designed to classify the CATH architecture of protein sequences based on the combination of their ESM-2 embeddings and structural embeddings generated from a trained GNN, in hopes of creating an embedding that holds as much information about the protein as possible. The architecture consists of the following components:

Self-Attention Layers:

Sequence Attention: This layer applies a multihead self-attention mechanism to the sequence embeddings. It allows the model to focus on different parts of the sequence, capturing dependencies and relationships within the sequence data.

Structure Attention: Similarly, this layer applies multihead self-attention to the structure embeddings, enabling the model to identify important features and relationships within the structural data.

By applying self-attention to both sequence and structure embeddings, the model captures contextual information from both sources. This dual attention mechanism ensures that the model can learn the most relevant features from both types of data, and then is able to effectively integrate sequence and structural information by concatenating the original embeddings with their attention-enhanced counterparts, creating a final embedding that is conditioned as well on the protein as possible. Using separate self-attention layers for sequence and structure embeddings allows the model to independently learn and focus on the most important aspects of each type of data. 

Fully Connected Layers:

Layer 1: Takes the concatenated output of the original sequence and structure embeddings along with their corresponding attention outputs. It applies a linear transformation followed by a ReLU activation function, reducing the dimensionality to 1280.
Layer 2: Further reduces the dimensionality to 640 with another linear transformation and ReLU activation.
Layer 3: Continues the reduction to 320 using a linear layer followed by ReLU.
Layer 4: The final layer maps the output to the number of classes, representing the different architectures the combined sequence and structure data can be classified into.


Loss: Cross-Entropy Loss

The loss function used in this model was Cross-Entropy Loss, which measures the difference between the predicted probability distribution over classes and the true distribution. The softmax function is applied to the logits at the end of the fully-connected layers to convert them into a probability distribution over the classes, and then loss is calculated between the predicted and true labels.

Optimizer: Adam

This model used the Adam optimizer, which is very commonly used in literature when training complex neural networks. It adapts the learning rate for each parameter individually, which is useful for noisy or sparse gradients, and it can help the model converge faster and more efficiently. Compared to other optimization algorithms, it is also relatively computaionally inexpensive, and its momentum calculation can help gradients converge faster as well.


Performance: Accuracy 

The goal of this model is to make as many correct classifications as possible, given sequence and structure data. Therefore, accuracy was used as the main performancce metric on the validation set of the model, which shows the total number of correct predictions as a percentage of the total number of predictions made.

**Experiments**:

![Evaluation Experiments](https://github.com/rishab-pulugurta/medium-bio-challenge/blob/main/evaluation.png)

The experiments conducted can be seen in the table. 

The different methods of encoding and clustering the training data had a significant impact on model performance. When using one-hot encoding for sequence data, both the Sequence-Only and Struct+Seq models achieved lower accuracies, 0.32 and 0.28, respectively, indicating that one-hot encoding cannot capture the complexities of the data effectively. In contrast, using ESM embeddings led to a substantial increase in accuracy for both the Sequence-Only model, from 0.32 to 0.68, and the Struct+Seq model. from 0.28 to 0.74, highlighting the effectiveness of ESM in capturing more sequence-rich features. Additionally, clustering the data using MMSeq resulted in higher accuracy (0.74) compared to a random split (0.64), suggesting that clustering with MMSeq provides more meaningful groupings that enhance model performance, creating a representative training and validation set.

**Analysis**:

Strengths:

Integration of Sequence and Structure Data: The Seq-Struct model combines both sequence and structure data using self-attention mechanisms, which allows it to capture rich and contextually relevant features from both sequence and structural data. This dual-attention approach enhances the model's ability to learn complex relationships in the data, leading to improved performance, particularly when both types of information are essential. 

Generalizability: The use of self-attention for both sequence and structure data allows the model to be flexible and applicable to various tasks where these types of data are available. The model is likely to generalize well to similar datasets, particularly those that include both sequence and structure information.

Scalability: The architecture can scale with larger datasets and more complex tasks by increasing the number of attention heads or the depth of the network. Additionally, the model's modular design using attention on structure and sequence separately makes it easier to adapt to different data types, such as using only one or the other, or more complex tasks without needing significant architectural changes.


Weaknesses:

Dependence on Embeddings: The model's performance heavily depends on the quality of the input embeddings. While ESM-2 has been proven in literature to be highly effective at capturing sequence representations, a newly-trained GNN, as used in this model, may not capture effective structural embeddings. Poor quality embeddings or noisy data can significantly degrade the model's performance, limiting its generalizability.

Multi-Step Process: The process of first embedding the sequence and structural data separately before being integrated in the model may be a bit inconvenient and could increase complexity, and could potentially work better if integrated directly into the testing loop.

The model tended to fail in terms of achieving any successful results past a certain threshold of accuracy, which is most likely due to the limitations of the added structural embeddings and limited data availibility. Because of how complex structural data from PDBs may be, simply capturing edge/connections data along with atom coordinates may not be enough to capture effective structure representations, which will lead to a poorly trained GNN. In addition, the initial dataset itself was slightly flawed, as during cleaning, it was noticed some of the original sequences did not match CATH domain when solely using the CATH indices on the full sequence fetched from RCSB PDB or UniProt, and there were only ~6000 training samples, which may not be enough to train a model of this scale effectively. However, the addition of structure embeddings still was able to allow the model to learn more information from the data, as seen in the performance increases, which goes to show the potential of improvements when adding even more established structural data, such as rotational information and other translational data from the space of the 3D structure.

**Future Work**:

Utilizing an IPA Transformer for Improved Structural Embeddings:

An Invariant Point Attention (IPA) Transformer is designed specifically for tasks that involve 3D structures, such as protein modeling. IPA Transformers have been shown to effectively capture complex spatial relationships and invariant features, making them ideal for generating high-quality structural embeddings, as this is exactly what is used in AlphaFold2 before finally generating a structure prediction. This architecture takes in far more complex structural data, such as rotation matrices, pairwise distances, angles, and translatiton vectors, which creates a far more structrally-rich conditioned embedding for use. 

Leveraging Inverse Folding for Refining Structural Embeddings:

Inverse folding refers to the process of determining the sequence that best fits a given protein structure, essentially the reverse of the typical sequence-to-structure prediction. By integrating inverse folding into the model, it would be possible to refine structural embeddings, ensuring that they not only represent the 3D conformation but also encapsulate the sequence information that corresponds to that structure. 

Enhancing Joint Representation:

Rather than simply concatenating the structural and sequence embeddings, it would very interesting to try an approach that effectively combines both embeddings in a latent space and does further transormations on this joint embedding. A new generative architecture in literature that has been very effective in certain protein models has been flow matching, which involves training a flow-based model to learn a transformation between a simple distribution, which can just be modeled as a simple Gaussian distribution, and a more complex distribution, such as our joint embeddings of structure and sequence data. This can result in a more feature-rich final embedding to be used for further downstream tasks, such as out CATH architecture classification prediction, but also  when combined with other data, can be used in designing structural binders, backbones, and other molecules.


















