# Medium-Biosciences-Challenge: Structure-Conditioned Classification of CATH Domain Architecture from Protein Sequence

For a quick overview before the written report: The 3 model checkpoints (Sequence Model, Sequence+Structure Model, GNN(For Struct Embeddings)) are all in the "checkpoints" folder. The final cleaned dataset that was used for training is in the "datasets" folder. The sequence and structure embeddings saved for training the models are in the "embeddings" folder. Finally, the 2 models, sequence_model.ipynb and seq_struct_model.ipynb, along with development.ipynb, which is all the code used for development (data cleaning, handling missing amino acids, homology clustering, training, etc...), are in the "models" folder. The paths are set to work when running in the repository, however, if being run on CoLab or something else, they may need to be changed. These same files are all in the shared Google Drive folder as well. 


**Data Processing**:

All initial data was provided in the Google Drive folder. In order to effectively use the data to train the model, the main columns of interest in the dataset provided were the 'sequences', 'architecture', 'cath_indices' and 'cath_id' columns. First, in order to handle gaps and missing amino acids in the sequences, a script was written to compare the CATH indices for each sequence with the indices retrieved from its corresponding PDB file, and if there were missing indices in the range, these sequences were put into another dataset. Using this dataset of only sequences with missing indices for time efficiency, a script was written to use the PDB ID of the sequence and scrape both UniProt and the RSCB PDB Bank for the full original sequence, and then the CATH indices were applied to the sequence to gather solely the real CATH domain of the sequence. These new sequences then replaced the old sequences in the original dataset, completing a dataset of full CATH domain sequences with no gaps.




