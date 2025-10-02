"""
Tutorial 4: Cell Type Prediction

In this tutorial, we will demonstrate how to use a pretrained Cell2Sentence (C2S) model 
to perform cell type prediction on single-cell RNA sequencing datasets. Cell type prediction 
is a crucial step in single-cell analysis, allowing researchers to identify and classify 
different cell populations within a dataset. By leveraging the power of C2S models, we can 
make accurate predictions based on the information encoded in cell sentences.

In this tutorial, you will:
1. Load an immune tissue single-cell dataset from Domínguez Conde et al. (preprocessed in 
   tutorial notebook 0, two sample donors)
   - Citation: Domínguez Conde, C., et al. "Cross-tissue immune cell analysis reveals 
     tissue-specific features in humans." Science 376.6594 (2022): eabl5197.
2. Load a pretrained C2S model that is capable of making cell type predictions.
3. Use the model to predict cell types based on the cell sentences derived from the dataset.
"""

# We will begin by importing the necessary libraries. These include Python's built-in libraries, 
# third-party libraries for handling numerical computations, progress tracking, and specific 
# libraries for single-cell RNA sequencing data and C2S operations.

# Python built-in libraries
import os
import pickle
import random
from collections import Counter
import logging

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data


def main():
    # Configure logging with the format: LEVEL - MESSAGE - DATETIME
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s - %(asctime)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)

    # Load Data
    # 
    # Next, we will load the preprocessed dataset from the tutorial 0. This dataset has already 
    # been filtered and normalized, so it it ready for transformation into cell sentences.
    # 
    # Please make sure you have completed the preprocessing steps in Tutorial 0 before running 
    # the following code, if you are using your own dataset. Ensure that the file path is 
    # correctly set in DATA_PATH to where your preprocessed data was saved from tutorial 0.

    DATA_PATH = "data/dominguez_conde/dominguez_conde_immune_tissue_two_donors_preprocessed.h5ad"

    adata = anndata.read_h5ad(DATA_PATH)
    logger.info(f"Loaded AnnData object: {adata}")

    adata.obs = adata.obs[["cell_type", "tissue", "batch_condition", "organism", "sex"]]

    logger.info("adata.obs.head():")
    logger.info(f"\n{adata.obs.head()}")

    logger.info("adata.var.head():")
    logger.info(f"\n{adata.var.head()}")

    sc.pl.umap(
        adata,
        color="cell_type",
        size=8,
        title="Human Immune Tissue UMAP",
    )

    logger.info(f"adata.X.max(): {adata.X.max()}")

    # We are expecting log10 base 10 transformed data, with a maximum value somewhere around 3 or 4. 
    # Make sure to start with processed and normalized data when doing the cell sentence conversion!

    # Cell2Sentence Conversion
    # 
    # In this section, we will transform our AnnData object containing our single-cell dataset 
    # into a Cell2Sentence (C2S) dataset by calling the functions of the CSData class in the C2S 
    # code base. Full documentation for the functions of the CSData class can be found in the 
    # documentation page of C2S.

    adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism", "sex"]

    # Create CSData object
    arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
        adata=adata, 
        random_state=SEED, 
        sentence_delimiter=' ',
        label_col_names=adata_obs_cols_to_keep
    )

    logger.info(f"Arrow dataset: {arrow_ds}")

    sample_idx = 0
    logger.info(f"arrow_ds[sample_idx]: {arrow_ds[sample_idx]}")

    # This time, we will leave off creating our CSData object until after we load our C2S model. 
    # This is because along with the model checkpoint, we saved the indices of train, val, and test 
    # set cells, which will allow us to select out test set cells for inference.

    # Load C2S Model
    # 
    # Now, we will load a C2S model with which we will do cell type annotation. For this tutorial, 
    # this model will be the last checkpoint of the training session from tutorial 3, where we 
    # finetuned our cell type prediction model to do cell type prediction specifically on our 
    # immune tissue dataset. We will load the last checkpoint saved from training, and specify 
    # the same save_dir as we used before during training.
    # 
    # Note: If you are using your own data for this tutorial, make sure to switch out to the model 
    # checkpoint which you saved in tutorial 3.
    # If you want to annotate cell types without finetuning your own C2S model, then tutorial 6 
    # demonstrates how to load the C2S-Pythia-410M cell type prediction foundation model and use 
    # it to predict cell types without any finetuning.
    # 
    # We can define our CSModel object with our pretrained cell type prediction model as follows, 
    # specifying the same save_dir as we used in tutorial 3:

    # Find the most recent training checkpoint from tutorial 3
    tutorial_3_experiment_dir = "experiments/tutorial_3"
    
    # Look for the most recent training run directory
    training_dirs = []
    if os.path.exists(tutorial_3_experiment_dir):
        for item in os.listdir(tutorial_3_experiment_dir):
            item_path = os.path.join(tutorial_3_experiment_dir, item)
            if os.path.isdir(item_path) and "finetune_cell_type_prediction" in item:
                training_dirs.append(item_path)
    
    if not training_dirs:
        raise FileNotFoundError(
            "No training checkpoints found from tutorial 3. Please run tutorial 3 first to generate "
            "a finetuned model, or modify the cell_type_prediction_model_path to point to your model."
        )
    
    # Use the most recent training directory
    most_recent_training_dir = sorted(training_dirs)[-1]
    logger.info(f"Using training directory: {most_recent_training_dir}")
    
    # Find the best checkpoint in the training directory
    checkpoint_dirs = []
    for item in os.listdir(most_recent_training_dir):
        item_path = os.path.join(most_recent_training_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            checkpoint_dirs.append(item_path)
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {most_recent_training_dir}")
    
    # Use the highest numbered checkpoint (typically the best/final one)
    cell_type_prediction_model_path = sorted(checkpoint_dirs)[-1]
    logger.info(f"Using model checkpoint: {cell_type_prediction_model_path}")

    # Define CSModel object
    save_dir = "experiments/tutorial_4"
    save_name = "cell_type_pred_pythia_410M_inference"

    csmodel = cs.CSModel(
        model_name_or_path=cell_type_prediction_model_path,
        save_dir=save_dir,
        save_name=save_name
    )

    logger.info(f"CSModel object: {csmodel}")

    # We will also load the data split indices saved alongside the C2S model checkpoint, so that 
    # we know which cells were part of the training and validation set. We will do inference on 
    # unseen test set cells, which are 10% of the original data.

    base_path = "/".join(cell_type_prediction_model_path.split("/")[:-1])
    logger.info(f"Model base path: {base_path}")

    data_split_indices_path = os.path.join(base_path, 'data_split_indices_dict.pkl')
    with open(data_split_indices_path, 'rb') as f:
        data_split_indices_dict = pickle.load(f)
    
    logger.info(f"Data split keys: {data_split_indices_dict.keys()}")
    logger.info(f"Train samples: {len(data_split_indices_dict['train'])}")
    logger.info(f"Validation samples: {len(data_split_indices_dict['val'])}")
    logger.info(f"Test samples: {len(data_split_indices_dict['test'])}")

    # Select out test set cells from full arrow dataset
    logger.info(f"Full arrow dataset: {arrow_ds}")

    test_ds = arrow_ds.select(data_split_indices_dict["test"])
    logger.info(f"Test dataset: {test_ds}")

    # Now, we will create our CSData object using only the test set cells:

    c2s_save_dir = "experiments/tutorial_4"  # C2S dataset will be saved into this directory
    c2s_save_name = "dominguez_immune_tissue_tutorial4"  # This will be the name of our C2S dataset on disk

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=test_ds, 
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name,
        dataset_backend="arrow"
    )

    logger.info(f"CSData object: {csdata}")

    # Predict cell types
    # 
    # Now that we have loaded our finetuned cell type prediction model and have our test set, we will 
    # do cell type prediction inference using our C2S model. We can use the function 
    # predict_cell_types_of_data() from the tasks.py, which will take a CSModel() object and apply 
    # it to do cell type prediction on a CSData() object.

    logger.info("Starting cell type prediction inference...")

    predicted_cell_types = predict_cell_types_of_data(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=200
    )

    logger.info(f"Predicted {len(predicted_cell_types)} cell types")
    logger.info(f"First 3 predictions: {predicted_cell_types[:3]}")

    # Calculate accuracy
    total_correct = 0.0
    for model_pred, gt_label in zip(predicted_cell_types, test_ds["cell_type"]):
        # C2S might predict a period at the end of the cell type, which we remove
        if model_pred[-1] == ".":
            model_pred = model_pred[:-1]
        
        if model_pred == gt_label:
            total_correct += 1

    accuracy = total_correct / len(predicted_cell_types)
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Show some example predictions
    logger.info("Example predictions:")
    for idx in range(0, min(100, len(predicted_cell_types)), 10):
        logger.info(f"Model pred: {predicted_cell_types[idx]}, GT label: {test_ds[idx]['cell_type']}")

    logger.info(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    logger.info("Cell type prediction completed successfully!")


# The model can achieve high accuracy, correctly predicting the cell type of unseen cells from 
# the immune tissue data! The model learned to predict cell type annotations in natural language 
# effectively from a short finetuning period on the new data.

if __name__ == "__main__":
    main()