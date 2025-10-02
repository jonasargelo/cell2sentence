"""
Tutorial 3: Finetuning on a New Single-Cell Dataset

In this tutorial, we will demonstrate how to fine-tune an existing Cell2Sentence (C2S) model 
on a new single-cell RNA sequencing dataset. Fine-tuning is a crucial step in adapting a 
pretrained model to perform well on a specific task or dataset, improving its accuracy and 
generalization. This tutorial will guide you through the process of fine-tuning a C2S model 
to perform cell type prediction on a new dataset.

In this tutorial, you will:
1. Load an immune tissue single-cell dataset from Domínguez Conde et al. (preprocessed in 
   tutorial notebook 0, two sample donors)
   - Citation: Domínguez Conde, C., et al. "Cross-tissue immune cell analysis reveals 
     tissue-specific features in humans." Science 376.6594 (2022): eabl5197.
2. Format the dataset using a Prompt Formatter object, which prepares the data for the 
   fine-tuning process.
3. Load a pretrained C2S model.
4. Fine-tune the C2S model to improve its performance on cell type prediction.
"""

# We will begin by importing the necessary libraries. These include Python's built-in libraries, 
# third-party libraries for handling numerical computations, progress tracking, and specific 
# libraries for single-cell RNA sequencing data and C2S operations.

# Python built-in libraries
import os
from datetime import datetime
import random
from collections import Counter
import logging

# Third-party libraries
import numpy as np
from tqdm import tqdm
from transformers import TrainingArguments

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs

# Check GPU availability
import torch


def main():
    # Configure logging with the format: LEVEL - MESSAGE - DATETIME
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s - %(asctime)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    # Log GPU information
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA is available. Total GPUs detected: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Log CUDA_VISIBLE_DEVICES setting
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        logger.info(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
        
        # Log GPU memory status
        for i in range(gpu_count):
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i} Total Memory: {memory_total:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will use CPU.")
    
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

    # Cell2Sentence Conversion + CSData Creation
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

    sample_idx = 0
    logger.info(f"arrow_ds[sample_idx]: {arrow_ds[sample_idx]}")

    c2s_save_dir = "data/dominguez_conde"  # C2S dataset will be saved into this directory
    c2s_save_name = "dominguez_conde_immune_tissue_two_donors_c2s_t3"  # This will be the name of our C2S dataset on disk

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds, 
        vocabulary=vocabulary,
        save_dir=c2s_save_dir,
        save_name=c2s_save_name,
        dataset_backend="arrow"
    )

    logger.info(f"CSData object: {csdata}")

    # Load C2S Model
    # 
    # Now, we will load a C2S model which will finetune on a new dataset. This model can be a LLM 
    # pretrained on natural language, or it can be a trained C2S model which will undergo further 
    # finetuning on a new dataset of interest. Typically, starting from a pretrained C2S model 
    # benefits performance, since C2S models were initialized from natural language-pretrained LLMs 
    # and trained on many single-cell datasets on different tasks.
    # 
    # For this tutorial, we will start finetuning from the C2S-Pythia-410M cell type prediction model, 
    # which was trained to do cell type prediction on many datasets from CellxGene and Human Cell Atlas. 
    # We will finetune it for cell type prediction on our immune tissue dataset which we have loaded, 
    # which will help align the model with the cell type annotations present in this dataset as well as 
    # the expression profiles of the cells in our two donor samples. More details about the C2S-Pythia-410M 
    # cell type prediction model can be found in the Model Zoo section of the ReadME in the GitHub repo, 
    # or in the Huggingface model card.
    # 
    # We can define our CSModel object with our pretrained cell type prediction model as follows:

    # Define CSModel object
    cell_type_prediction_model_path = "models/C2S-Pythia-410m-cell-type-prediction"
    save_dir = "experiments/tutorial_3"
    save_name = "cell_embedding_prediction_pythia_410M_2"

    csmodel = cs.CSModel(
        model_name_or_path=cell_type_prediction_model_path,
        save_dir=save_dir,
        save_name=save_name
    )

    # Note that the `model_name_or_path` parameter can be a name of a Huggingface model, for example 
    # 'EleutherAI/pythia-410m' for a 410 million parameter Pythia model pretrained on natural language 
    # (see https://huggingface.co/EleutherAI/pythia-410m), or it can be the path to a pretrained model 
    # saved on disk, as in the case in the cell above.

    logger.info(f"CSModel object: {csmodel}")

    # Finetune on new dataset
    # 
    # Now, we will finetune our loaded C2S model on our immune tissue dataset. For training, we will 
    # need to define training arguments for finetuning our C2S model on our new dataset. Huggingface's 
    # Trainer class is used to do training, so we can utilize different training techniques (e.g. mixed 
    # precision training, gradient accumulation, gradient checkpointing, etc.) by specifying the 
    # corresponding option in the TrainingArguments object. This gives us a vast array of possible 
    # options for training, and will allow us to specify important parameters such as batch size, 
    # learning rate, and learning rate schedulers. See the full documentation for training arguments at:
    # - https://huggingface.co/docs/transformers/en/main_classes/trainer

    # First, we define our training task, which in our case will be cell type prediction. Possible 
    # values for the training task parameter can be found in the `prompt_formatter.py` file in the 
    # source code, under `SUPPORTED_TASKS`.

    training_task = "cell_type_prediction"

    # We will create a datetimestamp to mark our training session:

    datetimestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    output_dir = os.path.join(csmodel.save_dir, datetimestamp + f"_finetune_{training_task}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # And here, we define our training arguments. For this tutorial, we will use a batch size of 8 
    # with 4 gradient accumulation steps, yielding an effective batch size of 32. We will use a 
    # learning rate of 1e-5 with a cosine annealing scheduler, and we will train for 5 epochs total. 
    # Some other important parameters specified here are:
    # - bf16: Uses mixed-precision training with bfloat16 dtype
    # - logging_steps: controls how often we log training loss
    # - eval_steps: controls how often we run the eval loop
    # - warmup_ratio: percentage of training in which learning rate warms up to the base learning rate specified
    # 
    # Full explanations of all possible training arguments can be found in the Huggingface Trainer documentation: 
    # https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/trainer#transformers.TrainingArguments

    train_args = TrainingArguments(
        bf16=True,
        fp16=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        learning_rate=1e-5,
        load_best_model_at_end=True,
        logging_steps=50,
        logging_strategy="steps",
        lr_scheduler_type="cosine",
        num_train_epochs=5, 
        eval_steps=50,
        evaluation_strategy="steps",
        save_steps=100,
        save_strategy="steps",
        save_total_limit=3,
        warmup_ratio=0.05,
        output_dir=output_dir,
        dataloader_pin_memory=False,  # Can help with multi-GPU setups
        remove_unused_columns=False,  # Sometimes needed for custom datasets
    )
    
    logger.info(f"Training arguments configured. Using {torch.cuda.device_count() if torch.cuda.is_available() else 0} GPUs for training.")

    csmodel.fine_tune(
        csdata=csdata,
        task=training_task,
        train_args=train_args,
        loss_on_response_only=False,
        top_k_genes=200,
        max_eval_samples=500,
    )

# Our trained models are now saved in the output directory we specified in the training arguments. 
# Huggingface will save the latest checkpoints of the training session, and will also keep the 
# checkpoint which has the lowest validation loss.
# 
# In the next tutorial notebook (tutorial 4), we will see how to run cell type prediction inference 
# with our trained model.

if __name__ == "__main__":
    main()