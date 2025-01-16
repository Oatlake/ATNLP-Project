# ATNLP-Project: Transformer-Based SCAN Task

This repository is a reimplementation of experiments from the paper [*Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks*](https://arxiv.org/abs/1711.00350) by Brendan Lake and Marco Baroni. Instead of using RNNs, GRUs, or LSTMs as in the original paper, we implement a Transformer-based model inspired by the architecture proposed in the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al. Hereto we compare the resulsts to a pre-trained model: T5-small, which has been fine-tuned using LoRa.


## Introduction
The goal of this project is to evaluate the compositional generalization capabilities of Transformer models on the SCAN dataset. SCAN is a synthetic dataset that pairs commands (e.g., "walk twice and jump") with corresponding actions (e.g., "WALK WALK JUMP"). We test the ability of our model to generalize across three key splits:
- **Experiment 1**: Simple split with varying training data sizes.
- **Experiment 2**: Length-based split.
- **Experiment 3**: Compositional split.

This project is designed for educational purposes.

The Transformer model used in this repository follows the implementation done by Vaswani et al., incorporating multi-head self-attention, positional encodings, and feed-forward layers.


## Dependencies
The project is implemented in Python using PyTorch. To get started use Poetry to install the needed packages, along with the requirements.txt file.
```bash
pip install poetry

poetry install

pip install -r requirements.txt

```


## Code Structure
This repository contains the following components:
- **`dataset.py`**: A custom data loader for the SCAN dataset, including vocabulary creation and text-token transformations.
- **`model/transformer.py`**: An implementation of a Transformer-based sequence-to-sequence model, including multi-head attention, positional encodings, encoder-decoder layers, and mask generation.
- **`train.py`**: The training and evaluation script for the model + T5-small model.
- **`utils/utils.py`**: Greedy and Greedy Oracale decoding.
- **`experiments/`**: Training scripts for any of the experiments.
- **`data/`**: Directory for SCAN dataset files. Use the preprocessed dataset from [Transformer-SCAN](https://github.com/jlrussin/transformer_scan).
- **`model/`**: Directory for saving trained model checkpoints.


## Data Input
The SCAN dataset is sourced from the [Transformer-SCAN repository](https://github.com/jlrussin/transformer_scan). Each dataset split consists of text files with lines in the format:
```
IN: <COMMAND> OUT: <ACTION>
```
Example:
```
IN: jump thrice OUT: JUMP JUMP JUMP
IN: turn left twice OUT: LTURN LTURN
```

The dataset is tokenized, and both source (commands) and target (actions) vocabularies are built dynamically. Special tokens such as `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>` are used for training and evaluation.


## Usage
1. **Clone the repository**:
    ```bash
    git clone https://github.com/Oatlake/ATNLP-Project.git
    cd ATNLP-Project
    ```

2. **Download the SCAN dataset**:
    Place the dataset files in the `data/` directory. Example structure:
    ```
    data/
      length_split/
        tasks_train_length.txt
        tasks_test_length.txt
      simple_split/
        size_variations/
          tasks_train_simple_p1.txt
          tasks_test_simple_p1.txt
          ...
    ```

3. **Train the model**:
    Use any of the training scripts in the `experiments/` directory to start training the transformer model:
```bash
python -m experiments.train_exp_3
```

## Fine-tuning T5-small
   To fine-tune the T5-small model go to the experiment b files under the 'experiments/' repository. You can find the full T5-small and LoRa code in the 'train_exp_1b' file along with the script for running the fine-tuning on experiment 1
```bash
python -m experiments.train_exp_1b
```
```bash
python -m experiments.train_exp_2b
```
## Evaluation
### Metrics
The following metrics are used to evaluate model performance:
- **Token-level accuracy**: Measures the percentage of correct token predictions.
- **Sequence-level accuracy**: Measures whether the entire output sequence matches the target sequence.

### Example Output
```plaintext
Dataset p16 - Epoch: 10
Train Loss: 0.1234
Test Loss: 0.4567
Greedy Search Loss: 0.3456
Accuracy: 0.92, Sequence Accuracy: 0.53
```

## Results
We aim to reproduce and compare the following key findings:
- Performance of Transformers across dataset splits and sizes.
- Ability of Transformers to generalize to longer or unseen command sequences.
- The performance of a our standard transformer model versus a larger pre-trained model

## Visualisation
To get a visual representation of some of the results, run the visualisation script:
```bash
python -m visualisations
```

## Acknowledgments
This project is inspired by the experiments conducted in [Lake & Baroni, 2017](https://arxiv.org/abs/1711.00350). We thank [Transformer-SCAN](https://github.com/jlrussin/transformer_scan) for providing preprocessed SCAN datasets and code references. The Transformer model architecture is adapted from the foundational paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al.
