# DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding
Pytorch re-implementation of [DiSAN: Directional Self-Attention Network for RNN/CNN-Free Language Understanding](https://arxiv.org/abs/1709.04696).

This is an unofficial implementation. There is [the implementation by the authors](https://github.com/taoshen58/DiSAN), which is implemented on Tensorflow.

## Results
Dataset: [SNLI](https://nlp.stanford.edu/projects/snli/)

| Model | Valid Acc(%) | Test Acc(%)
| ----- | ------------ | -----------
| Baseline from the paper | - | - |
| Re-implemenation | - | - |

## Development Environment
- OS: Ubuntu 16.04 LTS (64bit)
- Language: Python 3.6.6
- Pytorch: 0.4.0

## Requirements
Please install the following library requirements first.

    nltk==3.3
    tensorboardX==1.2
    torch==0.4.0
    torchtext==0.2.3
    
## Training
> python train.py --help
        usage: train.py [-h] [--batch-size BATCH_SIZE] [--data-type DATA_TYPE]
                        [--dropout DROPOUT] [--epoch EPOCH] [--gpu GPU]
                        [--learning-rate LEARNING_RATE] [--print-freq PRINT_FREQ]
                        [--weight-decay WEIGHT_DECAY] [--word-dim WORD_DIM]
                        [--d-e D_E] [--d-h D_H]

        optional arguments:
          -h, --help            show this help message and exit
          --batch-size BATCH_SIZE
          --data-type DATA_TYPE
          --dropout DROPOUT
          --epoch EPOCH
          --gpu GPU
          --learning-rate LEARNING_RATE
          --print-freq PRINT_FREQ
          --weight-decay WEIGHT_DECAY
          --word-dim WORD_DIM
          --d-e D_E
          --d-h D_H


**Note:** 
- Only codes to use SNLI as training data are implemented.
