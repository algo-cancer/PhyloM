
The tensoflorw implementation can be found under Tensorflow folder. The original code is from [here](https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow) that we adapted to our problem.

The keras implementation can be found under Keras folder.

# Training and Evaluation
In both implementations, data construction, training, and evaluation can be performed using the **main.py** file. Using the following command the list of tunable arguments for running this file can be found. To see the examples for using the arguments, please refer to the bottom of the page.

```console
foo@bar:~$ python main.py --help

usage: main.py [-h] [--hidden_dim HIDDEN_DIM] [--batch_size BATCH_SIZE]
               [--input_dimension INPUT_DIMENSION] [--nCells NCELLS]
               [--nMuts NMUTS] [--output_dir OUTPUT_DIR]
               [--nTestMats NTESTMATS] [--alpha ALPHA] [--beta BETA]
               [--gamma GAMMA] [--nb_epoch NB_EPOCH]
               [--inference_mode INFERENCE_MODE]
               [--restore_model RESTORE_MODEL] [--save_to SAVE_TO]
               [--restore_from RESTORE_FROM] [--ms_dir MS_DIR]

Configuration file

optional arguments:
  -h, --help            show this help message and exit

Network:
  --hidden_dim HIDDEN_DIM
                        actor LSTM num_neurons

Data:
  --batch_size BATCH_SIZE
                        batch size
  --input_dimension INPUT_DIMENSION
                        city dimension
  --nCells NCELLS       number of cells
  --nMuts NMUTS         number of mutations
  --output_dir OUTPUT_DIR
                        output matrices directory
  --nTestMats NTESTMATS
                        number of test instances
  --alpha ALPHA         False positive rate
  --beta BETA           False negative rate
  --gamma GAMMA         hyperparameter in the cost function

Training:
  --nb_epoch NB_EPOCH   nb epoch

User options:
  --inference_mode INFERENCE_MODE
                        switch to inference mode when model is trained
  --restore_model RESTORE_MODEL
                        whether or not model is retrieved
  --save_to SAVE_TO     saver sub directory
  --restore_from RESTORE_FROM
                        loader sub directory
  --ms_dir MS_DIR       ms program directory
```

## Training example
An example of how to run training is as follows.
```console
foo@bar:~$ python main.py --inference_mode=False --nb_epoch=10 --batch_size=128 --nCells=10 --nMuts=10 --save_to=out --ms_dir=msdir
```

## Evaluation example
 Using the following command a model based on the specified number of cells and mutations will be compiled, and the stored weights will be imported to the model. The program will simulate a number of evaluation instances and provide the solution it can find. An example of running the evaluation is as follows.

```console
foo@bar:~$ python main.py --inference_mode=True --nTestMats=10 --batch_size=128 --nCells=10 --nMuts=10 --restore_from=out --ms_dir=msdir --output_dir=out1
```
## Reproducibility
We provided our weights for trained model on 10x10 size matrices under folder Noise_Elimination/Model. For reproducing the results, these weights can be used in the evaluation phase.
