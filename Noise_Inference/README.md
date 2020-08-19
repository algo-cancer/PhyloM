
# Training and Evaluation
Training and evaluation can be performed using the **main.py** file. Using the following command the list of arguments and their meaning for running this file can be seen. The examples for how to run training and evaluation, please refer to the bottom of the page.

```console
foo@bar:~$ python main.py --help
usage: main.py [-h] [--batch_size BATCH_SIZE] [--nCells NCELLS]
               [--nMuts NMUTS] [--output_dir OUTPUT_DIR] [--h5_dir H5_DIR]
               [--ms_dir MS_DIR] [--nTrain NTRAIN] [--nTest NTEST]
               [--alpha ALPHA] [--beta BETA] [--nb_epoch NB_EPOCH]
               [--lexSort LEXSORT]

Configuration file

optional arguments:
  -h, --help            show this help message and exit

Data:
  --batch_size BATCH_SIZE
                        batch size
  --nCells NCELLS       number of cells
  --nMuts NMUTS         number of mutations
  --output_dir OUTPUT_DIR
                        output directory
  --h5_dir H5_DIR       Data in h5 format directory
  --ms_dir MS_DIR       MS program directory
  --nTrain NTRAIN       number of train instances
  --nTest NTEST         number of test instances
  --alpha ALPHA         False positive rate
  --beta BETA           False negative rate

Training:
  --nb_epoch NB_EPOCH   nb epoch
  --lexSort LEXSORT     sort the column of the matrices lexicographically
```
# Example
Data construction, training and evaluation can be done using the following command.
```console
foo@bar:~$ python main.py --nCells=10 --nMuts=10 --output_dir=out --h5_dir=out1 --nTrain=1000 --nTest=100 --alpha=0.00001 --beta=0.1 --nb_epoch=100 --ms_dir=msdir --lexSort=True
```
