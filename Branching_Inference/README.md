# Training
To train a model and reproduce plots similar to Fig. 2, use the following command:
```
python train.py -n 100 -m 100 -d 5000 -e 200 -s 100
```
This command will consider inputs to be 100 by 100. Then it will make a dataset with 5000 instances for each label. It will train a model with an architecture described in the paper and set the hidden layer's size to 100. The training will continute to 200 epochs.

To check the meaning of the arguments, use:
```
>> python train.py --help
usage: train.py [-h] [-n N] [-m M] [-d D] [-e E] [-s S]
optional arguments:
  -h, --help  show this help message and exit
  -n N        num of cells
  -m M        num of muts
  -d D        dataset size
  -e E        num epochs
  -s S        hidden size
```

After a successful run, a few files will be stored: a log file, history of accuracies and losses, the model, and csv file with summary of accuracy.

# Experiment with noise-tolerance
For learning the command  pattern, run:
```
>> python run_noisy.py --help
usage: run_noisy.py [-h] [--modelfile MODELFILE]

Run an experiment for evaluating noise-tolerance

optional arguments:
  -h, --help            show this help message and exit
  --modelfile MODELFILE, -m MODELFILE
                        model file name

```

For example, subsitute the model name printed from the training phase and run:
```
>> python run_noisy.py -m [MODEL FILENAME]
```
This will result in a csv file used in plotting Fig. 3.
You can also read the running times from the csv file and compare with the running time of [PhISCS](https://github.com/sfu-compbio/PhISCS).

# Predictions for a single file
Use the following command for running a given model on a single file:
```
python predict.py --help
usage: predict.py [-h] [--modelfile MODELFILE] [--inputfile INPUTFILE]
Run a trained model on an input matrix
optional arguments:
  -h, --help            show this help message and exit
  --modelfile MODELFILE, -m MODELFILE
                        model file name
  --inputfile INPUTFILE, -i INPUTFILE
                        input file name
```

For example,
```
python predict.py -m MODELFILE -i Gawad.SC.np
```
