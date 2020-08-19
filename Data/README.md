
You can download all of the datasets used in this paper [here](https://drive.google.com/drive/folders/1t4dSBmS4yjezMIcIPeC1KHfOguji_OXx?usp=sharing).

## Real datasets
We evaluated our branching inference approach on three real datasets.

1. Triple Negative Breast Cancer ([TNBC](https://drive.google.com/drive/folders/1MVSuXJ64lYlzLjW6LIHBxG2s1vzqU35-?usp=sharing)): this dataset includes 16 single cells and 18 mutations [1].
2.   Acute  Lymphoblastic  Leukemia  ([ALL](https://drive.google.com/drive/folders/1sZfVxRnBKyNLoOWReUG2CeO1wF819UbO?usp=sharing))  Patient  6: this dataset consists of 146 single cells and 10 mutations [2].
3. Human Colorectal Cancer ([CRC](https://drive.google.com/drive/folders/1RMP104_HobO0ZhpxvYeCrW42mZyRfquz?usp=sharing)) Patient 1: this dataset includes single cells from two sites of the patient body; 133 single cells from colon as primary tumor site and 45 single cells from liver as the tumor metastatic site. We remove duplicated cells in this dataset before feeding it to our neural network. After duplication removal, the number of unique cells are 36 and 30 from primary and metastatic sites, respectively [3].

## Simulated datasets


For simulating a dataset, the ms program is required. You can download the program and its instruction [here](https://uchicago.app.box.com/s/l3e5uf13tikfjm7e1il1eujitlsjdx13). This program should be compiled based on its own instruction.

For either of [noise inference](https://github.com/algo-cancer/PhyloM/tree/master/Noise_Inference) and [noise elimination](https://github.com/algo-cancer/PhyloM/tree/master/Noise_Elimination) problems, by running the main.py file and providing the compiled ms program directory, the dataset will be constructed on the fly before running training or evaluation. For the [branching inference](https://github.com/algo-cancer/PhyloM/tree/master/Branching_Inference) problem, the same procedure will be done using the train.py file. Note that in this file the ms program directory should be set inside the script. For more details please see the readme for each problem.

We also provided two of our simulated datasets under [Simulated](https://drive.google.com/drive/folders/1KMKRdXAi0A3Y6X2el1F8u17HhOhsFN-A?usp=sharing) folder.

1. [Simulated dataset](https://drive.google.com/drive/folders/1nT1fMuFDVzCXiBIxvnQ3BNZUtJ9cH-Ak?usp=sharing) with false negative and false positive rates of 0.1 and 0.002, respectively. This dataset is provided in the format of four .h5 file:

  | File name                      | Number of matrices | Lexicographically sorted  |
  |:------------------------------:|:------------------:| :------------------------:|
  | Train_002_1_10x10.h5           | 2M                 | No                        |
  | Test_002_1_10x10.h5            | 2K                 |   No                      |
  | Train_002_1_10x10_lexSorted.h5 | 2M                 |    Yes                    |
  |Test_002_1_10x10_lexSorted.h5   | 2K                 |      Yes                  |

Note that in each file half of the matrices are perfect phylogeny matrices and the other half are their corresponding noisy matrices (which contain at least one conflict).

2. [Simulated dataset](https://drive.google.com/drive/folders/1Qwx9h3TN2DTOftq9BMUrf1djj_dSycQw?usp=sharing) with false negative and false positive rates of 0.02 and 0.0004, respectively. Similar to the first simulated dataset, this dataset is also consisted of four .h5 files:

  | File name                      | Number of matrices | Lexicographically sorted  |
  |:------------------------------:|:------------------:| :------------------------:|
  | Train_0004_02_10x10.h5           | 2M                 | No                        |
  | Test_0004_02_10x10.h5            | 2K                 |   No                      |
  | Train_0004_02_10x10_lexSorted.h5 | 2M                 |    Yes                    |
  |Test_0004_02_10x10_lexSorted.h5   | 2K                 |      Yes                  |

Note that in this dataset also half of the matrices in each file are perfect phylogeny matrices and the other half are their corresponding noisy matrices.

## References

[1] Y. Wang, J. Waters, M. L. Leung, A. Unruh, W. Roh, X. Shi, K. Chen, P. Scheet, S. Vattathil, H. Liang, et al. [Clonal evolution in breast cancer revealed by single nucleus genome sequencing](https://www.nature.com/articles/nature13600). Nature, 512(7513):155–160, 2014.

[2] C. Gawad, W. Koh, and S. R. Quake. [Dissecting the clonal origins of childhood acute lym-phoblastic leukemia by single-cell genomics](https://www.pnas.org/content/111/50/17947). Proceedings of the National Academy of Sciences, 111(50):17947–17952, 2014.

[3] M. L. Leung, A. Davis, R. Gao, A. Casasent, Y. Wang, E. Sei, E. Vilar, D. Maru, S. Kopetz, and N. E. Navin. [Single-cell dna sequencing reveals a late-dissemination model in metastatic colorectal cancer](https://genome.cshlp.org/content/27/8/1287.short). Genome research, 27(8):1287–1299, 2017.
