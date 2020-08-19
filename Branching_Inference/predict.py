from util import *

parser = argparse.ArgumentParser(description='Run a trained model on an input matrix')
parser.add_argument('--modelfile', '-m',  type=str, help='model file name')
parser.add_argument('--inputfile', '-i',  type=str, help='input file name')

args = parser.parse_args()


def run_once():
  matrix_file_address = f"Data/{args.i}"
  matrix = np.loadtxt(matrix_file_address, dtype = np.int8)
  print(matrix.shape)

  X = matrix.copy()
  modelAddress = os.path.join(config.modelsDir, args.modelfile)
  model = load_model(modelAddress)
  ypredict = model.predict(X.reshape(1, -1))

  print(ypredict)
  print(X.shape)


if __name__ == '__main__':
  import __main__
  print(f"{__main__.__file__} starts here")

  run_once()