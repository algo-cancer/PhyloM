import config
from util import *

parser = argparse.ArgumentParser(description='Run an experiment for evaluating noise-tolerance')
parser.add_argument('--modelfile', '-m',  type=str, help='model file name')


args = parser.parse_args()


def run_experiment():
  n_cells, n_muts = 4, 4
  dataset_size = 100
  rows_dict = []
  rates = [0, 0]
  n_datapoint = 50
  factor = 1

  modelAddress = os.path.join(config.modelsDir, args.modelfile)
  model = load_model(modelAddress)

  ##### this is for activating cache
  data_set = make_dataset(n_cells, n_muts, 1)
  ypredict = model.predict(data_set[0])
  ###########

  for row_ind in tqdm(range(50)):
    rates[1] = 0.35 * row_ind / n_datapoint
    rates[0] = rates[1] * factor

    data_set = make_dataset(n_cells, n_muts, dataset_size)
    X = data_set[0]
    y = data_set[1]
    X_clean, y = shuffle(X, y)
    n = len(y)

    X = make_noisy(X_clean, n_cells, n_muts, rates=rates)

    DL_time = time.time()
    ypredict = model.predict(X)
    DL_time = time.time() - DL_time
    predicted_labels = np.argmax(ypredict, axis=1)
    original_labels = y[:, 1]

    n_correct = np.sum(predicted_labels == original_labels)
    accuracy = n_correct / n
    row = {
      "DL_time": DL_time,
      "n_correct": n_correct,
      "accuracy": accuracy,
      "fp_rate": rates[0],
      "fn_rate": rates[1],
    }
    rows_dict.append(row)

  df = pd.DataFrame(rows_dict)
  csv_file_name = os.path.join(config.outputsDir, f"output_{time.time()}.csv")
  df.to_csv(csv_file_name)
  print(f"csvFile is saved to {csv_file_name}")


if __name__ == '__main__':
  import __main__
  print(f"{__main__.__file__} starts here")
  run_experiment()
