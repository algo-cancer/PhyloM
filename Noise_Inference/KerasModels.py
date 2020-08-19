
import tensorflow as tf
from tensorflow import keras


def addDenseLayerClassification(modelsList,
                                inputDim ,
                                nameSuffix="0",
                                hiddenSize=10, nLayers = 3,
                                dropOutRate=0.8, dropOutFirst=True, dropOutAfterFirst=True,
                                activation = "sigmoid",
                                useSoftmax = True
                                ):
  if isinstance(hiddenSize, list):
    for arg in hiddenSize:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix,
                                  arg,
                                  nLayers, dropOutRate, dropOutFirst, dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(nLayers, list):
    for arg in nLayers:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix, hiddenSize,
                                  arg,
                                  dropOutRate, dropOutFirst, dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(dropOutRate, list):
    for arg in dropOutRate:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers,
                                  arg,
                                  dropOutFirst, dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(dropOutFirst, list):
    for arg in dropOutFirst:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers, dropOutRate,
                                  arg,
                                  dropOutAfterFirst, activation, useSoftmax)
  elif isinstance(dropOutAfterFirst, list):
    for arg in dropOutAfterFirst:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers, dropOutRate, dropOutFirst,
                                  arg,
                                  activation, useSoftmax)
  elif isinstance(activation, list):
    for arg in activation:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix, hiddenSize, nLayers,
                                  dropOutRate, dropOutFirst, dropOutAfterFirst,
                                  arg,
                                  useSoftmax)
  elif isinstance(useSoftmax, list):
    for arg in useSoftmax:
      addDenseLayerClassification(modelsList, inputDim, nameSuffix, hiddenSize,nLayers,
                                  dropOutRate, dropOutFirst, dropOutAfterFirst, activation,
                                  arg)
  else:
    assert (not dropOutAfterFirst) or dropOutRate != None
    name = "DenseDrOut_{nLayers}_{hiddenSize}_{activation}_{dropOutRate}_{dropOutFirst}_{dropOutAfterFirst}_{useSoftmax}_{nameSuffix}".format(nLayers = nLayers, \
      hiddenSize = hiddenSize, activation = activation, dropOutRate = dropOutRate, dropOutFirst = dropOutFirst,dropOutAfterFirst = dropOutAfterFirst, useSoftmax = useSoftmax, nameSuffix = nameSuffix)
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape = inputDim))

    if dropOutFirst == True:
      model.add(keras.layers.Dropout(dropOutRate))
    for i in range(nLayers):
      model.add(keras.layers.Dense(hiddenSize, activation = activation))
      if dropOutAfterFirst == True:
        model.add(keras.layers.Dropout(dropOutRate))

    if useSoftmax == True:
      model.add(keras.layers.Dense(2, activation='softmax'))
    else:
      model.add(keras.layers.Dense(1, activation=activation))
    model.compile(loss= 'binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    modelsList.append((name, model))


if __name__ == '__main__':
  import __main__
  print("{q} starts here".format(q = __main__.__file__))
  nCells = 10
  nMuts = 10
  models = []

  addDenseLayerClassification(models, nameSuffix="sf", inputDim = (nCells, nMuts),
                              hiddenSize=[100, 150, 200],
                              nLayers=2,
                              dropOutRate=[None, 0.9],
                              dropOutFirst=True,
                              dropOutAfterFirst=False,
                              activation="tanh",
                              useSoftmax=True)

  addDenseLayerClassification(models,  nameSuffix="sf", inputDim = (nCells, nMuts),
                              hiddenSize=[200, 300, 400, 500],
                              nLayers=2,
                              dropOutRate=[0.3, 0.4, 0.5, 0.6],
                              dropOutFirst=True,
                              dropOutAfterFirst=False,
                              activation="relu",
                              useSoftmax=True)

  for name, model in models:
    print(name)
  # saveModels(models)
  # printSummariesModels(models)
