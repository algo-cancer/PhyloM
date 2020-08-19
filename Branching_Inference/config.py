import configparser
import getpass  # For the username running the program
import inspect
import logging  # For logging
import os
import platform  # For the name of host machine
import time

projectDir = "."
dataDir = "."
tempDir = "."
msAddress = "../../external/ms"
historiesDir = None
modelsDir = None
h5dataDir = None
outputsDir = None
verbose = True


def mk_dir_if_not_exists(folder_address):
  starting_with_slash = folder_address[0] == "/"
  chain = folder_address.split("/")
  folder_address = "/" if starting_with_slash else ""
  changed = False
  for newPart in chain:
    folder_address = os.path.join(folder_address, newPart)
    assert isinstance(folder_address, str)
    if len(folder_address) > 0 and not os.path.exists(folder_address):
      os.mkdir(folder_address)
      changed = True
  return changed


userName = getpass.getuser()
platformName = platform.node()
print(f"Platform: {userName}@{platformName}")

projectDir = "."
# msAddress = f"{projectDir}/external/ms"

dataDir = f"{projectDir}/Data"
tempDir = f"{projectDir}/Temp"
historiesDir = f"{projectDir}/Histories"
modelsDir = f"{projectDir}/Models"
h5dataDir = f"{projectDir}/H5data"
outputsDir = f"{projectDir}/Outputs"



constants = configparser.ConfigParser()


constants.dataDir = dataDir
constants.tempDir = tempDir
constants.historiesDir = historiesDir
constants.modelsDir = modelsDir
constants.h5dataDir = h5dataDir
constants.outputsDir = outputsDir




logger = logging.getLogger('root')
loggingFileName = f"{time.time()}.log"
loggingFileAddress = f"{outputsDir}/{loggingFileName}"
print(f"Logs for this run will be stored at {loggingFileAddress}", flush=True)
shortFormatter = logging.Formatter('%(levelname)s: %(message)s')
# longFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

consoleHandler = logging.StreamHandler()
if verbose:
  consoleHandler.setLevel(logging.DEBUG)
else:
  consoleHandler.setLevel(logging.ERROR)
consoleHandler.setFormatter(shortFormatter)
logger.addHandler(consoleHandler)

fho = logging.FileHandler(filename=loggingFileAddress, mode='w')
fho.setLevel(logging.DEBUG)
logger.addHandler(fho)


logger.setLevel(logging.DEBUG)


for folder_address in [dataDir, tempDir, historiesDir, modelsDir, h5dataDir, outputsDir,]:
  changed = mk_dir_if_not_exists(folder_address)
  if changed:
    logger.info(f"Folder with name {folder_address} is made!")
