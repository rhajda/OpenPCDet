### Execution Suggestion:
Use a docker-compose.yml to start a docker container with all the necessary mounts (e.g. workdir or data one wants to use for training)

### Handling Paths:
To avoid using absolute paths which make development increasingly difficult when collaborating the file path_handle.py is used. This method is inspired by the singleton desing pattern to make pathmanagement easier. Just use     


### Mount Data
The data from "davs://webdisk.ads.mwn.de/hcwebdav/TUMW/ftm/Roborace/13_Vegas_Challenge/03_Data/02_Real/00_extracted_data/for_training/" to "data/indy" within the OpenPcDet working directory.  

### Setting the config files:
Everything is done as with the original, only that one entry called "DATA_PATH_REAL" is added and needs to be set. This should be used such that the original "DATA_PATH" containes the simulated data and "DATA_PATH_REAL" the real data. 