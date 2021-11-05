# DA_vis

## Dataset
The dataset used for this work is publicly available in The Cancer Imaging Archive (TCIA):

Shapey, J., Kujawa, A., Dorent, R., Wang, G., Bisdas, S., Dimitriadis, A., Grishchuck, D., Paddick, I., Kitchen, N., Bradford, R., Saeed, S., Ourselin, S., & Vercauteren, T. (2021). Segmentation of Vestibular Schwannoma from Magnetic Resonance Imaging: An Open Annotated Dataset and Baseline Algorithm [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.9YTJ-5Q73

### How to extract and prepare dataset
Follow instructions from [DA for brain VS segmentation](https://github.com/CarolineMagg/DA_brain/blob/main/README.md).
Note: perform all the steps in Section Dataset.

### Relevant Data split

The split training/test split was 80:20 and the training data is split again into 80:20. The relevant portion of the split is the test set which is used for the evaluation and visualization:

| Dataset    | # samples | numbers (excl.)          |
| ---------- |:---------:| ------------------------:|
| test       | 48        | 200 - 250 (208,219,227)  |

### Prepare results & data

For each patient id, an evaluation.json file is prepared by running `PrepareTestSet`. TODO: describe more.

### Collect results

The class `TestSet` is used to load the relevant information for the evaluation of different networks (DataContainer for patient data folders, evaluation files per patient folder, evaluation_all.json file if available). In order to reduce loading time at the start of the visualization application, `TestSet` is also able to pre-generate a collection of processed error score values and load them from the file evaluation_all.json. 

## Docker & Requirements
You can use [Docker](https://www.docker.com/) to setup your environment. For installation guide see [Install Docker](https://docs.docker.com/get-docker/). <br> 

The docker image contains (minimal requirements):
* Python 3.6
* Tensorflow 2.4.1 
* Jupyter notebooks
* Pycharm (installer.tgz)
* Dash 2.0.0
* Plotly 5.3.1

### How to build the docker image:
1. clone the github repository 
2. go to Dockerfile: ``` cd dockerfile/ ```
3. change Dockerfile to use your user name instead of *caroline* 
4. either download pycharm and store in installer.tgz or remove corresponding part in Dockerfile
5. Build docker image: ``` docker build --tag python:2.00 .``` 

### How to run the docker container:
(Note: Change *user* to your home folder name or */home/user* to the source folder name that you want to map to /tf/workdir/.)
* Run jupyter docker container: <br>
``` docker run -it --gpus all --name jupyter_notebook --rm -v /home/user/:/tf/workdir -p 8888:8888 python:2.00 ``` <br>

Recommended: modify sh files to fit your settings
