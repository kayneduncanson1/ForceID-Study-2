# ForceID-Study-2
Code repository - *Modelling individual variation in human walking gait across populations and
walking conditions via gait recognition*.

In: Journal of the Royal Society Interface.

By: Kayne A. Duncanson, Fabian Horst, Ehsan Abbasnejad, Gary Hanly, William S.P. Robertson, and Dominic Thewlis.
Corresponding author: KAD. Email: kayne.duncanson@adelaide.edu.au.

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## Data accessibility
Four datasets were used in this study: AIST, Gutenberg, GaitRec, and ForceID-A. The availability of these datasets is as follows:
- Gutenberg - Horst, Fabian; Slijepcevic, Djordje; Simak, Marvin; Schöllhorn, Wolfgang I. (2021). Gutenberg Gait Database: A ground reaction force database of level overground walking in healthy individuals. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.5311538.v1;
- GaitRec - Horsak, Brian; Slijepcevic, Djordje; Raberger, Anna-Maria; Schwab, Caterine; Worisch, Marianne; Zeppelzauer, Matthias (2020). GaitRec: A large-scale ground reaction force dataset of healthy and impaired gait. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.4788012.v1;
- ForceID-A - the version of the dataset used in this study is private due to participant consent. The public version is available at: Duncanson, Kayne; Thwaites, Simon; Booth, David; Hanly, Gary; Robertson, Will; Abbasnejad, Ehsan; et al. (2021). ForceID Dataset A. The University of Adelaide. Dataset. https://doi.org/10.25909/14482980.v6;
- AIST - since the dataset was accessed for this study, its distribution was suspended. Details are available at: Yoshiyuki Kobayashi, Naoto Hida, Kanako Nakajima, Masahiro Fujimoto, Masaaki Mochimaru, 2019: AIST Gait Database 2019. https://unit.aist.go.jp/harc/ExPART/GDB2019.html.

The version of GaitRec on figshare was applied directly in this study, so the files can be imported into the Datases/gr-all/spreadsheets subdirectory for use.
Gutenberg was refined for use in this study according to the procedure outlined in Supplementary Information - Section 1.1. The prepared version will be uploaded to figshare and the link will be added here.

## User guide
The datasets were individually prepared (i.e., pre-processed) for the machine learning gait recognition experiments
in this study using the 'prepare_...' scripts. At the same time, metadata was extracted from the datasets and saved in the
'Datasets/ds/objects' subdirectories. The demographic metadata was characterised in the 'fig_demographics' script that
shows the process for generating the demographic distribution figure in the manuscript.

The purposes of the remaining scripts are as follows:
- main - contains the code that was used to run the main experiments. Broadly, these experiments entailed configuring the datasets in different ways for machine learning gait recognition model training, validation, and testing.
- main_subset_executable - can be used to reproduce the results for dataset configurations that include Gutenberg or GaitRec. This can be run once the spreadsheet for GaitRec is imported and the spreadsheets for the refined version of Gutenberg *to be* uploaded to figshare are imported. This is currently a work in progress.
- performance_analysis - contains the code that was used to summarise and compare results across batch sizes and machine learning model architectures.
- umap_main - contains the code that was used to generate the UMAP figure in the manuscript.
- umap_supp - contains the code that was used to generate the UMAP figure in the Supplementary Information.
- osa - contains the code that was used to conduct the occlusion sensitivity analysis and generate associated figures in the manuscript and Supplementary Information.
- fig_overview - contains the code that was used to generate the study overview figure in the manuscript.

### Additional information
- For transparency, the Results directory contains certain anonymised results (e.g., accuracy measures from the main experiments and occlusion sensitivity analysis). Other results have been excluded for privacy reasons or due to their file size (e.g., can't be pushed to GitHub).
- Most scripts and packages are introduced via """string statements""" and thoroughly commented throughout. One notable exception is the TrainEval package which is yet to be commented.
- LICENSE.txt contains additional information on the software license.
- FIS2.yml contains the conda virtual environment that was used to complete the study.

For reference, the packages included in the interpreter are as follows:  
Name                      Version                   Build    Channel
_tflow_select             2.2.0                     eigen
absl-py                   0.15.0             pyhd3eb1b0_0
aiohttp                   3.8.1            py37h2bbff1b_1
aiosignal                 1.2.0              pyhd3eb1b0_0
appdirs                   1.4.4                    pypi_0    pypi
argon2-cffi               21.3.0             pyhd3eb1b0_0
argon2-cffi-bindings      21.2.0           py37h2bbff1b_0
astor                     0.8.1            py37haa95532_0
async-timeout             4.0.1              pyhd3eb1b0_0
asynctest                 0.13.0                     py_0
attrs                     21.4.0             pyhd3eb1b0_0
backcall                  0.2.0              pyhd3eb1b0_0
blas                      1.0                         mkl
bleach                    4.1.0              pyhd3eb1b0_0
blinker                   1.4              py37haa95532_0
bokeh                     2.4.3            py37haa95532_0
bottleneck                1.3.5            py37h080aedc_0
brotli                    1.0.9                h2bbff1b_7
brotli-bin                1.0.9                h2bbff1b_7
brotlipy                  0.7.0           py37h2bbff1b_1003
ca-certificates           2023.08.22           haa95532_0
cachetools                4.2.2              pyhd3eb1b0_0
certifi                   2022.12.7        py37haa95532_0
cffi                      1.15.1           py37h2bbff1b_0
chardet                   4.0.0           py37haa95532_1003
charset-normalizer        2.0.4              pyhd3eb1b0_0
click                     8.0.4            py37haa95532_0
cloudpickle               2.0.0              pyhd3eb1b0_0
colorama                  0.4.5            py37haa95532_0
colorcet                  3.0.1            py37haa95532_0
cryptography              37.0.1           py37h21b164f_0
cudatoolkit               11.3.1               h59b6b97_2
cycler                    0.11.0             pyhd3eb1b0_0
cytoolz                   0.11.0           py37he774522_0
dask-core                 2021.10.0          pyhd3eb1b0_0
datashader                0.14.4           py37haa95532_0
datashape                 0.5.4            py37haa95532_1
debugpy                   1.5.1            py37hd77b12b_0
decorator                 5.1.1              pyhd3eb1b0_0
defusedxml                0.7.1              pyhd3eb1b0_0
docker-pycreds            0.4.0                    pypi_0    pypi
entrypoints               0.4              py37haa95532_0
et_xmlfile                1.1.0            py37haa95532_0
fftw                      3.3.9                h2bbff1b_1
fonttools                 4.25.0             pyhd3eb1b0_0
freetype                  2.10.4               hd328e21_0
frozenlist                1.2.0            py37h2bbff1b_0
fsspec                    2022.7.1         py37haa95532_0
future                    0.18.2                   py37_1
gast                      0.2.2                    py37_0
gitdb                     4.0.11                   pypi_0    pypi
gitpython                 3.1.40                   pypi_0    pypi
google-auth               2.6.0              pyhd3eb1b0_0
google-auth-oauthlib      0.4.1                      py_2
google-pasta              0.2.0              pyhd3eb1b0_0
grpcio                    1.42.0           py37hc60d5dd_0
h5py                      3.7.0            py37h3de5c98_0
hdf5                      1.10.6               h1756f20_1
holoviews                 1.15.4           py37haa95532_0
icc_rt                    2022.1.0             h6049295_1
icu                       58.2                 ha925a31_3
idna                      3.3                pyhd3eb1b0_0
imageio                   2.19.3           py37haa95532_0
importlib-metadata        4.11.3           py37haa95532_0
importlib_metadata        4.11.3               hd3eb1b0_0
importlib_resources       5.2.0              pyhd3eb1b0_1
intel-openmp              2021.4.0          haa95532_3556
ipykernel                 6.15.2           py37haa95532_0
ipython                   7.31.1           py37haa95532_1
ipython_genutils          0.2.0              pyhd3eb1b0_1
jedi                      0.18.1           py37haa95532_1
jinja2                    3.1.2            py37haa95532_0
joblib                    1.1.0              pyhd3eb1b0_0
jpeg                      9e                   h2bbff1b_0
jsonschema                4.17.3           py37haa95532_0
jupyter_client            7.4.9            py37haa95532_0
jupyter_core              4.11.2           py37haa95532_0
jupyterlab_pygments       0.1.2                      py_0
keras                     2.3.1                         0
keras-applications        1.0.8                      py_1
keras-base                2.3.1                    py37_0
keras-preprocessing       1.1.2              pyhd3eb1b0_0
kiwisolver                1.4.2            py37hd77b12b_0
lerc                      3.0                  hd77b12b_0
libbrotlicommon           1.0.9                h2bbff1b_7
libbrotlidec              1.0.9                h2bbff1b_7
libbrotlienc              1.0.9                h2bbff1b_7
libdeflate                1.8                  h2bbff1b_5
libpng                    1.6.37               h2a8f88b_0
libprotobuf               3.20.1               h23ce68f_0
libsodium                 1.0.18               h62dcd97_0
libtiff                   4.4.0                h8a3f274_0
libuv                     1.40.0               he774522_0
libwebp                   1.2.2                h2bbff1b_0
llvmlite                  0.38.0           py37h23ce68f_0
locket                    1.0.0            py37haa95532_0
lz4-c                     1.9.3                h2bbff1b_1
markdown                  3.3.4            py37haa95532_0
markupsafe                2.1.1            py37h2bbff1b_0
matplotlib                3.5.2            py37haa95532_0
matplotlib-base           3.5.2            py37hd77b12b_0
matplotlib-inline         0.1.6            py37haa95532_0
mistune                   0.8.4           py37hfa6e2cd_1001
mkl                       2021.4.0           haa95532_640
mkl-service               2.4.0            py37h2bbff1b_0
mkl_fft                   1.3.1            py37h277e83a_0
mkl_random                1.2.2            py37hf11a4ad_0
multidict                 5.2.0            py37h2bbff1b_3
multipledispatch          0.6.0                    py37_0
munkres                   1.1.4                      py_0
nbclient                  0.5.13           py37haa95532_0
nbconvert                 6.4.1            py37haa95532_0
nbformat                  5.7.0            py37haa95532_0
nest-asyncio              1.5.6            py37haa95532_0
networkx                  2.6.3              pyhd3eb1b0_0
ninja                     1.10.2               haa95532_5
ninja-base                1.10.2               h6d14046_5
notebook                  6.4.12           py37haa95532_0
numba                     0.55.1           py37hc29f945_1    conda-forge
numexpr                   2.8.3            py37hb80d3ca_0
numpy                     1.21.5           py37h7a0a035_3
numpy-base                1.21.5           py37hca35cd5_3
oauthlib                  3.2.0              pyhd3eb1b0_1
olefile                   0.46                     py37_0
openpyxl                  3.0.10           py37h2bbff1b_0
openssl                   1.1.1w               h2bbff1b_0
opt_einsum                3.3.0              pyhd3eb1b0_1
packaging                 21.3               pyhd3eb1b0_0
pandas                    1.3.5            py37h6214cd6_0
pandocfilters             1.5.0              pyhd3eb1b0_0
panel                     0.14.3           py37haa95532_0
param                     1.12.3           py37haa95532_0
parso                     0.8.3              pyhd3eb1b0_0
partd                     1.2.0              pyhd3eb1b0_1
patsy                     0.5.2            py37haa95532_1
pickleshare               0.7.5           pyhd3eb1b0_1003
pillow                    9.2.0            py37hdc2b20a_1
pip                       22.1.2           py37haa95532_0
pkgutil-resolve-name      1.3.10           py37haa95532_0
prometheus_client         0.14.1           py37haa95532_0
prompt-toolkit            3.0.20             pyhd3eb1b0_0
prompt_toolkit            3.0.20               hd3eb1b0_0
protobuf                  3.20.1           py37hd77b12b_0
psutil                    5.9.0            py37h2bbff1b_0
pyasn1                    0.4.8              pyhd3eb1b0_0
pyasn1-modules            0.2.8                      py_0
pycparser                 2.21               pyhd3eb1b0_0
pyct                      0.5.0            py37haa95532_0
pygments                  2.11.2             pyhd3eb1b0_0
pyjwt                     2.4.0            py37haa95532_0
pynndescent               0.5.10             pyh1a96a4e_0    conda-forge
pyopenssl                 22.0.0             pyhd3eb1b0_0
pyparsing                 3.0.9            py37haa95532_0
pyqt                      5.9.2            py37hd77b12b_6
pyreadline                2.1                      py37_1
pyrsistent                0.18.0           py37h196d8e1_0
pysocks                   1.7.1                    py37_1
python                    3.7.13               h6244533_0
python-dateutil           2.8.2              pyhd3eb1b0_0
python-fastjsonschema     2.16.2           py37haa95532_0
python_abi                3.7                     2_cp37m    conda-forge
pytorch                   1.12.1          py3.7_cuda11.3_cudnn8_0    pytorch
pytorch-metric-learning   1.6.2              pyh39e3cac_0    metric-learning
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.1           py37haa95532_0
pyviz_comms               2.0.2              pyhd3eb1b0_0
pywavelets                1.3.0            py37h2bbff1b_0
pywin32                   305              py37h2bbff1b_0
pywinpty                  2.0.10           py37h5da7b33_0
pyyaml                    6.0              py37h2bbff1b_1
pyzmq                     23.2.0           py37hd77b12b_0
qt                        5.9.7            vc14h73c81de_0
requests                  2.28.1           py37haa95532_0
requests-oauthlib         1.3.0                      py_0
rsa                       4.7.2              pyhd3eb1b0_1
scikit-image              0.19.2           py37hf11a4ad_0
scikit-learn              1.0.2            py37hf11a4ad_1
scipy                     1.7.3            py37h7a0a035_2
seaborn                   0.11.2             pyhd3eb1b0_0
send2trash                1.8.0              pyhd3eb1b0_1
sentry-sdk                1.36.0                   pypi_0    pypi
setproctitle              1.3.3                    pypi_0    pypi
setuptools                63.4.1           py37haa95532_0
sip                       4.19.13          py37hd77b12b_0
six                       1.16.0             pyhd3eb1b0_1
smmap                     5.0.1                    pypi_0    pypi
sqlite                    3.39.2               h2bbff1b_0
statsmodels               0.13.2           py37h2bbff1b_0
tbb                       2021.5.0             h2d74725_1    conda-forge
tensorboard               2.6.0                      py_1
tensorboard-data-server   0.6.0            py37haa95532_0
tensorboard-plugin-wit    1.8.1            py37haa95532_0
tensorflow                2.1.0           eigen_py37hd727fc0_0
tensorflow-base           2.1.0           eigen_py37h49b2757_0
tensorflow-estimator      2.6.0              pyh7b7c402_0
termcolor                 1.1.0            py37haa95532_1
terminado                 0.17.1           py37haa95532_0
testpath                  0.6.0            py37haa95532_0
threadpoolctl             2.2.0              pyh0d69192_0
tifffile                  2020.10.1        py37h8c2d366_2
tk                        8.6.12               h2bbff1b_0
toolz                     0.11.2             pyhd3eb1b0_0
torchaudio                0.12.1               py37_cu113    pytorch
torchvision               0.13.1               py37_cu113    pytorch
tornado                   6.2              py37h2bbff1b_0
tqdm                      4.64.0           py37haa95532_0
traitlets                 5.1.1              pyhd3eb1b0_0
typing-extensions         4.3.0            py37haa95532_0
typing_extensions         4.3.0            py37haa95532_0
umap-learn                0.5.3            py37h03978a9_0    conda-forge
urllib3                   1.26.11          py37haa95532_0
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wandb                     0.16.0                   pypi_0    pypi
wcwidth                   0.2.5              pyhd3eb1b0_0
webencodings              0.5.1                    py37_1
werkzeug                  0.16.1                     py_0
wheel                     0.37.1             pyhd3eb1b0_0
win_inet_pton             1.1.0            py37haa95532_0
wincertstore              0.2              py37haa95532_2
winpty                    0.4.3                         4
wrapt                     1.14.1           py37h2bbff1b_0
xarray                    0.20.1             pyhd3eb1b0_1
xlrd                      2.0.1              pyhd3eb1b0_0
xlsxwriter                3.0.3              pyhd3eb1b0_0
xz                        5.2.6                h8cc25b3_0
yaml                      0.2.5                he774522_0
yarl                      1.8.1            py37h2bbff1b_0
zeromq                    4.3.4                hd77b12b_0
zipp                      3.8.0            py37haa95532_0
zlib                      1.2.12               h8cc25b3_3
zstd                      1.5.2                h19a0ad4_0