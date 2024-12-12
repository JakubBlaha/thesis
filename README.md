# Running the pipeline

1. Make sure to install required packages by running `pip install -r requirements.txt`
2. Copy dataset files:
   
   The DASPS dataset individual preprocessed `.mat` files need to be placed in `data/datasets/DASPS`. The files can be found in the following directory after downloading and extracting the DASPS archive from https://ieee-dataport.org/open-access/dasps-database:

    - `DASPS_Database/Preprocessed data.mat`

   The SAD dataset individual files are intended to be placed in `data/datasets/SAD/preprocessed/control` and `data/datasets/SAD/preprocessed/severe` directories. Only these severities are used currently. The files can be found in the following directories after downloading and extracting the provided archive (https://vutbr.sharepoint.com/:f:/s/DataSets/EncrNE2S6etLhttYQEIevyoBipy3Thy_s_ZA2BwfujHmsA?e=M3loxP):

    - `Subjects_Hakim_SAD/Control/Eyes close/preprocessed`
    - `Subjects_Hakim_SAD/Severe/Eyes close/preprocessed`

    The folder data structures should look like the following:

    ```
    data
    ├── DASPS.csv
    └── datasets
        ├── DASPS
        │   ├── S01preprocessed.mat
        │   ├── ...
        │   └── S23preprocessed.mat
        └── SAD
            └── preprocessed
                ├── control
                │   ├── C1.edf
                │   ├── ...
                │   └── C9.edf
                └── severe
                    ├── C1.edf
                    ├── ...
                    └── C9.edf
    ```

3. Run the pipeline:
   ```
   python -m scripts all --seglen <SEGMENT LENGTH IN SECONDS>
   ```
   After the pipeline is finished, extracted features along with labeling are saved in `data/features`. DASPS segments are labeled according to SAM or HAM
   labeling scheme. For details on how the labeling is done, see [utils.py](./scripts/utils.py).