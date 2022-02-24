# LIGN167_Final_Project_WI22
LIGN167 Final Project
Topic: Hate Speech Classification
Team members: Allen Cheung, Beibei Du, Kaiyuan Wang

## Repo Structure (After running ```prepare_datasets.sh```)

- get_data.sh
- datasets
    - [parler_pretrain.ndjson](./datasets/parler_pretrain.ndjson) (**Parler** dataset for pre-training)
    - [gab.csv](./datasets/gab.csv) (**Gab** dataset for fine-tune training and testing)
    - [twitter.csv](./datasets/twitter.csv) (**Twitter** dataset for fine-tune training and testing)
    - [Reddit.csv](./datasets/reddit.csv) (**Reddit** dataset for fine-tune training and testing)


## Use GPU Cluster (UCSD DataHub)
- Step1: Go to [DataHub Website](https://datahub.ucsd.edu/hub/spawn)
- Step2: Select the env with GPU ( Python 3, nbgrader (1 GPU, 8 CPU, 16G RAM) )
- Step3: Click "Launch Environment"
- Step4: On the top right corner, click "New". In drop-down menu, click "Terminal"
  - This should spawn a bash terminal
- Step5: Clone project repo in the new terminal, by running: 

  ```git clone https://github.com/kylewang1999/LIGN167_hate_speech_classification.git```

  **NOTE**: GitHub now requires personal access token for ssh authentication. See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for how to create the token.
- Step6: (Only after the first clone) Download dataset by running: 
  
  ```cd ~/LIGN167_hate_speech_classification/ && bash get_data.sh```

- Step7: Switich to the desired branch by running: 
  
  ```git checkout <branch_name>```

## Datasets [(Kaggle Link)](https://www.kaggle.com/kylewang1999/hate-speech-datasets) & References

1. [Hate Speach from **Parler**](https://zenodo.org/record/4442460#.YhcimC-B0iw) Aliapoulios et. al.
    ```
    @dataset{max_aliapoulios_2021_4442460,
        author       = {Max Aliapoulios and
                        Emmi Bevensee and
                        Jeremy Blackburn and
                        Barry Bradlyn and
                        Emiliano De Cristofaro and
                        Gianluca Stringhini and
                        Savvas Zannettou},
        title        = {{A Large Open Dataset from the Parler Social 
                        Network}},
        month        = jan,
        year         = 2021,
        publisher    = {Zenodo},
        version      = 1,
        doi          = {10.5281/zenodo.4442460},
        url          = {https://doi.org/10.5281/zenodo.4442460}
    }
    ```

2. [Hate Speech from **Twitter**](https://github.com/t-davidson/hate-speech-and-offensive-language), Davidson et. al.
    ```
    @inproceedings{hateoffensive,
        title = {Automated Hate Speech Detection and the Problem of Offensive Language},
        author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
        booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
        series = {ICWSM '17},
        year = {2017},
        location = {Montreal, Canada},
        pages = {512-515}
    }
    ```

3. [Hate Speech from **Gab** and **Reddit**](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech), Qian et. al.
    ```
    @misc{qian2019benchmark,
        title={A Benchmark Dataset for Learning to Intervene in Online Hate Speech}, 
        author={Jing Qian and Anna Bethke and Yinyin Liu and Elizabeth Belding and William Yang Wang},
        year={2019},
        eprint={1909.04251},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
    ```