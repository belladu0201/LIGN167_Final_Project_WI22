# LIGN167_Final_Project_WI22
LIGN167 Final Project
Topic: Hate Speech Classification
Team members: Allen Cheung, Beibei Du, Kaiyuan Wang

## Jump Start 
- Fine-tune on twitter datset for 10 epochs (approx. 1 min per epoch)

  ```python main.py --dataset twitter --save --epoch 10```

- Force a git pull on datahub (WARNING: All changes on datahub will be overwritten):
  ```
  git fetch --all
  git branch backup-master
  git reset --hard origin/main
  ```

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

  ```https://github.com/belladu0201/LIGN167_Final_Project_WI22.git```

  **NOTE**: GitHub now requires personal access token for ssh authentication. See [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) for how to create the token.
- Step6: (Only after the first clone) Download dataset by running: 
  
  ```cd ~/LIGN167_Final_Project_WI22/ && bash get_data.sh```

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

2. [Hate Speech from **Twitter**](https://github.com/t-davidson/hate-speech-and-offensive-language), Davidson et. al. The data are stored as a CSV and as a pickled pandas dataframe (Python 2.7). Each data file contains 5 columns:
    - `count` = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

    - `hate_speech` = number of CF users who judged the tweet to be hate speech.

    - `offensive_language` = number of CF users who judged the tweet to be offensive.

    - `neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.

    - **`class`** = class label for majority of CF users.
      0 - hate speech
      1 - offensive  language
      2 - neither
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
      ```
      datasets/twitter.csv Dataset Length: [24783]
      ---- Number of [neutral] tweets: 4163 (16.8%)
      0     [rt, as, a, woman, you, shouldn't, complain, a...
      40    [momma, said, no, pussi, cat, insid, my, doghous]
      63    [-simplyaddictedtoguy, woof, woof, hot, scalli...
      66                         [woof, woof, and, hot, sole]
      67    [lemmi, eat, a, oreo, &amp;, do, these, dish, ...
      Name: text, dtype: object
      ---- Number of [offensive] tweets: 19190 (77.4%)
      1    [rt, boy, dat, cold, tyga, dwn, bad, for, cuff...
      2    [rt, dawg, rt, you, ever, fuck, a, bitch, and,...
      3                     [rt, she, look, like, a, tranni]
      4    [rt, the, shit, you, hear, about, me, might, b...
      5    [the, shit, just, blow, me, claim, you, so, fa...
      Name: text, dtype: object
      ---- Number of [HATEFUL] tweets: 1430 (5.800000000000001%)
      85                                       [queer, gaywad]
      89     [alsarabsss, he, a, beaner, smh, you, can, tel...
      110    [you'r, fuck, gay, blacklist, hoe, hold, out, ...
      184    [lmfaoooo, i, hate, black, peopl, thi, is, whi...
      202              [at, least, i'm, not, a, nigger, lmfao]
      ```

3. [Hate Speech from **Gab** and **Reddit**](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech), Qian et. al.
    - The `'reponse'` column contains human annotation for hateful speech
      - For non-hatful speech, `respons` is `NaN`
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
    ```
    datasets/gab.csv Dataset Length: [45601]
    ---- Number of [neutral] tweets: 1723 (5.220000000000001%)
    42                           [is, thi, peak, redneck, 🤔]
    50     [so, i, get, dox, and, flame, and, i'm, just, ...
    51     [appar, some, peopl, fear, that, you, will, co...
    52     [my, journal, will, probabl, devolv, into, eve...
    242    [if, “men, have, a, hard, time, right, now”, t...
    ---- Number of [HATEFUL] tweets: 31259 (94.78%)
    0    [i, join, gab, to, remind, myself, how, retard...
    1    [thi, is, what, the, left, is, realli, scare, of]
    2    [that, liter, look, like, a, monkey, whi, are,...
    3                                         [dumb, cunt]
    4                          [it, make, you, an, asshol]
    5    [give, it, to, a, soldier, who, ha, defend, it...
    6    [so, they, manag, to, provid, a, whole, lot, o...
    ```
    ```
    datasets/reddit.csv Dataset Length: [22324]
    ---- Number of [neutral] tweets: 5335 (23.9%)
    7     [wouldn't, the, defend, or, whatev, they, are,...
    8                          ['inclusive', =, not, white]
    9     [“harvard, is, work, to, be, more, inclusive”,...
    10    [oh, yeah, and, that, lawsuit, go, to, benefit...
    11    [-, ***a***nti-***c***aucasian, ***l***itig, *...
    Name: text, dtype: object
    ---- Number of [HATEFUL] tweets: 16989 (76.1%)
    0    [a, subsect, of, retard, hungarian, ohh, boy, ...
    1    [hiii, just, got, off, work, is, mainli, the, ...
    2    [wow, i, guess, soyboy, are, the, same, in, ev...
    3    [owen, benjamin', soyboy, song, goe, for, ever...
    4    [y'all, hear, sumn, by, all, mean, i, live, in...
    5                                          [[removed]]
    6    [ah, a, liber, ha, slip, in, you, can, tell, b...
    ```

## Notes
1. [PreTraining BERT From Scratch](https://discuss.huggingface.co/t/tips-for-pretraining-bert-from-scratch/1175/10)