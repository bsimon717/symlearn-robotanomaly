# symlearn-robotanomaly

This is a project meant to prototype the *symbiotic learning* paradigm using a robotic arm dataset.

All details regarding the dataset can be found [here](https://www.kaggle.com/datasets/hkayan/industrial-robotic-arm-imu-data-casper-1-and-2).

This project has been formatted to be a multi-label classification task, where the network is meant to identify whether a given stack of frames is anomalous or not.

---

# Instructions
## Install Package and Process Dataset
First, run the following to retrieve the symbiotic-learning package and create the logs directory:

`pip install symbiotic-learning`

`mkdir sym_logs`

Then run the following to download the robotic arm dataset and process it into numpy arrays:

```
python process_data.py --seq_len=SEQ_LEN --max_samples=MAX_SAMPLES --seed=SEED
```

where `SEQ_LEN` is the number of frames to stack as one sequence, `MAX_SAMPLES` is the size of the slice taken from the non-anomalous dataframe, and `SEED` sets the random seed.

## Run Training

To perform a training, run the following:
```
python main.py --seq_len=SEQ_LEN
```

where `SEQ_LEN` is the same number as was specified when processing the datasets.

Hyperparameters can be specified using command-line arguments (run `python main.py --help` to see the full list).

