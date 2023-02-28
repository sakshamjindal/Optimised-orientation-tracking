## Optimised Orientation Tracking using Riemann Stochastic Gradient Descent (RSGD)

[(Report)](https://drive.google.com/file/d/1TwlCjqZtMoP40xfIxifT1dWg7Wo3BhL3/view?usp=sharing)

To run the orientation tracking on the trainset

```
python main.py --dataset_path data/trainset --set_name imuRaw1.p
```

where 
- `dataset_path` is path where training set is located (str)
- `set_name` is the dataset to run the tracking (list of str) eg. "imuRaw1.p imuRaw2.p"


To run the orientation tracking on the testset

```
python main.py --dataset_path data/testset --set_name imuRaw10.p
```

where 
- `dataset_path` is path where test set is located (str)
- `set_name` is the dataset on which you are running (str) eg. "imuRaw10.p" or "imuRaw11.p"