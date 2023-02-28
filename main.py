
import argparse
from orientation_tracking import run_orientation_tracking
from test_tracking import test_orientation_tracking

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="data/trainset", help="dataset path")
parser.add_argument("--set_name", type=str, nargs='+', default=["imuRaw1.p"], help="set name")

args = parser.parse_args()

if "train" in args.dataset_path:
    for dataset in args.set_name:
        run_orientation_tracking(args.dataset_path, dataset)
elif "test" in args.dataset_path:
    for dataset in args.set_name:
        test_orientation_tracking(args.dataset_path, dataset)
else:
    print("Invalid dataset path")
    

