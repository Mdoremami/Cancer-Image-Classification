import medmnist
from medmnist.info import INFO, DEFAULT_ROOT
import os


def available():
    '''List all available datasets.'''
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    print("All available datasets:")
    for key in INFO.keys():
        print("\t"+key)


def downloadall(root=DEFAULT_ROOT):
    '''Download all available datasets.'''
    for key in INFO.keys():
        print(f"Downloading {key}...")
        _ = getattr(medmnist, INFO[key]['python_class'])(
            split="train", root=root, download=True)


def downloadbreast(root=DEFAULT_ROOT):
    '''Download breast dataset.'''
    for key in INFO.keys():
        if key == 'breastmnist' :
            print(f"Downloading {key}...")
            _ = getattr(medmnist, INFO[key]['python_class'])(
                split="train", root=root, download=True)
            
def downloadblood(root=DEFAULT_ROOT):
    '''Download blood dataset.'''
    for key in INFO.keys():
        if key == 'bloodmnist' :
            print(f"Downloading {key}...")
            _ = getattr(medmnist, INFO[key]['python_class'])(
                split="train", root=root, download=True)

def clean(root=DEFAULT_ROOT):
    '''Delete all downloaded npz from root.'''
    import os
    from glob import glob

    for path in glob(os.path.join(root, "*mnist*.npz")):
        os.remove(path)


def info(flag):
    '''Print the dataset details given a subset flag.'''
    from pprint import pprint

    pprint(INFO[flag])


def save(flag, folder, postfix="png", root=DEFAULT_ROOT):
    '''Save the dataset as standard figures, which could be used for AutoML tools, e.g., Google AutoML Vision.'''

    for split in ["train", "val", "test"]:
        print(f"Saving {flag} {split}...")
        dataset = getattr(medmnist, INFO[flag]['python_class'])(
            split=split, root=root)
        dataset.save(folder, postfix)


def evaluate(path):
    '''Parse and evaluate a standard result file.

    A standard result file is named as:
        {flag}_{split}|*|@{run}.csv (|*| means anything)

    A standard evaluation file is named as:
        {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv

    In result/evaluation file, each line is (dataset index,float prediction).

    For instance,
    octmnist_test_[AUC]0.672_[ACC]0.892@3.csv
        0,0.125,0.275,0.5,0.2
        1,0.5,0.125,0.275,0.2
    '''
    #medmnist.Evaluator.parse_and_evaluate(path)
    medmnist.Evaluator(path)


def test(save_folder="tmp/", root=DEFAULT_ROOT):
    '''For developmemnt only.'''

    import os
    from glob import glob

    available()

    downloadall(root)

    for key in INFO.keys():
        if key.endswith("mnist"):
            postfix = "jpg"
            # continue
        else:
            postfix = "gif"
            # continue

        print(f"Verifying {key}....")

        info(key)

        save(key, save_folder, postfix=postfix, root=root)

        for split in ["train", "val", "test"]:

            dataset = getattr(medmnist, INFO[key]['python_class'])(
                split=split, root=root)
            assert len(dataset) == INFO[key]["n_samples"][split]

            evaluator = medmnist.Evaluator(key, split)
            dummy = evaluator.get_dummy_prediction()
            evaluator.evaluate(dummy, save_folder)

            dummy_evaluation_file = glob(os.path.join(
                save_folder, f"{key}_{split}*.csv"))[0]

            medmnist.Evaluator.parse_and_evaluate(
                dummy_evaluation_file, run="dummy")

        n_channels = INFO[key]["n_channels"]

        _, *shape = dataset.imgs.shape
        if n_channels == 3:
            assert shape == [28, 28, 3]
        else:
            assert n_channels == 1
            assert shape == [28, 28] or shape == [28, 28, 28]

        dataset.montage(save_folder=save_folder, replace=True)

    # clean(root)


def remove_subfolders():
    current_dir = os.getcwd()  # Get the current directory
    
    # Iterate over the items in the current directory
    for item in os.listdir(current_dir):
        item_path = os.path.join(current_dir, item)  # Get the full path of the item
        
        # Check if the item is a directory
        if os.path.isdir(item_path):
            print(f"Removing subfolder: {item}")
            os.rmdir(item_path)  # Remove the subfolder


#%% Get Breast Data

# Set data path 
myaddress = 'E:\\University (EMIMEO)\\Semester 3, Italy\\Image Data Analysis 2022-23\\PROJECT\\Breast'
os.chdir(myaddress)

# Clean directory of any datasets and folders
clean(myaddress)
remove_subfolders()

# Download the breast dataset
downloadbreast(myaddress)

# Get some info about data
flagname = 'breastmnist'
info(flagname)

# Save dataset in png format from numpy arrays
os.mkdir('Images')
save(flagname,'Images',postfix='png',root=myaddress)
