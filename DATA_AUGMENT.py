#%% INFO

'''
This file augments 3 more data to all data avilable in the dataset.

'''

#%% Libraries

import os
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#%% Getting Dataset
myaddress = 'E:\\University (EMIMEO)\\Semester 3, Italy\\Image Data Analysis 2022-23\\PROJECT\\Breast'
datapth = os.path.join(myaddress,'Images')
os.chdir(datapth)
try:
    os.mkdir('Augmented')
except FileExistsError:
    pass


# Setting a function to read images
Imgpath = os.path.join(datapth,'breastmnist\\')

data_dir = Imgpath
csv_file = datapth
output_dir = os.path.join(datapth,'Augmented')

# read CSV file
dfedit = pd.read_csv('breastmnist.csv',header=None)
Headers = ['type','filename','values']
dfedit.columns = Headers

# Modify the "VALIDATION" label to "VAL"
dfedit['type'] = dfedit['type'].replace('VALIDATION', 'VAL')
new_csv_file = 'breastmnist_new1.csv'
dfedit.to_csv(new_csv_file, index=False, header=False)

# Re-Open new file again
df = pd.read_csv('breastmnist_new1.csv',header=None)
Headers = ['type','filename','values']
df.columns = Headers

# Group the data by data type and label
grouped = df.groupby(['type', 'values'])

# Count the occurrences of each label for each data type
counts = grouped.size().unstack(fill_value=0)

# Compare the counts of label 0 and label 1 for each data type
for data_type in counts.index:
    count_0 = counts.loc[data_type, 0]
    count_1 = counts.loc[data_type, 1]
    
    if count_0 < count_1:
        print(f"Label 0 is less than Label 1 for {data_type}")
        print(f"Number of Label 0: {count_0}")
        print(f"Number of Label 1: {count_1}")
        print(f"Difference: {count_1 - count_0}\n")
    elif count_0 > count_1:
        print(f"Label 1 is less than Label 0 for {data_type}")
        print(f"Number of Label 0: {count_0}")
        print(f"Number of Label 1: {count_1}")
        print(f"Difference: {count_0 - count_1}\n")
    else:
        print(f"Label 0 and Label 1 are equal for {data_type}")
        print(f"Number of Label 0: {count_0}")
        print(f"Number of Label 1: {count_1}\n")


# Create Image Generator Function to augment data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1.0 / 255
)



# Find the last index for each data type (TRAIN, TEST, VAL)
last_indices = {
    'TRAIN': df[df['type'] == 'TRAIN']['filename'].apply(lambda x: int(x.split('_')[0][5:])).max(),
    'TEST': df[df['type'] == 'TEST']['filename'].apply(lambda x: int(x.split('_')[0][4:])).max(),
    'VAL': df[df['type'] == 'VAL']['filename'].apply(lambda x: int(x.split('_')[0][3:])).max()
}


# Set the initial index for augmented data
augmented_indices = {
    'TRAIN': last_indices['TRAIN'] + 1,
    'TEST': last_indices['TEST'] + 1,
    'VAL': last_indices['VAL'] + 1
}

#%% Start the process

augmented_data = []  # Store augmented data for updating the CSV file

for index, row in df.iterrows():
    data_type = row['type']
    filename = row['filename']
    label = row['values']

    # Construct the image path
    image_path = os.path.join(data_dir, filename)

    # Load the image using PIL
    image = Image.open(image_path)

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Expand the dimensions of the image array to match the input shape expected by the ImageDataGenerator
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)  # Add an extra dimension for the channel

    # Generate augmented images
    augmented_images = datagen.flow(image_array, batch_size=1)

    # Get the initial index for augmented data
    augmented_index = augmented_indices[data_type]
    
    # Save augmented images and update the CSV file
    for i, augmented_image in enumerate(augmented_images):
        if i >= 3:  # Specify the desired number of augmented images to generate (e.g., 5 in this case)
            break

        augmented_image = augmented_image.squeeze()

        augmented_filename = f"{data_type}{augmented_index + i}_{label}.png"
        augmented_image_path = os.path.join(output_dir, augmented_filename)

        # Save augmented image
        # augmented_image = augmented_image.squeeze()
        augmented_image = Image.fromarray((augmented_image * 255).astype(np.uint8))
        augmented_image.save(augmented_image_path)

        # To Update the CSV file with augmented data
        augmented_data.append({
            'type': data_type,
            'filename': augmented_filename,
            'values': label
        })
    # Increment the augmented index for the current data type
    augmented_indices[data_type] += 3  # Increment by the desired number of augmented images


# Append augmented data to the original dataframe
df = df.append(augmented_data, ignore_index=True)

# Save the updated CSV file
df.to_csv('breast_final.csv', index=False, header=False)


#%% Move 
# move all original files from main source to augmented ones to have all in 
# one place

import os
import shutil

source_dirs = [
    r'E:\University (EMIMEO)\Semester 3, Italy\Image Data Analysis 2022-23\PROJECT\Breast\Images\breastmnist',
    r'E:\University (EMIMEO)\Semester 3, Italy\Image Data Analysis 2022-23\PROJECT\Breast\Images\Augmented'
]

destination_dir = r'E:\University (EMIMEO)\Semester 3, Italy\Image Data Analysis 2022-23\PROJECT\Breast\Images\final_data'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Move files from the source directories to the destination directory
for source_dir in source_dirs:
    file_list = os.listdir(source_dir)
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copy(source_path, destination_path)
