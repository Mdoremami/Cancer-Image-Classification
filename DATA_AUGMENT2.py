#%% INFO

'''

In this code, we will only augment data to the labels with fewer data 
for example, in Training data, label 0 data is fewer than label 1, we will
add more data to label 0 to be equal to label 1

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


# Group the data by data type and label
grouped = df.groupby(['type', 'values'])

# Count the occurrences of each label for each data type
counts = grouped.size().unstack(fill_value=0)

# Augment the data to equalize the counts for each data type
augmented_data = []
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1.0 / 255
)

#%% Augment DATA

for data_type in counts.index:
    count_0 = counts.loc[data_type, 0]
    count_1 = counts.loc[data_type, 1]
    diff = count_1 - count_0
    
    if diff > 0:
        target_count = count_1
        
        # Augment label 0 to match the count of label 1
        label_0_data = df[(df['type'] == data_type) & (df['values'] == 0)].sample(n=diff, replace=True)
        for _, row in label_0_data.iterrows():
            if augmented_indices[data_type]-count_1 >= target_count:
                break
            image_path = os.path.join(data_dir, row['filename'])
            image = Image.open(image_path)
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=-1)
            
            augmented_images = datagen.flow(image_array, batch_size=1)
            augmented_index = augmented_indices[data_type]
            
            for i, augmented_image in enumerate(augmented_images):
                if i >= 1:  # Specify the desired number of augmented images to generate (e.g., 5 in this case)
                    break
                augmented_image = augmented_image.squeeze()
                augmented_filename = f"{data_type}{augmented_index + i}_{0}.png"
                augmented_image_path = os.path.join(output_dir, augmented_filename)
                augmented_image = Image.fromarray((augmented_image * 255).astype(np.uint8))
                augmented_image.save(augmented_image_path)
                
                augmented_data.append({
                    'type': data_type,
                    'filename': augmented_filename,
                    'values': 0
                })
                
            augmented_indices[data_type] += 1


    elif diff < 0:
        target_count = count_0
        
        # Augment label 1 to match the count of label 0
        label_1_data = df[(df['type'] == data_type) & (df['values'] == 1)].sample(n=abs(diff), replace=True)
        for _, row in label_1_data.iterrows():
            if augmented_indices[data_type]-count_0 >= target_count:
                break
            image_path = os.path.join(data_dir, row['filename'])
            image = Image.open(image_path)
            image_array = np.array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = np.expand_dims(image_array, axis=-1)
            
            augmented_images = datagen.flow(image_array, batch_size=1)
            augmented_index = augmented_indices[data_type]
            
            for i, augmented_image in enumerate(augmented_images):
                if i >= 1:  # Specify the desired number of augmented images to generate (e.g., 5 in this case)
                    break
                augmented_image = augmented_image.squeeze()
                augmented_filename = f"{data_type}{augmented_index + i}_{1}.png"
                augmented_image_path = os.path.join(output_dir, augmented_filename)
                augmented_image = Image.fromarray((augmented_image * 255).astype(np.uint8))
                augmented_image.save(augmented_image_path)
                
                augmented_data.append({
                    'type': data_type,
                    'filename': augmented_filename,
                    'values': 1
                })
                
            augmented_indices[data_type] += 1


# Concatenate the augmented data
if augmented_data:
    augmented_df = pd.DataFrame(augmented_data)
    df = pd.concat([df, augmented_df], ignore_index=True)

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


#%% TEST IF OK

os.chdir(datapth)

# Re-Open new file again
df = pd.read_csv('breast_final.csv',header=None)
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
