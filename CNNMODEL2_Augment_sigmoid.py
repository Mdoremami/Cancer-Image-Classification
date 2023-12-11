#%% Importing Libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout

#%% Getting Info about codes
tf_version = tf.__version__
print("TensorFlow version:", tf_version)

#%% Getting Dataset
myaddress = 'E:\\University (EMIMEO)\\Semester 3, Italy\\Image Data Analysis 2022-23\\PROJECT\\Breast'
datapth = os.path.join(myaddress,'Images')
os.chdir(datapth)

# make Dataframe for CSV file
df = pd.read_csv(datapth+"\\"+"breast_final.csv",header=None)
Headers = ['type','filename','values']
df.columns = Headers

# get data filepaths
train_file_paths = df[df['type'] == 'TRAIN']['filename'].tolist()
train_labels = df[df['type'] == 'TRAIN']['values'].tolist()
ds_train_tmp = tf.data.Dataset.from_tensor_slices((train_file_paths,train_labels))


val_file_paths = df[df['type'] == 'VAL']['filename'].tolist()
val_labels = df[df['type'] == 'VAL']['values'].tolist()
ds_val_tmp = tf.data.Dataset.from_tensor_slices((val_file_paths,val_labels))


test_file_paths = df[df['type'] == 'TEST']['filename'].tolist()
test_labels = df[df['type'] == 'TEST']['values'].tolist()
ds_test_tmp = tf.data.Dataset.from_tensor_slices((test_file_paths,test_labels))

# Setting a function to read images
Imgpath = os.path.join(datapth,'final_data\\')
os.chdir(Imgpath)

def read_image(image_file,label):
    image = tf.io.read_file(Imgpath + image_file)
    image = tf.image.decode_png(image,channels=1,dtype=tf.uint8)
    image = tf.cast(image, tf.float32)
    image = image / 255.0       # Normalize
    return image, label

ds_train = ds_train_tmp.map(read_image)
ds_val = ds_val_tmp.map(read_image)
ds_test = ds_test_tmp.map(read_image)

# Determine data size
train_data_size = len(list(ds_train.as_numpy_iterator()))
val_data_size = len(list(ds_val.as_numpy_iterator()))
test_data_size = len(list(ds_test.as_numpy_iterator()))

# Print data sizes
print("Train data size:", train_data_size)
print("Validation data size:", val_data_size)
print("Test data size:", test_data_size)

# Set shuffle buffer size to data size for perfect shuffling
ds_train = ds_train.shuffle(train_data_size)
ds_val = ds_val.shuffle(val_data_size)
ds_test = ds_test.shuffle(test_data_size)


# Batching
batch_size = 5
ds_train = ds_train.batch(batch_size)
ds_val = ds_val.batch(batch_size)
ds_test = ds_test.batch(batch_size)

#### JUST TO TEST ####
train_data_size = len(list(ds_train.as_numpy_iterator())) * batch_size
val_data_size = len(list(ds_val.as_numpy_iterator())) * batch_size
test_data_size = len(list(ds_test.as_numpy_iterator())) * batch_size

# Print data sizes
print("Train data size:", train_data_size)
print("Validation data size:", val_data_size)
print("Test data size:", test_data_size)


#%% Visualize Data

# Take a few samples from the dataset
batch_number = 0  # for example, nth batch; n=0 --> train images 0-4, n=1 --> train images 5-9
sample_images, sample_labels = next(iter(ds_train.skip(batch_number).take(1)))  # Take nth batch

# Reshape the batch to individual samples
sample_images = tf.squeeze(sample_images)[:batch_size]
sample_labels = sample_labels[:batch_size]

# Visualize the sample images
fig, axs = plt.subplots(1, batch_size, figsize=(12, 4))

for i in range(batch_size):
    axs[i].imshow(sample_images[i].numpy().squeeze(), cmap='gray')
    axs[i].set_title(f"Label: {sample_labels[i].numpy()}")
    axs[i].axis('off')

plt.show()

# Print the numbers
for image, label in zip(sample_images, sample_labels):
    print(label.numpy())


#%% CNN MODEL Creating
try:
    del model
    del hist
except NameError:
    pass


# Creating Model 1
model = Sequential()

# Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())


# Flatten the feature maps
model.add(Flatten())


# Dense (fully connected) layers
model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.2)))
model.add(Dropout(0.5))


model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))

# model.add(Dense(8, activation='relu'))
# model.add(Dropout(0.05))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()


#%% First Network

num_epochs1 = 100

# Define the early stopping criteria...patience = 5 means if val_loss wasn't improving
# in 5 consecutive runs, it'll stop
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

hist = model.fit(ds_train , epochs = num_epochs1 , validation_data = ds_val , callbacks=[early_stopping])



#%% Evaluation
import sklearn.metrics as sklmetricz


# Storing Values for each epoch to be able to plot
precision_values = []
recall_values = []
# Values for showing the final metric score
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
y_true = []     # real labels
y_pred_prob = []     # predicted labels by model

for batch in ds_test.as_numpy_iterator():
    x,y = batch
    yhat = model.predict(x)
    y_true.extend(y)
    y_pred_prob.extend(yhat.flatten())
    
    pre.update_state(y,yhat)        # Get precision every round
    re.update_state(y,yhat)         # Get recall every round
    acc.update_state(y,yhat)        # Get Accuracy every round


    precision_values.append(pre.result().numpy())   # use these for plots
    recall_values.append(re.result().numpy())       # use these for plots


y_pred = np.array(y_pred_prob)  # Convert to NumPy array


# Calculate AUC
AUC1 = sklmetricz.roc_auc_score(y_true, y_pred)

# Calculate F1-Score
f1_score = sklmetricz.f1_score(y_true, y_pred.round())


print(f'AUC: {AUC1} , Accuracy:{acc.result().numpy()} , Precision:{pre.result().numpy()} , Recall:{re.result().numpy()} , F1-Score: {f1_score} ')

# Obtain the confusion matrix
confusion_matrix = sklmetricz.confusion_matrix(y_true, y_pred.round())

# Extract TP, TN, FP, FN from confusion matrix
tn, fp, fn, tp = confusion_matrix.ravel()

# Print TP, TN, FP, FN
print("True Positives (TP):", tp)
print("True Negatives (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)

# Specificity
specificity = tn / (tn + fp)
print("Specificity:", specificity)




#### SAVE METRICS ######


directory = datapth = os.path.join(myaddress,'Results_Model2_Aug_Sig')

try:
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully and data are saved!")
    else:
        print(f"Directory '{directory}' already exists!")
except OSError:
    # Do nothing (skip) in case of an exception
    pass

os.chdir(directory)

# Save values in a notepad
with open('metrics.txt', 'w') as f:
    f.write(f'AUC: {AUC1}\n')
    f.write(f'Accuracy: {acc.result().numpy()}\n')
    f.write(f'Precision: {pre.result().numpy()}\n')
    f.write(f'Recall: {re.result().numpy()}\n')
    f.write(f'F1-Score: {f1_score}\n')
    f.write(f'TP: {tp}\n')
    f.write(f'TN: {tn}\n')
    f.write(f'FP: {fp}\n')
    f.write(f'FN: {fn}\n')
    f.write(f'Specificity: {specificity}\n')


# Save true labels and predicted labels in a notepad

with open('labels.txt', 'w') as f:
    f.write('True Labels, Predicted Labels (Threshold=0.5), Predicted Labels (Raw)\n')
    for true_label, pred_prob in zip(y_true, y_pred_prob):
        pred_label_threshold = 1 if pred_prob >= 0.5 else 0
        f.write(f'{true_label}, {pred_label_threshold}, {pred_prob}\n')




#%% Plot Performance Model 1
import pydot
from tensorflow.keras.utils import plot_model


# Calculate FPR, TPR for plotting
fpr, tpr, _ = sklmetricz.roc_curve(y_true, y_pred_prob)

# Calculate F1-Scre for Plottin
precision_values = np.array(precision_values)
recall_values = np.array(recall_values)
f1_score_values = 2 * (precision_values * recall_values) / (precision_values + recall_values)

# Get the metric values from the history object
loss = hist.history['loss']
val_loss = hist.history['val_loss']
accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']

# Plotting the Loss
plt.figure(figsize=(4, 3))
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.savefig('Loss_plot.jpg', dpi=300)
plt.show()

# Plotting the Accuracy
plt.figure(figsize=(4, 3))
plt.plot(accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.title('Accuracy over Epochs')
plt.legend()
plt.savefig('Accuracy_plot.jpg', dpi=300)
plt.show()


# Plot Precision
plt.figure(figsize=(4, 3))
plt.plot(precision_values)
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim([0,1])
plt.title('Precision over Epochs')
plt.savefig('Precision_plot.jpg', dpi=300)
plt.show()


# Plot Recall
plt.figure(figsize=(4, 3))
plt.plot(recall_values)
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim([0,1])
plt.title('Recall over Epochs')
plt.savefig('Recall_plot.jpg', dpi=300)
plt.show()


# Plot F1-Score
plt.figure(figsize=(4, 3))
plt.plot(f1_score_values)
plt.xlabel('Epoch')
plt.ylabel('F1-Score')
plt.ylim([0, 1])
plt.title('F1-Score over Epochs')
plt.savefig('F1score_plot.jpg', dpi=300)
plt.show()




# Plot the ROC curve
plt.figure(figsize=(4, 3))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % AUC1)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # Diagonal line for random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('ROC_plot.jpg', dpi=300)
plt.show()




# Generate the model diagram
plot_model(model, to_file='model.png', show_shapes=True)
