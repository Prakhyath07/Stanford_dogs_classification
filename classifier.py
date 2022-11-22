#Import libraries
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime



#Create a function to do all the pre-processing in one go (Data Pipeline)
IMG_SIZE = 224
BATCH_SIZE = 32

unique_breeds = np.array(['Afghan_hound','African_hunting_dog','Airedale'
,'American_Staffordshire_terrier','Appenzeller','Australian_terrier'
,'Bedlington_terrier','Bernese_mountain_dog','Blenheim_spaniel'
,'Border_collie','Border_terrier','Boston_bull','Bouvier_des_Flandres'
,'Brabancon_griffon','Brittany_spaniel','Cardigan'
,'Chesapeake_Bay_retriever','Chihuahua','Dandie_Dinmont','Doberman'
,'English_foxhound','English_setter','English_springer','EntleBucher'
,'Eskimo_dog','French_bulldog','German_shepherd'
,'German_shorthaired_pointer','Gordon_setter','Great_Dane'
,'Great_Pyrenees','Greater_Swiss_Mountain_dog','Ibizan_hound'
,'Irish_setter','Irish_terrier','Irish_water_spaniel','Irish_wolfhound'
,'Italian_greyhound','Japanese_spaniel','Kerry_blue_terrier'
,'Labrador_retriever','Lakeland_terrier','Leonberg','Lhasa','Maltese_dog'
,'Mexican_hairless','Newfoundland','Norfolk_terrier','Norwegian_elkhound'
,'Norwich_terrier','Old_English_sheepdog','Pekinese','Pembroke'
,'Pomeranian','Rhodesian_ridgeback','Rottweiler','Saint_Bernard','Saluki'
,'Samoyed','Scotch_terrier','Scottish_deerhound','Sealyham_terrier'
,'Shetland_sheepdog','ShihTzu','Siberian_husky'
,'Staffordshire_bullterrier','Sussex_spaniel','Tibetan_mastiff'
,'Tibetan_terrier','Walker_hound','Weimaraner','Welsh_springer_spaniel'
,'West_Highland_white_terrier','Yorkshire_terrier','affenpinscher'
,'basenji','basset','beagle','blackandtan_coonhound','bloodhound'
,'bluetick','borzoi','boxer','briard','bull_mastiff','cairn','chow'
,'clumber','cocker_spaniel','collie','curlycoated_retriever','dhole'
,'dingo','flatcoated_retriever','giant_schnauzer','golden_retriever'
,'groenendael','keeshond','kelpie','komondor','kuvasz','malamute'
,'malinois','miniature_pinscher','miniature_poodle','miniature_schnauzer'
,'otterhound','papillon','pug','redbone','schipperke','silky_terrier'
,'softcoated_wheaten_terrier','standard_poodle','standard_schnauzer'
,'toy_poodle','toy_terrier','vizsla','whippet','wirehaired_fox_terrier'])

# for folders in os.listdir("images"):
#     breed = "".join(folders.split("-")[1:])
#     unique_breeds.append(breed)
    
# # Converts to sorted ndarray
# unique_breeds = np.array(sorted(unique_breeds))

#Image Data Pipeline Function
def image_data_pipeline(path, augment=False, img_size=IMG_SIZE, batch_size=BATCH_SIZE, test_data=False, seed=42):
    """
    Reads images from path, and splits them into training and validation sets
    Create batches of data out of (image x) and (label y) pairs.
    Returns two data batches training_batch and validation_batch.

    Also accepts test data as input (no labels).
    """
    
    # Function for retrieving data
    def retrieve_data_from_path(path, test_data=False):
        filenames = []
        # Test data image retrieval (not separated per class)
        if test_data:
            for files in os.listdir(path):
                filenames.append(f"{path}/{files}")
            return np.array(filenames)
        
        # Train and val data image retrieval
        else:
            for folders in os.listdir(path):
                for files in os.listdir(f"{path}/{folders}"):
                    filenames.append(f"{path}/{folders}/{files}")
            filenames = np.array(filenames)
            np.random.shuffle(filenames)
            labels = np.array(["".join(name.split('/')[-2].split("-")[1:]) for name in filenames])
            unique_breeds = np.unique(labels)
            boolean_labels = np.array([label == unique_breeds for label in labels]).astype(int)
            return filenames, boolean_labels

    #Function for preprocessing
    def process_image(filename, img_size=IMG_SIZE):
        #read image
        image = tf.io.read_file(filename)

        #turn jpeg to numerical Tensor with 3 color channels (RGB)
        image = tf.image.decode_jpeg(image, channels=3)

        #Convert colour channels values 0-255 to 0-1 values.
        #This is a normalization process to make process more efficient.
        image = tf.image.convert_image_dtype(image, tf.float32)

        #Resize to (224,224)
        image = tf.image.resize(image, size=[img_size, img_size])

        return image
    
    # Function for data configuration (for performance) 
    def configure_tensor(ds, shuffle=False):
        if shuffle: # For train set
            ds = ds.shuffle(buffer_size=1000) 
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds
    
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        tf.keras.layers.RandomFlip(mode="horizontal")
    ])
    
    # ----------------------------------------------------------------------------------
    
    # Test data pipeline
    if test_data:
        print(f"Creating test data batches... BATCH SIZE={batch_size}")
        x = retrieve_data_from_path(path, test_data=True)
        x_test = tf.data.Dataset.from_tensor_slices(x).map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return configure_tensor(x_test)
    
    # Train and validation data pipeline
    else:
        print(f"Creating train & validation data batches... BATCH SIZE={batch_size}")
        x, y = retrieve_data_from_path(path)
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=seed)
        x_train = tf.data.Dataset.from_tensor_slices(x_train).map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x_valid = tf.data.Dataset.from_tensor_slices(x_valid).map(process_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if augment:
            x_train = x_train.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            x_valid = x_valid.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        y_train = tf.data.Dataset.from_tensor_slices(y_train)
        y_valid = tf.data.Dataset.from_tensor_slices(y_valid)
        train_data = tf.data.Dataset.zip((x_train, y_train)) 
        valid_data = tf.data.Dataset.zip((x_valid, y_valid)) 
        return configure_tensor(train_data, shuffle=True), configure_tensor(valid_data)


# Lets make a function that convert the array of numbers into a label prediction...
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  max_prob = np.max(prediction_probabilities)*100
  
  return unique_breeds[np.argmax(prediction_probabilities)],max_prob


def pred_label(prediction_probabilities, n=0):
  """
  View the prediction, ground truth and image for sample n
  """
  pred_prob = prediction_probabilities[n],

  # Get pred label
  pred_label,percentage = get_pred_label(pred_prob)
  
  return pred_label,percentage