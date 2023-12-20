# CNN_Model
CNN
Image Classification using CNN image import warnings warnings.filterwarnings('ignore') import tensorflow as tf from keras.preprocessing.image import ImageDataGenerator from keras.models import Sequential from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense from keras.callbacks import ReduceLROnPlateau, EarlyStopping import matplotlib.pyplot as plt

Path to the training, testing, and prediction datasets
train_path = r"Path_to_your_training_data" test_path = r"Path_to_your_test_data" image_size = (150, 150) batch_size = 32

Data augmentation for the training set
train_datagen = ImageDataGenerator( rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest' )

Data augmentation for the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

Loading the training and testing datasets
train_generator = train_datagen.flow_from_directory( train_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True )

validation_generator = test_datagen.flow_from_directory( test_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True )

Model definition
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3))) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(2, 2))) model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu')) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(2, 2))) model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu')) model.add(BatchNormalization()) model.add(MaxPooling2D(pool_size=(2, 2))) model.add(Dropout(0.2))

model.add(Flatten()) model.add(Dense(512, activation='relu')) model.add(BatchNormalization()) model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax')) # Assuming you have 6 classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Callbacks for training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1) early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

Training the model
history = model.fit( train_generator, steps_per_epoch=train_generator.samples // train_generator.batch_size, validation_data=validation_generator, validation_steps=validation_generator.samples // validation_generator.batch_size, epochs=30, callbacks=[reduce_lr, early_stopping], verbose=1 )

Saving the trained model weights
model.save_weights('model_weights.h5')

Plotting training & validation accuracy
plt.plot(history.history['accuracy']) plt.plot(history.history['val_accuracy']) plt.title('Model accuracy') plt.ylabel('Accuracy') plt.xlabel('Epoch') plt.legend(['Train', 'Validation'], loc='upper left') plt.show()

Plotting training & validation loss
plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('Model loss') plt.ylabel('Loss') plt.xlabel('Epoch') plt.legend(['Train', 'Validation'], loc='upper left') plt.show()

Testing the model on the test set
test_generator = test_datagen.flow_from_directory( test_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False )

test_loss, test_accuracy = model.evaluate(test_generator) print(f"Test Accuracy: {test_accuracy * 100:.2f}%") print(f"Test Loss: {test_loss:.4f}")
