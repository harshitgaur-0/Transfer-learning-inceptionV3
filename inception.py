from keras.applications.inception_v3 import InceptionV3
pre_trained_model = InceptionV3(input_shape=(150,150,3),include_top=False,weights="imagenet")
from keras import layers
import keras
from keras import Model
from keras.optimizers import adam
"""sequential_output = [layer.output for layer in pre_trained_model.layers]
print(sequential_output)
layers_name = [layer.name for layer in pre_trained_model.layers]
print(layers_name)
weights = [layer.weights for layer in pre_trained_model.layers]
print(len(weights))
trainable_weights = [layer.trainable_weights for layer in pre_trained_model.layers]
print(trainable_weights)
print(len(trainable_weights))
non_trainable_weights = [layer.non_trainable_weights for layer in pre_trained_model.layers]
print(len(non_trainable_weights ))"""
for layer in pre_trained_model.layers:
    layer.trainable = False
last_layer_output = pre_trained_model.get_layer("mixed7").output

#model
x = layers.Flatten()(last_layer_output)
x = layers.Dense(1024,activation="relu")(x)
x = layers.Dropout(0.25)(x)
x = layers.Dense(1,activation="sigmoid")(x)

model = Model(pre_trained_model.input,x)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


from keras.preprocessing.image import ImageDataGenerator
train_gen = ImageDataGenerator(rescale=(1/255),horizontal_flip=True,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2)
validation_gen = ImageDataGenerator(rescale=(1/255))
training_data = train_gen.flow_from_directory(r"./train/",target_size=(150,150),class_mode="binary",batch_size=20)
validation_data = validation_gen.flow_from_directory(r"./validation/",target_size=(150,150),class_mode="binary",batch_size=10)

history = model.fit_generator(training_data,validation_data=validation_data,epochs=10,validation_steps=50)
import pickle
pickle.dump(model,open("model.sav","wb"))
pickle.dump(history,open("history.sav","wb"))
