from tensorflow import keras

model = keras.models.load_model("weights/0/last_checkpoint.hdf5", compile=False)

print(model.summary())

# model.save("weights/0/saved_model")