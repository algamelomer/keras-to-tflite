import tensorflow as tf

def custom_mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

model = tf.keras.models.load_model('model.h5', custom_objects={'mse': custom_mse})

model.compile(optimizer='adam', loss='mse', metrics=['mse'])

input_shape = model.input_shape[1:]  # Get the input shape excluding the batch size

@tf.function(input_signature=[tf.TensorSpec(shape=[None, *input_shape], dtype=tf.float32)])
def model_func(inputs):
    return model(inputs)

converter = tf.lite.TFLiteConverter.from_concrete_functions([model_func.get_concrete_function()])
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as 'model.tflite'")