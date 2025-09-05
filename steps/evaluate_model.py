from zenml import step
import tensorflow as tf
import datasets as ds
from typing import Dict , Any
@step
def evaluate_model(processed_ds_test:ds.Dataset ,compile_config:Dict, save_path = "./save_model")->Any:
    tf_ds_test = processed_ds_test.to_tf_dataset(
        columns = ['features'],
        label_cols=['labels'],
        batch_size = 10,
        shuffle = True
    )
    model = tf.keras.models.load_model(save_path+"/model.keras")
    model.compile(**compile_config)
    result = model.evaluate(tf_ds_test)
    return result