from zenml import step
import tensorflow as tf
from tensorflow.keras.layers import LSTM , Input , Embedding , Dropout , Dense
from tensorflow.keras import Model
import datasets as ds
from typing import Tuple , Dict , Any
@step
def train_model(processed_ds_train:ds.Dataset  , maxInDs:int, vocab_size:int)->Tuple[Dict[str,Any] , str]:
    tf_ds_train = processed_ds_train.to_tf_dataset(
        columns = ['features'],
        label_cols=['labels'],
        batch_size = 10,
        shuffle = True
    )


    def embedding_layer_generator():
        embedding_layer = Embedding(
            input_dim = vocab_size,
            output_dim = 300,
            mask_zero = True
        )
        return embedding_layer

    emb_layer = embedding_layer_generator()

    def shakesphereModel():
        inputs = tf.keras.layers.Input(shape = (maxInDs,))
        X = emb_layer(inputs)
        # LSTM Block 1
        X , _ , _ = LSTM(units = 156,activation = 'tanh' , return_sequences=True , return_state = True)(X)
        X = Dropout(0.4)(X)
        X , _ , _ = LSTM(units = 256,activation = 'tanh' , return_sequences=True ,return_state = True)(X)
        X = Dropout(0.4)(X)
        # X = LSTM(units = 256,activation = 'tanh' , return_sequences=True)(X)
        # X = Dropout(0.4)(X)
        X = Dense(units = vocab_size , activation = 'softmax')(X)
        outputs = X
        model = Model(inputs = inputs , outputs = outputs)
        return model
    
    compile_config = {
        "loss" : tf.keras.losses.SparseCategoricalCrossentropy(),
        "optimizer" : "adam" ,
        "metrics" : [tf.keras.metrics.SparseCategoricalCrossentropy(),'accuracy']
    }

    model = shakesphereModel()
    model.compile(**compile_config)
    model.fit(tf_ds_train,epochs=1)
    save_path = "./saved_model"
    model.save(save_path+"/model.keras")
    model.export(save_path+"/data/weights.h5")
    return compile_config , save_path