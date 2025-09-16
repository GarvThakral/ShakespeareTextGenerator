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
        inputs = tf.keras.layers.Input(shape=(maxInDs,))
        X = emb_layer(inputs)
        
        # Light dropout after embedding
        X = tf.keras.layers.Dropout(0.2)(X)
        
        # LSTM Block 1 - keep your proven architecture
        X, _, _ = tf.keras.layers.LSTM(
            units=156, 
            activation='tanh', 
            return_sequences=True, 
            return_state=True,
            recurrent_dropout=0.3  # Add recurrent dropout
        )(X)
        X = tf.keras.layers.Dropout(0.5)(X)  # Increase dropout
        
        # LSTM Block 2
        X, _, _ = tf.keras.layers.LSTM(
            units=256, 
            activation='tanh', 
            return_sequences=True, 
            return_state=True,
            recurrent_dropout=0.3
        )(X)
        X = tf.keras.layers.Dropout(0.6)(X)  # Heavy dropout before output
        
        # Output layer
        X = tf.keras.layers.Dense(units=vocab_size, activation='softmax')(X)
        
        model = tf.keras.Model(inputs=inputs, outputs=X)
        return model


    
    compile_config = {
        "loss" : tf.keras.losses.SparseCategoricalCrossentropy(),
        "optimizer" : "adam" ,
        "metrics" : [tf.keras.metrics.SparseCategoricalCrossentropy(),'accuracy']
    }

    model = shakesphereModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_categorical_crossentropy']  # String, not function call
    )
    
    # Warmup + decay schedule
    def lr_schedule(epoch):
        if epoch < 3:
            return 0.01  # High LR for first 3 epochs
        elif epoch < 10:
            return 0.005  # Medium LR
        else:
            return 0.002  # Lower LR for fine-tuning
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    model.fit(
        tf_ds_train,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )
    save_path = "./saved_model"
    model.save(save_path+"/model.keras")
    model.export(save_path+"/data/weights.h5")
    return compile_config , save_path