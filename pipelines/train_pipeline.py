from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.preprocess_data import preprocessing_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
@pipeline
def train_pipeline():
    ds , maxInDs = ingest_data()
    processed_ds_train , processed_ds_test , maxInDs , index_to_word , word_to_index , vocab_size = preprocessing_data(ds , maxInDs)
    compile_config , save_path = train_model(processed_ds_train  , maxInDs , vocab_size)
    result = evaluate_model(processed_ds_test,compile_config,save_path)
    return result
print(train_pipeline())