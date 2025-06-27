import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

train_dir = "/kaggle/input/playground-series-s5e6/train.csv"
test_dir = "/kaggle/input/playground-series-s5e6/test.csv"
sample_submission_dir = "/kaggle/input/playground-series-s5e6/sample_submission.csv"

def load_data(directory):
    data = pd.read_csv(directory)
    return data

def split_data(data):
    data = data.dropna()
    X = data.drop(columns=['id', 'Fertilizer Name'])
    y = data['Fertilizer Name']
    return X, y

def preprocess_data(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_features = X[num_cols]
    num_features_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in dict(num_features).items()}
    
    inputs={}
    for name, column in X.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif name in cat_cols:
            dtype = tf.int64
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.concatenate([value for key, value in sorted(num_features_dict.items())]))
    num_inputs=[]
    for name in num_cols:
        num_inputs.append(inputs[name])
    num_inputs = tf.keras.layers.Concatenate(axis=-1)(num_inputs)
    num_normalized = normalizer(num_inputs)
    preprocessed=[]
    preprocessed.append(num_normalized)
    
    for name in cat_cols:
        vocab = sorted(set(X[name]))
        print(f'name: {name}')
        print(f"vocab: {vocab}\n")
        if type(vocab[0]) is str:
            lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
        else:
            lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
        x = inputs[name]
        x = lookup(x)
        preprocessed.append(x)
    preprocessed_result = tf.keras.layers.Concatenate(axis=1)(preprocessed)
    return inputs, preprocessed_result

def process_labels(y_train):
    vocab = sorted(y_train.unique())
    vocab_tensor = tf.constant(vocab)
    lookup_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocab_tensor, tf.range(len(vocab))),
        default_value=-1
    )
    return lookup_table, vocab_tensor

def build_model(inputs, preprocessor, num_classes):
    x = preprocessor(inputs)
    body = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    result = body(x)
    model = tf.keras.Model(inputs, result)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def master():
    train_data = load_data(train_dir)
    test_data = load_data(test_dir)
    sample_submission = load_data(sample_submission_dir)
    
    X_train, y_train = split_data(train_data)
    inputs, preprocessed_result = preprocess_data(X_train)
    preprocessor = tf.keras.Model(inputs, preprocessed_result)
    
    label_lookup, label_vocab = process_labels(y_train)
    y_indices = label_lookup.lookup(tf.constant(y_train.values))
    y_encoded = tf.one_hot(y_indices, depth=len(label_vocab))
    
    model = build_model(inputs, preprocessor, len(label_vocab))
    
    ds = tf.data.Dataset.from_tensor_slices((dict(X_train), y_encoded))
    ds = ds.shuffle(len(y_train)).batch(512).prefetch(tf.data.AUTOTUNE)
    history = model.fit(ds, epochs=5)
    
    X_test = test_data.drop(columns=['id'])
    test_dict = {name: column.values for name, column in X_test.items()}
    
    predictions = model.predict(test_dict)
    
    # Get top 3 predictions for MAP@3
    top3_indices = tf.nn.top_k(predictions, k=3).indices
    top3_labels = tf.gather(label_vocab, top3_indices)
    
    # Convert to space-delimited strings
    final_predictions = []
    for row in top3_labels.numpy():
        prediction_str = ' '.join([label.decode('utf-8') for label in row])
        final_predictions.append(prediction_str)
    
    sample_submission['Fertilizer Name'] = final_predictions
    sample_submission.to_csv('submission.csv', index=False)
    
    return model, history, final_predictions

if __name__ == '__main__':
    model, history, predictions = master()