import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

train_dir = "/kaggle/input/playground-series-s5e6/train.csv"
test_dir = "/kaggle/input/playground-series-s5e6/test.csv"
sample_submission_dir = "/kaggle/input/playground-series-s5e6/sample_submission.csv"

def load_data(directory):
    data = pd.read_csv(directory)
    return data

def feature_engineering(df):
    df = df.copy()
    
    df['N_to_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-5)
    df['N_to_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + 1e-5)
    df['P_to_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + 1e-5)
    
    df['Effective_Nitrogen'] = df['Moisture'] * df['Nitrogen']
    df['Effective_Phosphorous'] = df['Moisture'] * df['Phosphorous']
    
    df['Weather_Index'] = df['Temparature'] * df['Humidity']
    
    df['Temperature_Category'] = pd.cut(
        df['Temparature'], 
        bins=3, 
        labels=['low', 'medium', 'high']
    ).astype(str)
    
    df['Humidity_Category'] = pd.cut(
        df['Humidity'], 
        bins=3, 
        labels=['low', 'medium', 'high']
    ).astype(str)
    
    df['Moisture_Category'] = pd.cut(
        df['Moisture'], 
        bins=3, 
        labels=['low', 'medium', 'high']
    ).astype(str)
    
    print("Feature engineering completed. New features added:")
    print("- Nutrient ratios: N_to_P_ratio, N_to_K_ratio, P_to_K_ratio")
    print("- Moisture-nutrient interactions: Effective_Nitrogen, Effective_Phosphorous")
    print("- Environmental index: Weather_Index")
    print("- Categorical bins: Temperature_Category, Humidity_Category, Moisture_Category")
    
    return df

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
        if type(column.iloc[0]) == str:
            dtype = tf.string
        elif name in cat_cols:
            dtype = tf.int64
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
    
    preprocessed=[]
    
    if num_cols:
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.concatenate([value for key, value in sorted(num_features_dict.items())]))
        num_inputs=[]
        for name in num_cols:
            num_inputs.append(inputs[name])
        num_inputs = tf.keras.layers.Concatenate(axis=-1)(num_inputs)
        num_normalized = normalizer(num_inputs)
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
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_map3(y_true_indices, y_pred_probs, k=3):
    top_k_indices = tf.nn.top_k(y_pred_probs, k=k).indices
    y_true_expanded = tf.expand_dims(y_true_indices, axis=1)
    matches = tf.equal(top_k_indices, y_true_expanded)
    
    precisions = tf.cumsum(tf.cast(matches, tf.float32), axis=1) / tf.range(1, k+1, dtype=tf.float32)
    ap = tf.reduce_sum(precisions * tf.cast(matches, tf.float32), axis=1)
    return tf.reduce_mean(ap)

def master():
    print("Loading Data")
    train_data = load_data(train_dir)
    test_data = load_data(test_dir)
    sample_submission = load_data(sample_submission_dir)
    
    print("Performing Feature Engineering on Training Data")
    train_data_engineered = feature_engineering(train_data)
    
    print("Performing Feature Engineering on Test Data")
    test_data_engineered = feature_engineering(test_data)
    
    print("Split Data")
    X, y = split_data(train_data_engineered)
    
    print(f"Dataset shape after feature engineering: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    print(f"Feature types:\n{X.dtypes}")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Preprocess data")
    inputs, preprocessed_result = preprocess_data(X)
    preprocessor = tf.keras.Model(inputs, preprocessed_result)
    
    label_lookup, label_vocab = process_labels(y_train)
    num_classes = len(label_vocab)
    
    y_train_indices = label_lookup.lookup(tf.constant(y_train.values))
    y_train_encoded = tf.one_hot(y_train_indices, depth=num_classes)
    
    y_val_indices = label_lookup.lookup(tf.constant(y_val.values))
    y_val_encoded = tf.one_hot(y_val_indices, depth=num_classes)
    
    print(f"Preprocessed feature shape: {preprocessed_result.shape}")
    print(f"Number of classes: {num_classes}")
    
    print("Build model")
    model = build_model(inputs, preprocessor, num_classes)
    
    train_ds = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train_encoded))
    train_ds = train_ds.shuffle(len(y_train)).batch(256).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((dict(X_val), y_val_encoded))
    val_ds = val_ds.batch(256).prefetch(tf.data.AUTOTUNE)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5)
    ]
    
    print("Training model")
    history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks, verbose=2)
    
    print("Evaluating Model")
    val_dict = {name: column.values for name, column in X_val.items()}
    val_predictions = model.predict(val_dict, verbose=2)
    map3_score = evaluate_map3(y_val_indices, val_predictions)
    print(f"Validation MAP@3: {map3_score:.4f}")
    
    X_test = test_data_engineered.drop(columns=['id'])
    test_dict = {name: column.values for name, column in X_test.items()}
    print("Predicting on test set")
    predictions = model.predict(test_dict, verbose=2)
    
    top3_indices = tf.nn.top_k(predictions, k=3).indices
    top3_labels = tf.gather(label_vocab, top3_indices)
    
    final_predictions = []
    for row in top3_labels.numpy():
        prediction_str = ' '.join([label.decode('utf-8') for label in row])
        final_predictions.append(prediction_str)
    
    print("Creating Submission file")
    sample_submission['Fertilizer Name'] = final_predictions
    sample_submission.to_csv('submission.csv', index=False)
    
    return model, history, sample_submission

if __name__ == '__main__':
    model, history, sample_submission = master()