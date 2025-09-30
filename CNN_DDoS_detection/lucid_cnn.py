import tensorflow as tf
import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import os
import csv
import pprint
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lucid_dataset_parser_e import *

import tensorflow.keras.backend as K
from util_functions import *

def calculate_flops(model):
    """
    Calculate the number of Floating Point Operations (FLOPs) for a Keras model
    """
    # Estimate FLOPS based on model layers
    flops = 0
    for layer in model.layers:
        try:
            # Convolutional layers FLOPS calculation
            if isinstance(layer, tf.keras.layers.Conv2D):
                # FLOPs = (2 * input_channels * kernel_height * kernel_width * output_height * output_width * output_channels)
                input_shape = layer.input_shape
                output_shape = layer.output_shape
                kernel_shape = layer.kernel_size
                
                input_channels = input_shape[-1]
                output_channels = layer.filters
                output_height = output_shape[1]
                output_width = output_shape[2]
                
                layer_flops = 2 * input_channels * kernel_shape[0] * kernel_shape[1] * output_height * output_width * output_channels
                flops += layer_flops
            
            # Dense layers FLOPS calculation
            elif isinstance(layer, tf.keras.layers.Dense):
                # FLOPs = (2 * input_neurons * output_neurons)
                input_neurons = layer.input_shape[-1]
                output_neurons = layer.units
                layer_flops = 2 * input_neurons * output_neurons
                flops += layer_flops
        except Exception as e:
            print(f"Could not calculate FLOPS for layer {layer.name}: {e}")
    
    return flops

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)


tf.random.set_seed(SEED)
K.set_image_data_format('channels_last')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config.gpu_options.allow_growth = True  

OUTPUT_FOLDER = "./output/"

VAL_HEADER = ['Model', 'Samples', 'Accuracy', 'F1Score', 'Hyper-parameters', 'Validation Set','Training Time (s)']
PREDICT_HEADER = ['Model', 'Total Prediction Time', 'Packets', 'Samples', 'DDOS%', 'Accuracy', 'F1Score', 
                 'TPR', 'FPR', 'TNR', 'FNR', 'Source', 'Total FLOPs', 'Avg Inference/Sample (ms)']

PATIENCE = 10
DEFAULT_EPOCHS = 50
hyperparamters = {
    "learning_rate": [0.1,0.01,0.001],
    "batch_size": [1024,2048],
    "kernels": [1,2,4,8,16,32,64],
    "regularization" : ['l1','l2'],
    "dropout" : [0.2,0.3,0.4]
}

# Custom Callback to track training and validation accuracy
class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.val_acc = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.train_acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

def plot_accuracy(history, model_name, output_folder):
    """
    Plot and save the accuracy diagram showing train and validation accuracy
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.train_acc, label='Training Accuracy')
    plt.plot(history.val_acc, label='Validation Accuracy')
    plt.title(f'Model Accuracy for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_accuracy_plot.png'))
    plt.close()

def plot_inference_times(model_name, inference_times, batch_sizes, output_folder):
    """
    Plot and save inference time per sample across different batch sizes
    """
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, inference_times, 'o-', linewidth=2)
    plt.title(f'Inference Time per Sample for {model_name}')
    plt.xlabel('Batch Size')
    plt.ylabel('Inference Time per Sample (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{model_name}_inference_times.png'))
    plt.close()

def Conv2DModel(model_name,input_shape,kernel_col, kernels=64,kernel_rows=3,learning_rate=0.01,regularization=None,dropout=None):
    K.clear_session()

    model = Sequential(name=model_name)
    regularizer = regularization

    model.add(Conv2D(kernels, (kernel_rows,kernel_col), strides=(1, 1), input_shape=input_shape, kernel_regularizer=regularizer, name='conv0'))
    if dropout != None and type(dropout) == float:
        model.add(Dropout(dropout))
    model.add(Activation('relu'))

    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', name='fc1'))

    # Print model summary
    model.summary()

    # Calculate and print FLOPS
    flops = calculate_flops(model)
    print(f"Estimated FLOPs: {flops:,}")

    compileModel(model, learning_rate)
    return model

def compileModel(model,lr):
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])  

def main(argv):
    help_string = 'Usage: python3 lucid_cnn.py --train <dataset_folder> -e <epocs>'

    parser = argparse.ArgumentParser(
        description='DDoS attacks detection with convolutional neural networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--train', nargs='+', type=str,
                        help='Start the training process')

    parser.add_argument('-e', '--epochs', default=DEFAULT_EPOCHS, type=int,
                        help='Training iterations')

    parser.add_argument('-cv', '--cross_validation', default=0, type=int,
                        help='Number of folds for cross-validation (default 0)')

    parser.add_argument('-a', '--attack_net', default=None, type=str,
                        help='Subnet of the attacker (used to compute the detection accuracy)')

    parser.add_argument('-v', '--victim_net', default=None, type=str,
                        help='Subnet of the victim (used to compute the detection accuracy)')

    parser.add_argument('-p', '--predict', nargs='?', type=str,
                        help='Perform a prediction on pre-preprocessed data')

    parser.add_argument('-pl', '--predict_live', nargs='?', type=str,
                        help='Perform a prediction on live traffic')

    parser.add_argument('-i', '--iterations', default=1, type=int,
                        help='Predict iterations')

    parser.add_argument('-m', '--model', type=str,
                        help='File containing the model')

    parser.add_argument('-y', '--dataset_type', default=None, type=str,
                        help='Type of the dataset. Available options are: DOS2017, DOS2018, DOS2019, SYN2020')

    args = parser.parse_args()

    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if args.train is not None:
        subfolders = glob.glob(args.train[0] +"/*/")
        if len(subfolders) == 0: 
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)
        
        for full_path in subfolders:
            full_path = full_path.replace("//", "/") 
            folder = full_path.split("/")[-2]
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + '-train.hdf5')
            
            X_val, Y_val = load_dataset(dataset_folder + "/*" + '-val.hdf5')

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            train_file = glob.glob(dataset_folder + "/*" + '-train.hdf5')[0]
            filename = train_file.split('/')[-1].strip()
            time_window = int(filename.split('-')[0].strip().replace('t', ''))
            max_flow_len = int(filename.split('-')[1].strip().replace('n', ''))
            dataset_name = filename.split('-')[2].strip()

            print ("\nCurrent dataset folder: ", dataset_folder)

            model_name = dataset_name + "-LUCID"
            keras_classifier = KerasClassifier(build_fn=Conv2DModel,model_name=model_name, input_shape=X_train.shape[1:],kernel_col=X_train.shape[2])
            rnd_search_cv = GridSearchCV(keras_classifier, hyperparamters, cv=args.cross_validation if args.cross_validation > 1 else 2, refit=True, return_train_score=True)

            # Create custom accuracy history callback
            history = AccuracyHistory()

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=PATIENCE)
            best_model_filename = OUTPUT_FOLDER + str(time_window) + 't-' + str(max_flow_len) + 'n-' + model_name
            mc = ModelCheckpoint(best_model_filename + '.h5', monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
            
            # Early stopping to prevent overfitting
            es = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
   )

            # Include the custom history callback
            start_time = time.time()
            rnd_search_cv.fit(
            X_train,
            Y_train,
            epochs=args.epochs,
            validation_data=(X_val, Y_val),
            callbacks=[es, mc, history],
            verbose=2
      )
                        # ========================== Add this block BELOW rnd_search_cv.fit(...) ==========================
            gs_results_path = os.path.join(OUTPUT_FOLDER, f'{model_name}_gridsearch_results.csv')
            gs_df = pd.DataFrame(rnd_search_cv.cv_results_)
            gs_df.to_csv(gs_results_path, index=False)
            print(f"✅ GridSearchCV results saved to CSV: {gs_results_path}")

            try:
                params_df = pd.json_normalize(gs_df['params'])
                gs_plot_df = pd.concat([params_df, gs_df[['mean_test_score']]], axis=1)

                assert 'dropout' in gs_plot_df.columns, "Missing 'dropout' in Grid Search parameters"
                assert 'learning_rate' in gs_plot_df.columns, "Missing 'learning_rate' in Grid Search parameters"

                plt.figure(figsize=(10, 6))
                sns.lineplot(
                    data=gs_plot_df,
                    x='dropout',
                    y='mean_test_score',
                    hue='learning_rate',
                    style='learning_rate',
                    marker='o',
                    palette='muted'
                )

                for i in range(len(gs_plot_df)):
                    dropout_val = gs_plot_df['dropout'].iloc[i]
                    acc_val = gs_plot_df['mean_test_score'].iloc[i]
                    label = f"{acc_val:.2f}"
                    plt.text(dropout_val, acc_val + 0.005, label, fontsize=8, ha='center')

                plt.title(f'Grid Search Accuracy vs Dropout - {model_name}')
                plt.xlabel('Dropout Rate')
                plt.ylabel('Mean Validation Accuracy')
                plt.ylim(0, 1.05)
                plt.grid(True)
                plt.legend(title='Learning Rate', loc='lower right')
                plt.tight_layout()

                grid_plot_path = os.path.join(OUTPUT_FOLDER, f'{model_name}_gridsearch_plot.png')
                plt.savefig(grid_plot_path)
                plt.close()

                print(f"📊 Grid Search plot saved to: {grid_plot_path}")

            except Exception as e:
                print(f"⚠️ Could not generate Grid Search plot: {e}")
            # ================================================================================================

 
                 
            end_time = time.time()
            training_duration = end_time - start_time 
            # Plot and save the accuracy diagram
            plot_accuracy(history, model_name, OUTPUT_FOLDER)
            
            best_model = rnd_search_cv.best_estimator_.model
            
            best_model.save(best_model_filename + '.h5')

            # Benchmark inference time for different batch sizes
            batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
            inference_times = []
            
            # Use validation set for benchmarking
            for batch_size in batch_sizes:
                # Warmup
                _ = best_model.predict(X_val[:batch_size], batch_size=batch_size)
                
                # Measure
                inferences = []
                for _ in range(5):  # 5 iterations for stability
                    start_time = time.time()
                    _ = best_model.predict(X_val[:batch_size], batch_size=batch_size)
                    end_time = time.time()
                    inferences.append((end_time - start_time) / batch_size * 1000)  # ms per sample
                
                # Average inference time
                inference_times.append(np.mean(inferences))
            
            # Plot inference times
            plot_inference_times(model_name, inference_times, batch_sizes, OUTPUT_FOLDER)

            best_model.summary()
            x=input()
            Y_pred_val = (best_model.predict(X_val) > 0.5)
            Y_true_val = Y_val.reshape((Y_val.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            val_file = open(best_model_filename + '.csv', 'w', newline='')
            val_file.truncate(0)  
            val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
            val_writer.writeheader()
            val_file.flush()
            row = {'Model': model_name, 'Samples': Y_pred_val.shape[0], 'Accuracy': '{:05.4f}'.format(accuracy), 'F1Score': '{:05.4f}'.format(f1_score_val),
                  'Hyper-parameters': rnd_search_cv.best_params_, "Validation Set": glob.glob(dataset_folder + "/*" + '-val.hdf5')[0],'Training Time (s)': '{:06.2f}'.format(training_duration)}
            val_writer.writerow(row)
            val_file.close()

            print("Best parameters: ", rnd_search_cv.best_params_)
            print("Best model path: ", best_model_filename)
            print("F1 Score of the best model on the validation set: ", f1_score_val)

    if args.predict is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        iterations = args.iterations

        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split('/')[-1].strip()
            filename_prefix = model_filename.split('-')[0].strip() + '-' + model_filename.split('-')[1].strip() + '-'
            model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
            model = load_model(model_path)

            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split('/')[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split('/')[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    [packets] = count_packets_in_dataset([X])

                    Y_pred = None
                    Y_true = Y
                    avg_time = 0
                    
                    # Measure FLOPs
                    flops = calculate_flops(model)
                    
                    # Multiple prediction iterations to get average performance
                    Y_pred_list = []
                    times_list = []
                    for iteration in range(iterations):
                        pt0 = time.time()
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                        pt1 = time.time()
                        
                        Y_pred_list.append(Y_pred)
                        times_list.append(pt1 - pt0)

                    # Calculate metrics
                    avg_time = np.mean(times_list)
                    inference_per_sample = avg_time / X.shape[0]
                    
                    # Use the last prediction (typically most representative)
                    Y_pred = Y_pred_list[-1]

                    prediction_metrics = {
                        'prediction_time': avg_time,
                        'flops': flops,
                        'inference_per_sample': inference_per_sample
                    }

                    report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, filename, prediction_metrics, predict_writer)
                    predict_file.flush()

        predict_file.close()

    if args.predict_live is not None:
        predict_file = open(OUTPUT_FOLDER + 'predictions-' + time.strftime("%Y%m%d-%H%M%S") + '.csv', 'a', newline='')
        predict_file.truncate(0)  
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        if args.predict_live is None:
            print("Please specify a valid network interface or pcap file!")
            exit(-1)
        elif args.predict_live.endswith('.pcap'):
            pcap_file = args.predict_live
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split('/')[-1].strip()
        else:
            cap =  pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live

        print ("Prediction on network traffic from: ", data_source)

        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        if args.model is not None and args.model.endswith('.h5'):
            model_path = args.model
        else:
            print ("No valid model specified!")
            exit(-1)

        model_filename = model_path.split('/')[-1].strip()
        filename_prefix = model_filename.split('n')[0] + 'n-'
        time_window = int(filename_prefix.split('t-')[0])
        max_flow_len = int(filename_prefix.split('t-')[1].split('n-')[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split('.')[0].strip()
        model = load_model(args.model)

        mins, maxs = static_min_max(time_window)

        while (True):
            samples = process_live_traffic(cap, args.dataset_type, labels, max_flow_len, traffic_type="all", time_window=time_window)
            if len(samples) > 0:
                X,Y_true,keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X = np.expand_dims(X, axis=3)
                # Measure FLOPs
                flops = calculate_flops(model)
                
                # Multiple prediction iterations
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5, axis=1)
                pt1 = time.time()
                prediction_time = pt1 - pt0
                
                # Calculate metrics
                inference_per_sample = prediction_time / X.shape[0]

                prediction_metrics = {
                    'prediction_time': prediction_time,
                    'flops': flops,
                    'inference_per_sample': inference_per_sample
                }

                [packets] = count_packets_in_dataset([X])
                report_results(np.squeeze(Y_true), Y_pred, packets, model_name_string, data_source, prediction_metrics, predict_writer)
                predict_file.flush()

            elif isinstance(cap, pyshark.FileCapture) == True:
                print("\nNo more packets in file ", data_source)
                break

        predict_file.close()

def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_metrics, writer):
    """
    Enhanced result reporting with more detailed metrics
    
    prediction_metrics is a dictionary containing:
    - prediction_time: Total prediction time
    - flops: Number of floating point operations
    - inference_per_sample: Average inference time per sample
    """
    ddos_rate = '{:04.3f}'.format(sum(Y_pred) / Y_pred.shape[0])

    # Create a base row of metrics
    row = {
        'Model': model_name, 
        'Total Prediction Time': '{:04.3f}'.format(prediction_metrics['prediction_time']), 
        'Packets': packets,
        'Samples': Y_pred.shape[0], 
        'DDOS%': ddos_rate,
        'Source': data_source,
        'Total FLOPs': '{:,}'.format(prediction_metrics['flops']),
        'Avg Inference/Sample (ms)': '{:04.3f}'.format(prediction_metrics['inference_per_sample'] * 1000)
    }

    # Add accuracy metrics if ground truth is available
    if Y_true is not None and len(Y_true.shape) > 0:  
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        
        try:
            tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            row.update({
                'Accuracy': '{:05.4f}'.format(accuracy), 
                'F1Score': '{:05.4f}'.format(f1),
                'TPR': '{:05.4f}'.format(tpr), 
                'FPR': '{:05.4f}'.format(fpr), 
                'TNR': '{:05.4f}'.format(tnr), 
                'FNR': '{:05.4f}'.format(fnr)
            })
        except Exception as e:
            print(f"Error calculating confusion matrix metrics: {e}")
            row.update({
                'Accuracy': '{:05.4f}'.format(accuracy), 
                'F1Score': '{:05.4f}'.format(f1),
                'TPR': 'N/A', 'FPR': 'N/A', 'TNR': 'N/A', 'FNR': 'N/A'
            })
    else:
        row.update({
            'Accuracy': "N/A", 
            'F1Score': "N/A",
            'TPR': "N/A", 
            'FPR': "N/A", 
            'TNR': "N/A", 
            'FNR': "N/A"
        })
    
    # Print and write results
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)

if __name__ == "__main__":
    main(sys.argv[1:])