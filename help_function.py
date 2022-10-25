from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(data, num_mins=60, step=5,
                 horizon=1, test_train_ratio=0.8,
                 scaler = 'min_max',
                 type_ml ='class'):

    ## initialize
    images = np.zeros([0, num_mins, data.shape[1]])
    if type_ml == 'class':
        labels = np.zeros([0,2])
    if type_ml == 'reg':
        labels = np.zeros([0,1])
    ## for each date
    for idx in range(0, data.shape[0] - (num_mins + horizon), step):
        ## get stock data
        new_image = data[range(idx, idx+(num_mins)),:]

        ## normalize the data
        if scaler == 'min_max':
            col_min = new_image.min(axis=0)
            col_max = new_image.max(axis=0)
            new_image = (new_image - col_min) / (col_max - col_min)
        if scaler == 'std':
            scaler = StandardScaler()
            new_image = scaler.fit_transform(new_image)
        ## save
        images = np.concatenate((images, np.expand_dims(new_image, axis=0)), axis=0)
        ## one-hot labelling: stock price went up or not
        if type_ml == 'class':
            if data[idx+(num_mins+horizon), 0] > data[idx+(num_mins), 0]:
                new_label = [[1.0, 0.0]]
            else:
                new_label = [[0.0, 1.0]]
        if type_ml == 'reg':
            new_label = [[data[idx + (num_mins + horizon), -1]]]
        labels = np.concatenate((labels, new_label), axis=0)

    ## make train-test sets
    train_test_idx = int(test_train_ratio * labels.shape[0])
    x_train = images[:train_test_idx,:,:]
    y_train = labels[:train_test_idx]
    x_test = images[train_test_idx:,:,:]
    y_test = labels[train_test_idx:]

    return x_train, y_train, x_test, y_test

def train_plot_model(x_train, y_train, x_test, y_test, model,
                      type_ml ='class'):

    input_shape = x_train[0].shape
    num_classes = 2


    model.build((None,) + input_shape)

    print(model.summary())

    #Train
    ## hyper-parameters
    BATCH_SIZE = 32
    EPOCHS = 20

    ## compile model with optimizer and loss function
    if type_ml == 'class':
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    if type_ml == 'reg':
        model.compile(loss='mse',
                      optimizer='rmsprop')

    ## fit!
    history = model.fit(x_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=1)

    if type_ml == 'class':

        plt.figure(figsize=(6, 4))
        plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
        plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
        plt.plot(history.history['loss'], "r--", label="Loss of training data")
        plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
        plt.title('Model Accuracy and Loss')
        plt.ylabel('Accuracy and Loss')
        plt.xlabel('Training Epoch')
        plt.ylim(0)
        plt.legend()
        plt.show()

    score = model.evaluate(x_test, y_test, verbose=1)
    if type_ml == 'class':

        print("\nAccuracy on test data: %0.2f" % score[1])
        print("\nLoss on test data: %0.2f" % score[0])
    if type_ml == 'reg':
        print("\nScore on test data: %0.2f" % score)

    import seaborn as sns

    y_pred_test = model.predict(x_test)
    if type_ml == 'class':
        y_true = y_test[:,0]
        y_pred = (y_pred_test[:,0] > 0.5) #.astype(int)

        cf_mat = confusion_matrix(y_true, y_pred)
        f1_score_val = f1_score(y_true, y_pred)

        print("\nPercentage of UP on test data: %0.2f" % np.mean(100*y_true))
        print("\nConfusion matrix on test data:\n", cf_mat)
        print("\nF1 score on test data: %0.2f" % f1_score_val)
    if type_ml == 'reg':
        rmse = np.sqrt(np.mean(((y_pred_test - y_test) ** 2)))
        print('rmse=',rmse)

    return model

