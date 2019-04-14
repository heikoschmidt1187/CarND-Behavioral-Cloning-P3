from preprocessor import Preprocessor
from nvidiamodel import NvidiaModel

if __name__ == '__main__':
    # create samples to hold the paths
    preprocessor = Preprocessor('./training_data/')

    # create model for CNN
    model = NvidiaModel(preprocessor)

    ### neural network and training ###
    parameter_set = {
        # size for batch processing
        'batch_size' : 16,
        # learning rate
        'learning_rate' : 0.001,
        # validation set size in percent
        'validation_set' : 0.2,
        # epochs for training
        'epochs' : 10
    }

    # train the model
    model.train_model(parameter_set, 'model.h5')
