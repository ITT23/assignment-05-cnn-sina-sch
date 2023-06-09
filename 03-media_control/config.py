# include only those gestures
CONDITIONS = ['like', 'stop', 'dislike']

# image size
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3

# hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
ACTIVATION = 'relu'
ACTIVATION_CONV = 'LeakyReLU'  # LeakyReLU
LAYER_COUNT = 2
NUM_NEURONS = 64