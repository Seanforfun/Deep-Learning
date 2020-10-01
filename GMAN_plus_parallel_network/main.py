

from path_dataloader import data_path, dataloader
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam
from dehaze_net import gman_net
from train import train_model


# Hyperparameters
epochs = 10
batch_size = 8
k_init = tf.keras.initializers.random_normal(stddev=0.008, seed = 101)      
regularizer = tf.keras.regularizers.L2(1e-4)
b_init = tf.constant_initializer()

train_data, val_data = data_path(orig_img_path = '../input/dehaze/clear_images', hazy_img_path = '../input/dehaze/haze')
train, val = dataloader(train_data, val_data, batch_size)

optimizer = Adam(learning_rate = 1e-4)
net = gman_net()

train_loss_tracker = tf.keras.metrics.MeanSquaredError(name = "train loss")
val_loss_tracker = tf.keras.metrics.MeanSquaredError(name = "val loss")


train_model(epochs, train, val, net, train_loss_tracker, val_loss_tracker, optimizer)
