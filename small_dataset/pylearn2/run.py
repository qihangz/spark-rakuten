from pylearn2.utils import serial
train_obj = serial.load_train_file('small_dataset.yaml')
train_obj.main_loop()