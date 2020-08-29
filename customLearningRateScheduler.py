class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    '''Without this, an AttributeError : 'AccumOptimizer' object has no attribute 'lr' will be thrown'''
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        #print(scheduled_lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %s." % (epoch, scheduled_lr))
