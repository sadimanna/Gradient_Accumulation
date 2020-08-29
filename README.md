# Gradient_Accumulation

**Optimizer with Gradient Accumulation and Custom Learning Rate Scheduler**

This repository is an extension of the works in https://github.com/bojone/accum_optimizer_for_keras

Building on the works done in that repository and another file contributed via *Pull Request* which seems to be there unmerged since June 1st, 2020. So, instead of adding another *Pull Request* (which may not be accepted in my lifetime!) the author of these files decided to create another repo, so that the changes I made should not remain hidden from the rest of the Human Civilization. Link to that very useful file in _Pull Request_ : https://github.com/bojone/accum_optimizer_for_keras/blob/c5dee50757192458819a3aba67df399154e280ed/accum_optimizer_tf2.ipynb

Without the Custom Learning Rate Scheduler, an **AttributeError : 'AccumOptimizer' object has no attribute 'lr'** will be thrown. This happens if we use _tf.keras.callbacks.LearningRateScheduler_ in _model.fit()_. Apparently, even after creating the attribute _self.lr = self.optimizer.lr_, the error persists. Printing using _print(self.lr)_ gives the error too.


