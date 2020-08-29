class AccumOptimizer(Optimizer):
    #name
    def __init__(self,optimizer,steps_per_update=1,**kwargs):
        self.name=kwargs['name']
        super(AccumOptimizer,self).__init__(**kwargs)
        self.optimizer=optimizer
        with tf.name_scope(self.__class__.__name__):
            self.steps_per_update=steps_per_update
            self.iterations=tf.Variable(0,dtype='int64',name='iterations')
            self.cond=tf.equal(self.iterations%self.steps_per_update,0)
            #self.lr=self.optimizer.lr
            self.learning_rate = self.optimizer.learning_rate #tf.Variable(initial_value=self.optimizer.lr.numpy(),name='lr')
            self.optimizer.learning_rate=tf.cond(self.cond,lambda:self.optimizer.learning_rate.value(), lambda:0.)
            for attr in ['momentum', 'rho', 'beta_1', 'beta_2']:
                if hasattr(self.optimizer,attr):
                    value=getattr(self.optimizer,attr)
                    setattr(self, attr, value)
                    setattr(self.optimizer, attr, tf.cond(self.cond, lambda:value.value(), lambda:1 - 1e-7))
            for attr in self.optimizer.get_config():
                #print(attr)
                if not hasattr(self, attr):
                    value = getattr(self.optimizer, attr)
                    setattr(self, attr, value)
            # debug
            self._create_slots=self.optimizer._create_slots
            self._resource_apply_dense=self.optimizer._resource_apply_dense
            
            #Override the original method of obtaining gradients, pointing to cumulative gradients
            # Cover the original get_gradients method with accumulative gradients.
            def get_gradients(loss,params):
                return [ag / self.steps_per_update for ag in self.accum_grads]
            self.optimizer.get_gradients = get_gradients
            
    def get_updates(self,loss,params):
        self.learning_rate = self.optimizer.learning_rate
        self.iterations=tf.add(self.iterations, 1)
        self.optimizer.iterations=tf.add(self.optimizer.iterations, tf.cast(self.cond, 'int64'))
        self.updates=[
            self.iterations,
            self.optimizer.iterations
        ]
        #  (gradient accumulation)
        self.accum_grads = [tf.zeros(p.shape,dtype=p.dtype) for p in params]
        grads = self.get_gradients(loss, params)
                                     
        for g, ag in zip(grads, self.accum_grads):
            self.updates.append(ag=tf.cond(self.cond,lambda:g,lambda:ag+g))
        
        # optimizer (inheriting updates of original optimizer)
        self.updates.extend(self.optimizer.get_updates(loss, params)[1:])
        self.weights.extend(self.optimizer.weights)
        
        return self.updates     
    
    def get_config(self):
        config = self.optimizer.get_config()
        return config
