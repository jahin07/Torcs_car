import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.layers.core import Reshape, Permute
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras import layers
from keras import models
from keras import backend as K
from keras.layers import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test

i_avg = 0
flag = 0
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        global tmp
        #  tmp1, tmp2,i_avg,flag
        print("Now we build the model")

        S = Input(shape=[state_size])
        S_in = Lambda(lambda img: img / 255.0)(S)

        S_res = Reshape((3, 64, 64))(S_in)
        S_per = Permute((2, 3, 1))(S_res)  # (1, 2, 3) -> (2, 3, 1)


        conv1 = Convolution2D(16, nb_row=8, nb_col=8, subsample=(4, 4), activation='relu')(S_per)
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = Convolution2D(32, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        conv3 = Convolution2D(32, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu')(batch_norm2)
        batch_norm3 = BatchNormalization()(conv3)
        flat = Flatten()(batch_norm3)

        h0 = Dense(HIDDEN1_UNITS, activation='relu')(flat)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        Steering = Dense(1,activation='tanh',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
        # tmp = Steering
        # Steering =(Steering+tmp)/2
        # if flag == 0:
        #     tmp1 = Steering
        #     tmp1 -= tmp1
        #     flag = 1.
        #     tmp2 = Steering
        #
        #  tmp = layers.add([tmp1,Steering])
        # model = models.Model([tmp1,Steering],tmp)
        #
        # Steering = tmp /2
        # # i_avg += 1
        # # if i_avg == 2:
        # #     Steering = tf.mod(tmp1,2)
        # #     tmp2 = Steering
        # #     tmp1 = tf.sub(tmp1,tmp1)
        # #     i_avg = 0
        # #     flag = 1
        # # else:
        # #     Steering = tmp2

        Acceleration = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)   
        Brake = Dense(1,activation='sigmoid',init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1) 
        V = merge([Steering,Acceleration,Brake],mode='concat')          
        model = Model(input=S,output=V)
        print(model.summary())
        return model, model.trainable_weights, S

        '''
        S = Input(shape=state_size)
        S_in = Lambda(lambda img: img / 255.0)(S)
        conv1 = Convolution2D(16, nb_row=8, nb_col=8, subsample=(4, 4), activation='relu')(S_in)
        batch_norm1 = BatchNormalization()(conv1)
        conv2 = Convolution2D(32, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu')(batch_norm1)
        batch_norm2 = BatchNormalization()(conv2)
        conv3 = Convolution2D(32, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu')(batch_norm2)
        batch_norm3 = BatchNormalization()(conv3)
        flat = Flatten()(batch_norm3)
        den = Dense(300, activation='relu')(flat)

        return den, S
        '''
