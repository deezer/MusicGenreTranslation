import os
import numpy as np
import tensorflow as tf
from tag_translation.translators.translator import LogRegTranslator


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None  # Will be set in the input_fn

    def after_create_session(self, session, coord):
        # Initialize the iterator with the data feed_dict
        print("Initializing the iterator")
        self.iterator_initializer_func(session)


def get_weights_from_logdir(logdir):
    checkpoint = tf.train.get_checkpoint_state(logdir)
    loss = None
    W0 = None
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            checkpoint.model_checkpoint_path + '.meta')
        saver.restore(sess, checkpoint.model_checkpoint_path)
        b = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == "b:0"][0]
        W = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == "W:0"][0]
        try:
            W0 = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name == "W0:0"][0]
            W0 = sess.run(W0)
        except IndexError:
            pass
        b = sess.run(b).squeeze()
        W = sess.run(W)
    return W, b, W0, loss


class MapLogReg(LogRegTranslator):

    def __init__(self, model_dir, epochs=500, batch_size=100000, sigma=None, lr=0.5,
                 map0=None, eval_every=100, bias_reg=0.1):
        self.map0 = map0
        self.model_dir = model_dir
        self.epochs = epochs
        self.sigma = sigma
        self.lr = lr
        self.batchsize = batch_size
        self.eval_every = eval_every
        self.bias_reg = bias_reg
        self._built = False
        self.W = None
        self.b = None
        self._weights = []

    def select_sigma(self, train_data):
        if self.sigma is None:
            m = np.mean(np.sum(train_data, axis=1))
            self.sigma = 4 * m ** 4 / 5 ** 4
        print("Choosing sigma = {}".format(self.sigma))

    def build(self, X, y, nb_target_cls, nb_src_cls, trainable=True):
        print("Building model sigma={} bias_reg={}, lr={}, batchsize={}".format(
            self.sigma, self.bias_reg, self.lr, self.batchsize
        ))
        W = tf.get_variable(
            "W", [nb_target_cls, nb_src_cls], initializer=tf.glorot_uniform_initializer,
            trainable=trainable)
        b = tf.get_variable(
            "b", [1, nb_target_cls], initializer=tf.glorot_uniform_initializer,
            trainable=trainable)
        y_pred_logits = tf.matmul(X, W, transpose_b=True) + b

        probas = tf.nn.sigmoid(y_pred_logits)
        loss = None
        loss_prior = None
        loss_classif = None
        if y is not None:
            loss_classif = tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y, logits=y_pred_logits))
            loss = loss_classif
            if self.bias_reg is not None and self.bias_reg != 0:
                loss += self.bias_reg*tf.reduce_sum(b**2)

            if self.map0 is not None:
                print("Using KB prior with shape {}".format(self.map0.shape))
                assert self.map0.shape == (nb_target_cls, nb_src_cls)
                init = tf.constant_initializer(self.map0)
                prior = tf.get_variable("W0", shape=self.map0.shape, initializer=init, trainable=False)
                self.prior = prior
                loss_prior = self.sigma*tf.reduce_sum((W-prior)*(W-prior))
                loss += loss_prior
            else:
                print("Using l2 reg on W")
                loss += 0.5*tf.reduce_sum(W**2)

        self._weights = [W, b]
        self._built = True
        return probas, loss, loss_classif, loss_prior

    def get_model_fn(self, X, y, mode, params):
        nb_target_cls, nb_src_cls = params["nb_target_cls"], params["nb_src_cls"]
        probas, loss, loss_classif, loss_prior = self.build(
            X, y, nb_target_cls, nb_src_cls, trainable=mode == tf.estimator.ModeKeys.TRAIN)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=optimizer.minimize(
                        loss, tf.train.get_or_create_global_step())
                )
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=probas)

    def train_and_evaluate(self, train_data, target_data, eval_data, eval_target, score_function, config_proto={}):
        self.select_sigma(train_data)
        print("Starting to train")
        params = {
              'nb_target_cls': target_data.shape[-1],
              'nb_src_cls': train_data.shape[-1]
              }
        print(params)
        estimator = tf.estimator.Estimator(
            model_fn=lambda features, labels, mode, params: self.get_model_fn(
                features, labels, mode, params),
            model_dir=self.model_dir,
            config=tf.estimator.RunConfig(
                    save_summary_steps=1000,
                    save_checkpoints_steps=1000,
                    log_step_count_steps=1000,
                    session_config=tf.ConfigProto(**config_proto)
                ),
            params=params
            )

        train_fn, train_hook = self.make_input_fn(train_data, target_data)
        for e in range(1, 1+self.epochs//self.eval_every):
            epoch_nb = e*self.eval_every
            print("Epochs {} to {}".format((e-1)*self.eval_every, epoch_nb))
            estimator.train(input_fn=train_fn, hooks=[train_hook])
            print("Done training")
            W, b, W0, loss = get_weights_from_logdir(self.model_dir)
            if self.map0 is not None:
                assert np.allclose(W0, self.map0)
            self.W = W
            self.b = b
            np.save(os.path.join(self.model_dir, "W_{}".format(epoch_nb)), W)
            np.save(os.path.join(self.model_dir, "b_{}".format(epoch_nb)), b)
            print("Evaluating...")
            self._evaluate(eval_data, eval_target, score_function, epoch=epoch_nb)

    def make_input_fn(self, train_data, target_data):
        iterator_initializer_hook = IteratorInitializerHook()

        def _input_fn():
            X_pl = tf.placeholder(train_data.dtype, train_data.shape)
            y_pl = tf.placeholder(target_data.dtype, target_data.shape)
            dataset = tf.data.Dataset.from_tensor_slices((X_pl, y_pl))
            iterator = dataset.batch(self.batchsize).repeat(self.eval_every).make_initializable_iterator()
            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
                iterator.initializer, feed_dict={X_pl: train_data, y_pl: target_data})
            return iterator.get_next()

        return _input_fn, iterator_initializer_hook
