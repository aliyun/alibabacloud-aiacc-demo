# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, use_perseus, use_horovod, use_fp16=False, use_amp=False):
  """Creates an optimizer training op."""
  # loss scale may need adjustment for new datasets
  loss_scale = 128.0 if use_fp16 else 1.0
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  # optimizer = AdamWeightDecayOptimizer(
  optimizer = LAMBOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  if use_horovod:
    import horovod.tensorflow as hvd
    optimizer = hvd.DistributedOptimizer(optimizer, compression=hvd.Compression.fp16)
  elif use_perseus:
    import perseus.tensorflow.horovod as hvd
    optimizer = hvd.DistributedOptimizer(optimizer)
  
  if use_amp:
    optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

  tvars = tf.trainable_variables()

  if use_horovod or use_perseus:
    if loss_scale != 1.0:
      grads_and_vars  = optimizer.compute_gradients(loss*loss_scale, tvars)
    else:
      grads_and_vars  = optimizer.compute_gradients(loss, tvars)
    grads = list()
    for grad, var in grads_and_vars:
      grads.append(grad)
    if loss_scale != 1.0:
      grads = [tf.math.scalar_mul(1.0/loss_scale,grad) for grad in grads]   
  else:
    if loss_scale != 1.0:
      grads = tf.gradients(loss*loss_scale, tvars)
      grads = [tf.math.scalar_mul(1.0/loss_scale, grad) for grad in grads]
    else:
      grads = tf.gradients(loss, tvars)

  # Convert all gradients to fp32 so clip_by_global_norm will work with
  # fp16 tensors.
  # clip_by_global_norm requires all gradients to be of same dtype.
  grads_fp32 = []
  for grad in grads:
    if grad is not None:
      grads_fp32.append(tf.cast(grad, tf.float32))
    else:
      grads_fp32.append(None)
  grads = grads_fp32

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None, use_fp16=False):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      has_shadow = use_fp16 and param.dtype.base_dtype != tf.float32
      if has_shadow:
        # create shadow fp32 weights for fp16 variable
        param_fp32 = tf.get_variable(
            name=param_name + "/shadow",
            dtype=tf.float32,
            trainable=False,
            initializer=tf.cast(param.initialized_value(),tf.float32))
      else:
        param_fp32 = param

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param_fp32

      update_with_lr = self.learning_rate * update

      next_param = param_fp32 - update_with_lr

      if has_shadow:
        # cast shadow fp32 weights to fp16 and assign to trainable variable
        param.assign(tf.cast(next_param, param.dtype.base_dtype))
      assignments.extend(
          [param_fp32.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


class LAMBOptimizer(tf.train.Optimizer):
  """
  LAMBOptimizer optimizer. 
  https://github.com/ymcui/LAMB_Optimizer_TF
  # IMPORTANT NOTE
  - This is NOT an official implementation.
  - LAMB optimizer is changed from arXiv v1 ~ v3.
  - We implement v3 version (which is the latest version on June, 2019.).
  - Our implementation is based on `AdamWeightDecayOptimizer` in BERT (provided by Google).
  
  # References
  - Large Batch Optimization for Deep Learning: Training BERT in 76 minutes. https://arxiv.org/abs/1904.00962v3
  - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. https://arxiv.org/abs/1810.04805
  # Parameters
  - There is nothing special, just the same as `AdamWeightDecayOptimizer`.

  implementation from NVIDIA at: 
  https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/optimization.py#L284
  """

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.01,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None, use_fp16=False):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      has_shadow = use_fp16 and param.dtype.base_dtype != tf.float32
      if has_shadow:
        # create shadow fp32 weights for fp16 variable
        param_fp32 = tf.get_variable(
            name=param_name + "/shadow",
            dtype=tf.float32,
            trainable=False,
            initializer=tf.cast(param.initialized_value(),tf.float32))
      else:
        param_fp32 = param

      m = tf.get_variable(
          name=param_name + "/lamb_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/lamb_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param_fp32

      ############## BELOW ARE THE SPECIFIC PARTS FOR LAMB ##############

      # Note: Here are two choices for scaling function \phi(z)
      # minmax:   \phi(z) = min(max(z, \gamma_l), \gamma_u)
      # identity: \phi(z) = z
      # The authors does not mention what is \gamma_l and \gamma_u
      # UPDATE: after asking authors, they provide me the code below.
      # ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
      #      math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      r1 = tf.sqrt(tf.reduce_sum(tf.square(param_fp32)))
      r2 = tf.sqrt(tf.reduce_sum(tf.square(update)))

      r = tf.where(tf.greater(r1, 0.0),
                   tf.where(tf.greater(r2, 0.0), 
                            r1 / r2,
                            1.0),
                   1.0)

      eta = self.learning_rate * r

      update_with_lr = eta * update

      next_param = param_fp32 - update_with_lr

      if has_shadow:
        # cast shadow fp32 weights to fp16 and assign to trainable variable
        param.assign(tf.cast(next_param, param.dtype.base_dtype))

      assignments.extend(
          [param_fp32.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name



'''
class LAMBOptimizer(tf.train.Optimizer):
  """A LAMB optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = tf.identity(learning_rate, name='learning_rate')
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.steps = 0

  def apply_gradients(self, grads_and_vars, global_step=None, name=None,
      use_fp16=False):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)
      has_shadow = use_fp16 and param.dtype.base_dtype != tf.float32
      if has_shadow:
        # create shadow fp32 weights for fp16 variable
        param_fp32 = tf.get_variable(
            name=param_name + "/shadow",
            dtype=tf.float32,
            trainable=False,
            initializer=tf.cast(param.initialized_value(),tf.float32))
      else:
        param_fp32 = param

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # LAMB update
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      self.steps += 1
      beta1_correction = (1 - self.beta_1 ** self.steps)
      beta2_correction = (1 - self.beta_2 ** self.steps)

      next_m_unbiased = next_m / beta1_correction
      next_v_unbiased = next_v / beta2_correction

      update = next_m_unbiased / (tf.sqrt(next_v_unbiased) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param_fp32

      w_norm = linalg_ops.norm(param, ord=2)
      g_norm = linalg_ops.norm(update, ord=2)
      ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
          math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      update_with_lr = ratio * self.learning_rate * update

      next_param = param_fp32 - update_with_lr

      if has_shadow:
        # cast shadow fp32 weights to fp16 and assign to trainable variable
        param.assign(tf.cast(next_param, param.dtype.base_dtype))
      assignments.extend(
          [param_fp32.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
'''
