# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-locals, too-many-arguments, too-many-statements
"""Executor manager."""
from __future__ import absolute_import

import logging
import numpy as np

from mxnet.base import mx_real_t
from mxnet import ndarray as nd
from mxnet.context import cpu
from mxnet.io import DataDesc

import mxnet as mx
import mxnet.autograd as autograd
import perseus.mxnet as perseus

def _split_input_slice(batch_size, work_load_list):
    """Get input slice from the input shape.

    Parameters
    ----------
    batch_size : int
        The number of samples in a mini-batch.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as `ctx`.

    Returns
    -------
    slices : list of slice
        The split slices to get a specific slice.

    Raises
    ------
    ValueError
        In case of too many splits, leading to some empty slices.
    """
    total_work_load = sum(work_load_list)
    batch_num_list = [round(work_load * batch_size / total_work_load)
                      for work_load in work_load_list]
    batch_num_sum = sum(batch_num_list)
    if batch_num_sum < batch_size:
        batch_num_list[-1] += batch_size - batch_num_sum
    slices = []
    end = 0
    for batch_num in batch_num_list:
        begin = int(min((end, batch_size)))
        end = int(min((begin + batch_num, batch_size)))
        if begin >= end:
            raise ValueError('Too many slices. Some splits are empty.')
        slices.append(slice(begin, end))
    return slices

def _check_arguments(symbol):
    """Check the argument names of symbol.
    This function checks the duplication of arguments in Symbol.
    The check is done for feedforward net for now.

    Parameters
    ----------
    symbol : Symbol
        The network configuration.
    """
    arg_set = set()
    arg_names = symbol.list_arguments()
    for name in arg_names:
        if name in arg_set:
            raise ValueError(('Find duplicated argument name \"%s\", ' +
                              'please make the weight name non-duplicated(using name arguments), ' +
                              'arguments are %s') % (name, str(arg_names)))
        arg_set.add(name)

    aux_set = set()
    aux_names = symbol.list_auxiliary_states()
    for name in aux_names:
        if name in aux_set:
            raise ValueError(
                ('Find duplicated auxiliary param name \"%s\", ' +
                 'please make the weight name non-duplicated(using name arguments), ' +
                 'arguments are %s, auxiliary params are %s'
                ) % (name, str(arg_names), str(aux_names)))
        aux_set.add(name)

def _load_general(data, targets):
    """Load a list of arrays into a list of arrays specified by slices."""
    for d_src, d_targets in zip(data, targets):
        if isinstance(d_targets, nd.NDArray):
            d_src.copyto(d_targets)
        else:
            assert d_targets[-1][0].stop == d_src.shape[0], \
                "Batch size miss match. Expected %d, got %d"%( \
                    d_targets[-1][0].stop, d_src.shape[0])
            for slice_idx, d_dst in d_targets:
                d_src[slice_idx].copyto(d_dst)

def _load_data(batch, targets):
    """Load data into sliced arrays."""
    _load_general(batch.data, targets)

def _load_label(batch, targets):
    """Load label into sliced arrays."""
    _load_general(batch.label, targets)

# pylint: disable=too-many-branches
def _bind_exec(sym, ctx, input_shapes, param_names, need_grad=False,
               base_exec=None, shared_data_arrays=None, input_types=None, logger=logging):
    """bind executor for bucketing, potentially sharing data with an existing executor."""
    arg_shape, _, aux_shape = sym.infer_shape(**input_shapes)
    assert(arg_shape is not None)
    if input_types is None:
        input_types = {k: mx_real_t for k in input_shapes.keys()}
    arg_types, _, aux_types = sym.infer_type(**input_types)
    assert(arg_types is not None)

    arg_arrays = []
    grad_arrays = {} if need_grad is not False else None

    arg_names = sym.list_arguments()

    if need_grad is False:
        need_grad = set()
    elif need_grad is True:
        need_grad = set(arg_names) - set(input_shapes.keys())
    elif isinstance(need_grad, set):
        pass
    else:
        raise AssertionError("need_grad must be boolean or set.")
    grad_req = {name:('write' if name in need_grad else 'null') for name in arg_names}


    # create or borrow arguments and gradients
    for i, name in enumerate(arg_names):
        if not name in param_names:
            # data or label
            if shared_data_arrays is not None and \
                    name in shared_data_arrays:
                arg_arr = shared_data_arrays[name]

                if np.prod(arg_arr.shape) >= np.prod(arg_shape[i]):
                    # good, we can share this memory
                    assert(arg_types[i] == arg_arr.dtype)
                    arg_arr = arg_arr.reshape(arg_shape[i])
                else:
                    logger.warning(('bucketing: data "%s" has a shape %s' % (name, arg_shape[i])) +
                                   (', which is larger than already allocated ') +
                                   ('shape %s' % (arg_arr.shape,)) +
                                   ('. Need to re-allocate. Consider putting ') +
                                   ('default_bucket_key to be the bucket taking the largest ') +
                                   ('input for better memory sharing.'))
                    arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])

                    # replace existing shared array because the new one is bigger
                    shared_data_arrays[name] = arg_arr
            else:
                arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                if shared_data_arrays is not None:
                    shared_data_arrays[name] = arg_arr

            arg_arrays.append(arg_arr)
        else:
            # model parameter
            if base_exec is None:
                arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                if name in need_grad:
                    grad_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                    grad_arrays[name] = grad_arr
            else:
                arg_arr = base_exec.arg_dict[name]
                assert arg_arr.shape == arg_shape[i]
                assert arg_arr.dtype == arg_types[i]
                if name in need_grad:
                    grad_arrays[name] = base_exec.grad_dict[name]
            arg_arrays.append(arg_arr)

    # create or borrow aux variables
    if base_exec is None:
        aux_arrays = [nd.zeros(s, ctx, dtype=t) for s, t in zip(aux_shape, aux_types)]
    else:
        for i, a in enumerate(base_exec.aux_arrays):
            assert aux_shape[i] == a.shape
            assert aux_types[i] == a.dtype

        aux_arrays = [a for a in base_exec.aux_arrays]

    executor = sym.bind(ctx=ctx, args=arg_arrays, args_grad=grad_arrays,
                        aux_states=aux_arrays,
                        grad_req=grad_req, shared_exec=base_exec)
    return executor
'''
class DataParallelExecutorGroup(object):
    """A group of executors living on different devices, for data parallelization.

    Parameters
    ----------
    sym: Symbol
        The network configuration.
    arg_names: list of str
        Equals `sym.list_arguments()`
    param_names: list of str
        List of names of all trainable parameters.
    ctx: list of Context
        List of devices for training (data parallelization).
    slices: list of int
        Describes how the data parallelization splits data into different devices.
    train_data: DataIter (or DataBatch)
        The dataset for training. It could be any object with `provide_data` and
        `provide_label` properties. Loading of actual data is not necessarily needed
        at this stage.
    shared_grop: DataParallelExecutorGroup
        An existing executor group, if to share parameters with it.
    """
    def __init__(self, sym, arg_names, param_names, ctx, slices, train_data, shared_group=None):
        # make sure the architecture is valid
        _check_arguments(sym)

        if shared_group is None:
            self.shared_data_arrays = [{} for _ in ctx]
        else:
            self.shared_data_arrays = shared_group.shared_data_arrays

        self.data_names = [x[0] for x in train_data.provide_data]
        self.label_names = [x[0] for x in train_data.provide_label]
        self.aux_names = sym.list_auxiliary_states()
        self.param_idx = [i for i in range(len(arg_names)) if arg_names[i] in param_names]
        self.param_names = [arg_names[i] for i in self.param_idx]

        self.train_execs = []
        for i, ctxi in enumerate(ctx):
            data_shapes = {}
            data_types = {}
            for x in train_data.provide_data + train_data.provide_label:
                data_shapes[x[0]] = tuple([slices[i].stop - slices[i].start] + list(x[1][1:]))
                if isinstance(x, DataDesc):
                    data_types[x.name] = x.dtype
                else:
                    data_types[x[0]] = mx_real_t
            shared_exec = None if shared_group is None else shared_group.train_execs[i]
            train_exec = _bind_exec(sym, ctxi, data_shapes, self.param_names,
                                    need_grad=True, base_exec=shared_exec,
                                    shared_data_arrays=self.shared_data_arrays[i],
                                    input_types=data_types)
            self.train_execs.append(train_exec)

        # data structure
        self.data_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                            for name in self.data_names]
        self.label_arrays = [[(slices[i], e.arg_dict[name]) for i, e in enumerate(self.train_execs)]
                             for name in self.label_names]

        self.param_arrays = [[e.arg_arrays[i] for e in self.train_execs]
                             for i in self.param_idx]
        self.grad_arrays = [[e.grad_arrays[i] for e in self.train_execs]
                            for i in self.param_idx]

        self.aux_arrays = [[e.aux_arrays[i] for e in self.train_execs]
                           for i in range(len(self.aux_names))]

        self.slices = slices

    def load_data_batch(self, data_batch):
        """Load data and labels into arrays."""
        _load_data(data_batch, self.data_arrays)
        _load_label(data_batch, self.label_arrays)

    def forward(self, is_train=False):
        """Perform a forward pass on each executor."""
        for texec in self.train_execs:
            texec.forward(is_train=is_train)

    def backward(self):
        """Perform a backward pass on each executor."""
        for texec in self.train_execs:
            texec.backward()

    def update_metric(self, metric, labels, pre_sliced=False):
        """Update evaluation metric with label and current outputs."""
        for current_exec, (texec, islice) in enumerate(zip(self.train_execs, self.slices)):
            if not pre_sliced:
                labels_slice = [label[islice] for label in labels]
            else:
                labels_slice = labels[current_exec]
            metric.update(labels_slice, texec.outputs)
'''
class ModelParallelExecutorManager(object):
    """ Helper class to manage multiple executors for data parallelism.

    Parameters
    ----------
    symbol : Symbol
        Output symbol.
    ctx : list of Context
        Devices to run on.
    param_names: list of str
        Name of all trainable parameters of the network.
    arg_names: list of str
        Name of all arguments of the network.
    aux_names: list of str
        Name of all auxiliary states of the network.
    train_data : DataIter
        Training data iterator.
    work_load_list : list of float or int, optional
        The list of work load for different devices,
        in the same order as ctx.
    logger : logging logger
        When not specified, default logger will be used.
    sym_gen : A function that generate new Symbols depending on different
        input shapes. Used only for bucketing.
    """
    def __init__(self, kvstore, ctx, args, batchsize,  
                 updater,
                 arg_names, #param_names, aux_names,
                 work_load_list=None, logger=None, sym_gen=None):
        if logger is None:
            logger = logging
        # preparation
        num_device = kvstore.num_workers #size #len(ctx)
        logger.info('Start training with %s', str(ctx))

        if work_load_list is None:
            work_load_list = [1] * num_device
        assert isinstance(work_load_list, list) and len(work_load_list) == num_device, \
            "Invalid settings for work load. "

        #self.slices = slices
        self.args = args
        embedding = args.emb_size
        classes = args.num_classes

        self.arg_names = arg_names
        #self.param_names = param_names
        #self.aux_names = aux_names
        self.ctx = ctx
        self.kvstore = kvstore
        self.rank = kvstore.rank
        self.size = kvstore.num_workers
        self.classes = int(classes / self.size) 
        self.updater = updater
        self.batchsize = batchsize
        self.embedding = embedding
        self.each_gpu_batchsize = int(self.batchsize / self.size)
        self.slices = slice(0, self.each_gpu_batchsize)
        self.each_gpu_slice = slice(self.each_gpu_batchsize*self.rank, self.each_gpu_batchsize*(self.rank+1))
        self.each_gpu_label_slices = [slice(i*self.classes, (i+1) * self.classes) for i in range(self.size)]
        if self.classes * self.size != classes: # if not equal-div
            self.each_gpu_label_slices[-1] = slice((self.size - 1) * self.classes, classes)
        self.each_gpu_label_start_list = nd.array([i.start for i in self.each_gpu_label_slices])
        self.label_start = self.classes * self.rank
        assert self.label_start == self.each_gpu_label_start_list[self.rank]
        self.label_stop = self.classes * (self.rank+1)

        self.data_batch = nd.empty((self.batchsize, self.embedding), ctx=self.ctx[0])
        self.label_batch = nd.empty((self.batchsize, ), ctx=self.ctx[0])
        self.each_gpu_label = nd.empty((self.each_gpu_batchsize, ), ctx=self.ctx[0])
        self.pick_fc_of_cur_gpu = None

        self.allreduce_dict_name = {}

        self.global_max_fc = nd.zeros((self.batchsize,), ctx=self.ctx[0])
        self.global_sum_fc = nd.zeros((self.batchsize,), ctx=self.ctx[0])

        self.logit = nd.zeros((self.batchsize, self.classes),ctx=self.ctx[0])
        self.grad = nd.zeros((self.batchsize, self.classes), ctx=self.ctx[0])
        self.return_each_gpu_grad = nd.zeros((self.each_gpu_batchsize,), ctx=self.ctx[0])
        self.return_feature_grad = nd.zeros((self.batchsize, ), ctx=self.ctx[0])
        self.weight_grad = nd.zeros((self.classes, self.embedding), ctx=self.ctx[0])
        self.bias_grad = nd.zeros((self.classes,), ctx=self.ctx[0])
        self.loss = nd.empty((self.batchsize,), ctx=self.ctx[0])

        # params
        self.weight = nd.empty((self.classes, self.embedding), ctx=self.ctx[0])
        self.bias = nd.empty((self.classes, ), ctx=self.ctx[0])
        self.weight_temp_grad = nd.empty((self.classes, self.embedding), ctx=self.ctx[0])
        self.bias_temp_grad = nd.empty((self.classes,), ctx=self.ctx[0])
        self.weight_norm = nd.empty((self.classes, self.embedding), ctx=self.ctx[0])

        self.fc_output = nd.empty((self.batchsize, self.classes), ctx=self.ctx[0])
        self.pick_fc = None
        self.pick_index = None
        

    def install_monitor(self, monitor):
        """Install monitor on all executors."""
        if self.sym_gen is not None:
            raise NotImplementedError("Monitoring is not implemented for bucketing")

        for train_exec in self.execgrp.train_execs:
            monitor.install(train_exec)

    def set_params(self, weight_param, bias_param): #arg_params, aux_params):
        """Set parameter and aux values.

        Parameters
        ----------
        arg_params : list of NDArray
            Source parameter arrays
        aux_params : list of NDArray
            Source aux arrays.
        """
        weight_param.copyto(self.weight)
        bias_param.copyto(self.bias)
        weight_param.copyto(self.weight_norm)

    def _initialize_kvstore(self):
        self.kvstore.init('weight', self.weight)
        self.kvstore.init('bias', self.bias)
        self.kvstore.pull('weight', self.weight, priority=0) #1)
        self.kvstore.pull('bias', self.bias, priority=-1) #2)

    def update_param(self):
        self.updater(10000, self.weight_grad, self.weight)
        self.updater(1000, self.bias_grad, self.bias)
        self.weight_grad[:] = 0.0
        self.bias_grad[:] = 0.0

    def get_return_each_gpu_grad(self):
        return self.return_each_gpu_grad

    def get_return_each_gpu_loss(self):
        return self.loss #[self.each_gpu_slice]

    def get_return_each_gpu_predict(self):
        '''
        total_max_index = nd.empty(self.batchsize) #nd.empty((1, self.batchsize))
        total_max_value = nd.empty(self.batchsize) #nd.empty((1, self.batchsize))
        #nd_slices = nd.empty((1,))
        total_max_index[:] = nd.argmax(self.logit, axis=1)[:]
        total_max_value[:] = nd.max(self.logit, axis=1)[:] #self.logit[total_max_index[0]]
        #nd_slices[0] = 0
        #max_value = nd.argmax(total_max_value, axis=0)
        #max_index = total_max_index[max_value, nd.arange(self.batchsize)]
        #predict_index = (nd_slices[max_value] + max_index)[:]
        predict_index = total_max_index #max_index[:]
        predict_value = total_max_value #nd.max(total_max_value, axis=0)
        '''
        predict_index = nd.argmax(self.logit, axis=1)[:]
        predict_value = nd.max(self.logit, axis=1)[:]

        # allgather predict 
        global_predict_index = nd.zeros((self.size, self.batchsize))
        global_predict_value = nd.zeros((self.size, self.batchsize))
        use_allreduce_fun = True
        if not use_allreduce_fun:
            if 'global_predict_index' not in self.allreduce_dict_name:
                self.kvstore.init('global_predict_index', global_predict_index, param_only=1)
                self.allreduce_dict_name['global_predict_index'] = 1
            if 'global_predict_value' not in self.allreduce_dict_name:
                self.kvstore.init('global_predict_value', global_predict_value, param_only=1)
                self.allreduce_dict_name['global_predict_value'] = 1
            global_predict_index[self.rank: (self.rank+1)] = predict_index
            global_predict_value[self.rank: (self.rank+1)] = predict_value
            self.kvstore.push('global_predict_value', global_predict_value)
            self.kvstore.pull('global_predict_value', global_predict_value)
            assert len(global_predict_value) > 0, "rank:{}, global_predict_value".format(self.rank)
            self.kvstore.push('global_predict_index', global_predict_index)
            self.kvstore.pull('global_predict_index', global_predict_index)
            assert len(global_predict_index) > 0, "rank:{}, global_predict_index".format(self.rank)
        else:
            global_predict_index[self.rank: (self.rank+1)] = predict_index
            global_predict_value[self.rank: (self.rank+1)] = predict_value
            global_predict_value = self.allreduce('global_predict_value', global_predict_value)
            assert len(global_predict_value) > 0, "rank:{}, global_predict_value".format(self.rank)
            global_predict_index = self.allreduce('global_predict_index', global_predict_index)
            assert len(global_predict_index) > 0, "rank:{}, global_predict_index".format(self.rank)
        
        predict_max_index = nd.argmax(global_predict_value, axis=0)
        predict_max_value = nd.max(global_predict_value, axis=0) # global_predict_value[predict_max_index]
        return_predict_value = self.each_gpu_label_start_list[predict_max_index] + global_predict_index[predict_max_index, nd.arange(self.batchsize)]

        return return_predict_value


    @property
    def param_arrays(self):
        """Shared parameter arrays."""
        # param arrays should be shared by all executor groups
        return self.execgrp.param_arrays
    @property
    def grad_arrays(self):
        """Shared gradient arrays."""
        # grad arrays should be shared by all executor groups
        return self.execgrp.grad_arrays

    @property
    def aux_arrays(self):
        """Shared aux states."""
        # aux arrays are also shared by all executor groups
        return self.execgrp.aux_arrays

    def load_data_batch(self, data_batch, label_batch):
        """Load data and labels into arrays."""
        data_batch.copyto(self.data_batch)
        label_batch.copyto(self.label_batch)
        self.get_each_gpu_label()

    def get_each_gpu_label(self):
        # split label to each gpu
        self.each_gpu_label = ((self.label_batch >= self.label_start) * (self.label_batch < self.label_stop) * (self.label_batch + 1))
        filter_numpy = self.each_gpu_label.asnumpy()
        self.data_of_cur_gpu = nd.array(np.where(filter_numpy>0), ctx=self.ctx[0])
        self.label_of_cur_gpu = nd.array(filter_numpy[filter_numpy>0], ctx=self.ctx[0])
        
        # convert global label to 0-N local
        if self.data_of_cur_gpu.size > 0:
            self.label_of_cur_gpu -= (1.0 + self.label_start)

    def get_weight_norm(self):
        weight_n = []
        norm = self.weight.norm(axis=1).mean().asscalar()
        weight_n += [norm]
        return np.mean(weight_n)

    def forward(self, is_train=False):
        """Run forward on the current executor."""
        #self.curr_execgrp.forward(is_train=is_train)
        
        self.get_each_gpu_label()

        # l2-norm forward
        self.weight_norm = nd.L2Normalization(self.weight, mode='instance')

        # fc forward
        no_bias = True
        if no_bias:
            nd.FullyConnected(data=self.data_batch, weight=self.weight_norm, no_bias=True, num_hidden=self.classes, out=self.fc_output)
        else:
            nd.FullyConnected(data=self.data_batch, weight=self.weight_norm, bias=self.bias, num_hidden=self.classes, out=self.fc_output)
        # margin forward
        self.get_each_gpu_label()
        if self.data_of_cur_gpu.size > 0:
            margin_temp = self.fc_output[self.data_of_cur_gpu, self.label_of_cur_gpu]
            self.pick_fc_of_cur_gpu = margin_temp.copy()
            tem_data = self.margin_loss(self.pick_fc_of_cur_gpu)
            self.fc_output[self.data_of_cur_gpu, self.label_of_cur_gpu] = tem_data[:]
        else:
            self.pick_fc_of_cur_gpu = None
        
        # softmax forward
        # first allreduce sum
        sum_fc = nd.sum(nd.exp(self.fc_output), axis=1)
        sum_fc = self.allreduce('global_sum_fc', sum_fc)
        assert len(sum_fc) > 0, "rank:{}, sum_fc".format(self.rank)
        self.global_sum_fc[:] = sum_fc[:]
        # second allreduce max
        max_fc = nd.max(self.fc_output, axis=1)
        max_fc = self.allreduce('global_max_fc', max_fc, op=perseus.PerseusOp.Max)
        assert len(max_fc) > 0, "rank:{}, max_fc".format(self.rank)
        self.global_max_fc[:] = max_fc[:]
        

    def backward(self):
        """Run backward on the current executor."""
        #self.curr_execgrp.backward()
        # softmax
        self.get_each_gpu_label()

        self.logit = nd.exp(self.fc_output)[:]
        self.logit /= self.global_sum_fc.reshape((self.batchsize, 1))[:]
        self.grad[:] = self.logit[:] #.copy() #[:] #.copy()
        #assert self.data_of_cur_gpu.size > 0 
        if self.data_of_cur_gpu.size > 0:
            self.grad[self.data_of_cur_gpu, self.label_of_cur_gpu] -= 1.0
            self.loss[self.data_of_cur_gpu] = -nd.log(nd.maximum(self.logit[self.data_of_cur_gpu, self.label_of_cur_gpu], 1e-32))[:]
        else:
            #print(self.data_of_cur_gpu)
            pass
            
        # margin
        if self.data_of_cur_gpu.size > 0:
            grad_fc = self.pick_fc_of_cur_gpu
            grad_fc.attach_grad()
            with autograd.record():
                s = self.margin_loss(grad_fc)
            s.backward(self.grad[self.data_of_cur_gpu, self.label_of_cur_gpu])
            self.grad[self.data_of_cur_gpu, self.label_of_cur_gpu] = grad_fc.grad.copy() #[:] #.copy()
        self.pick_fc_of_cur_gpu = None

        # fc
        self.data_batch.attach_grad()
        #self.weight.attach_grad()
        self.weight_norm.attach_grad()
        self.bias.attach_grad()
        with autograd.record():
            no_bias = True
            if no_bias:
                nd.FullyConnected(data=self.data_batch, weight=self.weight_norm, no_bias=True, num_hidden=self.classes, out=self.fc_output)
            else:
                nd.FullyConnected(data=self.data_batch, weight=self.weight_norm, bias=self.bias, num_hidden=self.classes, out=self.fc_output)
        self.fc_output.backward(self.grad)
        self.return_feature_grad = self.data_batch.grad.copy() #[:] #.copy() #[:] #.copy()

        #self.weight_grad += self.weight.grad
        self.weight_temp_grad[:] = self.weight_norm.grad[:]
        #self.bias_grad += self.bias.grad
        
        # allreduce grad
        self.return_feature_grad = self.allreduce('return_feature_grad', self.return_feature_grad)
        assert len(self.return_feature_grad), "rank:{}, grad".format(self.rank)
        #print('all feature grad:', self.return_feature_grad)
        self.return_each_gpu_grad = self.return_feature_grad[self.each_gpu_batchsize * self.rank: self.each_gpu_batchsize * (self.rank + 1)]

        # l2-norm
        self.weight.attach_grad()
        with autograd.record():
            s2 = nd.L2Normalization(self.weight, mode='instance')
        s2.backward(self.weight_temp_grad) #weight_grad)
        self.weight_grad += self.weight.grad

    def update_metric(self, metric, labels, pre_sliced=False):
        """Update metric with the current executor."""
        self.curr_execgrp.update_metric(metric, labels, pre_sliced)

    # define for model parallel
    def margin_loss(self, pick_fc):
        args = self.args
        import math

        m = args.margin_m
        s = args.margin_s 
        assert s > 0.0
        #assert m >= 0.1
        assert m < (math.pi/2)
        
        # cos_t * s
        cos_t = pick_fc / s
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m #sin(pi-m)*m=sin(m)*m
        # threadhold = 0.0
        threshold = math.cos(math.pi - m) # threshold < -cos(m)
        if args.easy_margin:
            cond = nd.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold
            cond = nd.Activation(data=cond_v, act_type='relu')
        body = cos_t*cos_t
        body = 1.0-body
        sin_t = nd.sqrt(body) #mx.sym.sqrt(body)
        new_zy = cos_t*cos_m  # cos(t+m) = c*c - s*s
        b = sin_t*sin_m
        new_zy = new_zy - b
        new_zy = new_zy*s
        if args.easy_margin:
            zy_keep = pick_fc
        else:
            zy_keep = pick_fc - s*mm # zy-s*sin(m)*m = s*cos(t) - s*m*sin(m)
        new_zy = nd.where(cond, new_zy, zy_keep) # cond < 0, zy_keep= s*cos(theta) or s*cos(theta)-s*m*sin(m)
        return new_zy

    def pick_label(self,):
        return 
        
    def allreduce(self, tensor_name, tensor, op=perseus.PerseusOp.Sum):
        # allreduce in-place 
        if tensor_name not in self.allreduce_dict_name:
            self.kvstore.init(tensor_name, tensor, param_only=1)
            self.allreduce_dict_name[tensor_name] = 1
        self.kvstore.push(tensor_name, tensor, op=op)
        self.kvstore.pull(tensor_name, tensor)
        return tensor


