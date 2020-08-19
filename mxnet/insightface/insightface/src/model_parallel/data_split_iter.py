from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
#import sklearn
import time
import datetime
import numpy as np
import cv2
import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet import context
from mxnet.ndarray._internal import _cvimresize as imresize
import mxnet.gluon.data.dataloader as dataloader
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import multiprocessing
import threading
import Queue

logger = logging.getLogger()

def pick_triplets_impl(q_in, q_out):
  more = True
  while more:
      deq = q_in.get()
      if deq is None:
        more = False
      else:
        embeddings, emb_start_idx, nrof_images, alpha = deq
        print('running', emb_start_idx, nrof_images, os.getpid())
        for j in xrange(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                #all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    #triplets.append( (a_idx, p_idx, n_idx) )
                    q_out.put( (a_idx, p_idx, n_idx) )
        #emb_start_idx += nrof_images
  print('exit',os.getpid())

class FaceImageIter(io.DataIter):

    def __init__(self, batch_size, data_shape,
                 path_imgrec=None,
                 shuffle=False,
                 mean=None, 
                 rand_mirror=False, cutoff=0,
                 split_size=1, rank=0,
                 data_extra=None,
                 data_name='data', label_name='softmax_label',
                 **kwargs):
        super(FaceImageIter, self).__init__()
        self.split_size = split_size
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...',
                         path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            logging.info('unpack32 used!')
            header, _ = recordio.unpack(s)
            if header.flag>0:
              logging.info('header0 label %s'%(str(header.label)))
              self.header0 = (int(header.label[0]), int(header.label[1]))
              #assert(header.flag==1)
              self.imgidx = range(1, int(header.label[0]))
              self.id2range = {}
              self.id_num = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
                count = b-a

                self.id_num[identity] = count
              logging.info('id2range %d'%(len(self.id2range)))
            else:
              self.imgidx = list(self.imgrec.keys)
            if shuffle or split_size > 1:
              self.seq = self.imgidx
              self.len_seq = len(self.imgidx)#for local test
              self.oseq = self.imgidx
            else:
              self.seq = None

            if split_size > 1:
                assert rank < split_size
                random.seed(10)
                random.shuffle(self.seq)
                logging.info('[Init]Now rank: %d, and random seq is: %s'%(rank,str(self.seq[:100])))
                epc_size = len(self.seq)
                epc_size_part = epc_size // split_size
                if rank == split_size-1:
                   self.seq = self.seq[rank*epc_size_part : ]
                else:
                   self.seq = self.seq[rank*epc_size_part : (rank+1)*epc_size_part]

        self.mean = mean
        self.nd_mean = None
        if self.mean:
            self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
            self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        #self.provide_label = [(label_name, (batch_size,))]
        self.provide_label = []

        self.batch_size = batch_size
        self.data_shape = data_shape
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        self.cutoff = cutoff

        self.data_extra = None
        if data_extra is not None:
            self.data_extra = nd.array(data_extra)
            self.provide_data = [(data_name, (batch_size,) + data_shape), ('extra', data_extra.shape)]

        self.cur = 0
        self.is_init = False
        self.reset()


    def ____pick_triplets(self, embeddings, nrof_images_per_class):
        pass

    def _pairwise_dists(self, embeddings):
        pass

    def pairwise_dists(self, embeddings):
        pass

    def pick_triplets(self, embeddings, nrof_images_per_class):
        pass

    def __pick_triplets(self, embeddings, nrof_images_per_class):
        pass

    def triplet_oseq_reset(self):
        pass

    def time_reset(self):
        self.time_now = datetime.datetime.now()

    def time_elapsed(self):
        time_now = datetime.datetime.now()
        diff = time_now - self.time_now
        return diff.total_seconds()

    def select_triplets(self):
        pass

    def triplet_reset(self):
        self.select_triplets()

    def hard_mining_reset(self):
        pass

    def reset_c2c(self):
        pass

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        print('call reset()')
        self.cur = 0

        if self.shuffle:
          random.seed(10)
          random.shuffle(self.seq)

        if self.split_size > 1:
           self.seq = self.seq[:self.len_seq]

        if self.seq is None and self.imgrec is not None:
            self.imgrec.reset()

    @property
    def num_samples(self):
      return len(self.seq)

    def next_sample(self, index=None, lock=None, imgrec=None):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          while True:
            idx = self.seq[index]
            if imgrec is not None:
              lock.acquire()
              s = imgrec.read_idx(idx)
              lock.release()
              header, img = recordio.unpack(s)
              label = header.label
              return label, img, None, None
            elif self.imgrec is not None:
              if self.cur >= len(self.seq):
                  raise StopIteration
              idx = self.seq[self.cur]
              self.cur += 1

              s = self.imgrec.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              return label, img, None, None
            else:
              label, fname, bbox, landmark = self.imglist[idx]
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

    def brightness_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      src *= alpha
      src = nd.clip(src, 0, 255)
      return src

    def contrast_aug(self, src, x):
        alpha = 1.0 + random.uniform(-x, x)
        coef = np.array([[[0.299, 0.587, 0.114]]])
        gray = src * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * nd.sum(gray)
        src *= alpha
        src += gray
        src = nd.clip(src, 0, 255)
        return src

    def saturation_aug(self, src, x):
      alpha = 1.0 + random.uniform(-x, x)
      coef = np.array([[[0.299, 0.587, 0.114]]])
      gray = src * coef
      gray = np.sum(gray, axis=2, keepdims=True)
      gray *= (1.0 - alpha)
      src *= alpha
      src += gray
      return src

    def color_aug(self, img, x):
        augs = [self.brightness_aug, self.contrast_aug, self.saturation_aug]
        random.shuffle(augs)
        for aug in augs:
            #print(img.shape)
            img = aug(img, x)
            #print(img.shape)
        return img

    def mirror_aug(self, img):
      _rd = random.randint(0,1)
      if _rd==1:
        for c in xrange(img.shape[2]):
          img[:,:,c] = np.fliplr(img[:,:,c])
      return img

    def next(self, lock=None, imgrec=None, index=None):
        """Returns the next batch of data."""
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        # if self.provide_label is not None:
        if self.provide_label:
            batch_label = nd.empty(self.provide_label[0][1])
        else:
            batch_label = nd.empty((batch_size,))
        if index is None:
            index = random.sample(range(0, len(self.seq)), batch_size)
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample(index[i], lock, imgrec)
                _data = self.imdecode(s)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                  _data = _data.astype('float32')
                  _data -= self.nd_mean
                  _data *= 0.0078125
                if self.cutoff>0:
                  centerh = random.randint(0, _data.shape[0]-1)
                  centerw = random.randint(0, _data.shape[1]-1)
                  half = self.cutoff//2
                  starth = max(0, centerh-half)
                  endh = min(_data.shape[0], centerh+half)
                  startw = max(0, centerw-half)
                  endw = min(_data.shape[1], centerw+half)
                  _data = _data.astype('float32')
                  _data[starth:endh, startw:endw, :] = 127.5
                #_data = self.augmentation_transform(_data)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    if not isinstance(label, numbers.Number):
                        label = label[0]
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size: # same as last_batch_handle='discard'
                raise StopIteration
        _label = None
        if self.provide_label is not None:
            _label = [batch_label]
        if self.data_extra is not None:
            return io.DataBatch([batch_data, self.data_extra], _label, batch_size - i)
        else:
            return io.DataBatch([batch_data], _label, batch_size - i)
        

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s) #mx.ndarray
        return img

    def read_image(self, fname):
        """Reads an input image `fname` and returns the decoded raw bytes.

        Example usage:
        ----------
        >>> dataIter.read_image('Face.jpg') # returns decoded raw bytes.
        """
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """Transforms input data with specified augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

class FaceImageIterList(io.DataIter):
  def __init__(self, iter_list):
    assert len(iter_list)>0
    self.provide_data = iter_list[0].provide_data
    self.provide_label = iter_list[0].provide_label
    self.iter_list = iter_list
    self.cur_iter = None

  def reset(self):
    self.cur_iter.reset()

  def next(self):
    self.cur_iter = random.choice(self.iter_list)
    while True:
      try:
        ret = self.cur_iter.next()
      except StopIteration:
        self.cur_iter.reset()
        continue
      return ret

class PrefetchingIter(io.DataIter):
    """Performs pre-fetch for other data iterators.

    This iterator will create another thread to perform ``iter_next`` and then
    store the data in memory. It potentially accelerates the data read, at the
    cost of more memory usage.

    Parameters
    ----------
    iters : DataIter or list of DataIter
        The data iterators to be pre-fetched.
    rename_data : None or list of dict
        The *i*-th element is a renaming map for the *i*-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data.
    rename_label : None or list of dict
        Similar to ``rename_data``.

    Examples
    --------
    >>> iter1 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> iter2 = mx.io.NDArrayIter({'data':mx.nd.ones((100,10))}, batch_size=25)
    >>> piter = mx.io.PrefetchingIter([iter1, iter2],
    ...                               rename_data=[{'data': 'data_1'}, {'data': 'data_2'}])
    >>> print(piter.provide_data)
    [DataDesc[data_1,(25, 10L),<type 'numpy.float32'>,NCHW],
     DataDesc[data_2,(25, 10L),<type 'numpy.float32'>,NCHW]]
    """
    def __init__(self, iters, prefetch_process=1, capacity=2, rank=0, shuffle=False):
        super(PrefetchingIter, self).__init__()
        self.n_iter = iters.num_samples #len(iters)
        assert self.n_iter > 0
        self.iters = iters
        self.batch_size = self.provide_data[0][1][0]
        self.rank = rank
        self.batch_counter = 0

        if hasattr(self.iters, 'epoch_size'):
            self.epoch_size = self.iters.epoch_size
            if self.iters.epoch_size is None:
                self.epoch_size = int(self.iters.num_samples/self.batch_size)
        else:
            self.epoch_size = int(self.iters.num_samples/self.batch_size)

        self.next_iter = 0
        self.prefetch_process = prefetch_process
        self.shuffle = shuffle

        self._data_queue = dataloader.Queue(maxsize=capacity)
        self._data_buffer = Queue.Queue(maxsize=capacity*2)
        self._index_queue = multiprocessing.Queue()

        self.prefetch_reset_event = multiprocessing.Event()
        self.epoch_end_event = multiprocessing.Event()
        self.next_reset_event = threading.Event()

        self.lock = multiprocessing.Lock()
        self.imgrec = self.iters.imgrec

        def prefetch_func(data_queue, event, end_event):
            while True:
                if event.is_set() and (not end_event.is_set()):
                    #index = []
                    #i = 0
                    #while i < self.batch_size:
                    #    try:
                    #        index.append(self._index_queue.get())
                    #        i += 1
                    #    except:
                    #        end_event.set()
                    #if i == self.batch_size:
                    index = None
                    try:
                        index = self._index_queue.get()
                    except:
                        end_event.set()

                    if index is not None and len(index) == self.batch_size:
                        next_data = self.iters.next(self.lock, self.imgrec, index)
                        data_queue.put((dataloader.default_mp_batchify_fn(next_data.data[0]),
                                    dataloader.default_mp_batchify_fn(next_data.label[0])))

        def next_func(data_queue, event):
            while True:
                if event.is_set():
                    batch, label = data_queue.get(block=True)
                    batch = dataloader._as_in_context(batch, context.cpu())
                    label = dataloader._as_in_context(label, context.cpu())
                    label = label.reshape((label.shape[0],))
                    self._data_buffer.put((batch, label))

        # producer next
        self.produce_lst = []
        for ith in range(prefetch_process):
            p_process = multiprocessing.Process(target=prefetch_func,
                                                args=(self._data_queue, self.prefetch_reset_event,
                                                      self.epoch_end_event))
            p_process.daemon = True
            p_process.start()
            self.produce_lst.append(p_process)

        # consumer get
        self.data_buffer = {}
        self.prefetch_thread = threading.Thread(target=next_func,
                                                args=(self._data_queue, self.next_reset_event))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        # first epoch
        self.reset()

    def __del__(self):
        self.__clear_queue()

        for i_process in self.produce_lst:
            i_process.join()
        self.prefetch_thread.join()

    def __clear_queue(self):
        """ clear the queue"""
        while True:
            try:
                self._data_queue.get_nowait()
            except:
                break
        while True:
            try:
                self._data_buffer.get_nowait()
            except:
                break
        while True:
            try:
                self._index_queue.get_nowait()
            except:
                break

    @property
    def provide_data(self):
        return self.iters.provide_data

    @property
    def provide_label(self):
        return self.iters.provide_label

    def reset(self):
        self.epoch_end_event.set()
        self.next_iter = 0
        self.iters.reset()
        self.__clear_queue()

        assert self._index_queue.empty()
        logging.info("Prefetch Dataiter Inqueue")
        seq_index = range(0, len(self.iters.seq))
        if self.shuffle:
            random.shuffle(seq_index)
        #for index in range(0, len(self.iters.seq)):
        #    self._index_queue.put(seq_index[index])
        for index in range(0, len(self.iters.seq), self.batch_size):
            self._index_queue.put(seq_index[index:index+self.batch_size])
        logging.info("Queue Reset Done")

        self.prefetch_reset_event.set()
        self.next_reset_event.set()
        self.epoch_end_event.clear()

    def iter_next(self):
        self.next_iter += 1
        if self.next_iter > self.epoch_size:
            self.prefetch_reset_event.clear()
            self.next_reset_event.clear()
            return False
        else:
            return True

    def next(self):
        if self.iter_next():
            self.batch_counter += 1
            batch, label = self._data_buffer.get(block=True)
            return io.DataBatch(data=[batch], label=[label], pad=0)
        else:
            raise StopIteration

def PrefetchFaceIter(prefetch_process=1, capacity=2, prefetch=False, **kwargs):
    if prefetch:
      iters = PrefetchingIter(
              FaceImageIter(**kwargs),
              prefetch_process, capacity,
              shuffle=kwargs['shuffle'],
              rank=kwargs['rank'],
              )
      import atexit
      atexit.register(lambda a : a.__del__(), iters)
    else:
      iters = FaceImageIter(**kwargs)
      iters.epoch_size=int(iters.num_samples/iters.provide_data[0][1][0])
    return iters



