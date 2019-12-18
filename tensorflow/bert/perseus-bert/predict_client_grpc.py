# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Send news text to tensorflow_model_server loaded with BERT model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import requests
import tensorflow as tf
import os
import re
import numpy as np
import time

import tokenization
from run_classifier_util import NewsProcessor, convert_examples_to_features, serving_input_fn, InputExample

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('text_dir', '', 'path to text dir, the text file format is tsv.')
tf.app.flags.DEFINE_string('model_name', '', 'Model name to do inference.')
tf.app.flags.DEFINE_string('vocab_file', '', 'path of vocab file.')
FLAGS = tf.app.flags.FLAGS


def main(_):
  processor = NewsProcessor()
  label_list = processor.get_labels()
  predict_examples = processor.get_test_examples(FLAGS.text_dir)
  tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=True)
  features = convert_examples_to_features(predict_examples, label_list, 128, tokenizer)


  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  request = predict_pb2.PredictRequest()
  request.model_spec.name = FLAGS.model_name
  request.model_spec.signature_name = 'serving_default'
  # request.inputs = features
  for i, feature in enumerate(features):
    text = predict_examples[i].text_a
    request.inputs['input_ids'].CopyFrom(
      tf.contrib.util.make_tensor_proto(feature.input_ids, shape=[1, 128]))
    request.inputs['input_mask'].CopyFrom(
      tf.contrib.util.make_tensor_proto(feature.input_mask, shape=[1, 128]))
    request.inputs['segment_ids'].CopyFrom(
      tf.contrib.util.make_tensor_proto(feature.segment_ids, shape=[1, 128]))
    request.inputs['label_ids'].CopyFrom(
      tf.contrib.util.make_tensor_proto(feature.label_id, shape=[1]))
    # start = time.time()
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    predictions = tf.make_ndarray(result.outputs['probabilities'])
    predictions = np.squeeze(predictions)
    top_k = predictions.argsort()[-5:][::-1]
    print('input text: ' + text + ' -> result class is: ' + label_list[top_k[0]])
    # print(result.outputs['probabilities'])
    # stop = time.time()


if __name__ == '__main__':
  tf.app.run()
