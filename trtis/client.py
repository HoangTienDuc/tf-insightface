import argparse
import numpy as np
import os
import colorsys
from builtins import range
from tensorrtserver.api import *
import tensorrtserver.api.model_config_pb2 as model_config
import time
import cv2


path = './test.jpg'
dropout_rate = np.array(0.5)

image = cv2.imread(path)


cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resize_image = cv2.resize(image, (112, 112))
# print(resize_image.shape)

input_name_1 = "img_inputs"
input_name_2 = "dropout_rate"
output_name = "resnet_v1_50/E_BN2/Identity"
model_name = "faces"
protocol = ProtocolType.from_str('http')
ctx = InferContext('localhost:8000', protocol, model_name, 1, False)

request_ids = []
request_ids.append(ctx.async_run(
        { input_name_1 : [resize_image],
          input_name_2 : [dropout_rate]},
        { output_name : (InferContext.ResultFormat.RAW) }, 1)
        )

for i, request_id in enumerate(request_ids):
    vals = ctx.get_async_run_results(request_id, True) 
