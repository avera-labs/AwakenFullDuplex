# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
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
#
# Vendored from FunAudioLLM/CosyVoice (cosyvoice/utils/file_utils.py): only the ONNX ->
# TensorRT conversion is kept; the module-level logging configuration was removed.
"""Build the TensorRT engine of the flow estimator from the upstream ONNX export."""

import logging

logger = logging.getLogger(__name__)


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logger.info("Converting onnx to trt...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    # load onnx model
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error("%s", parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # set input shapes
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i], trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    # set input and output data type
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    # save trt engine
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logger.info("Succesfully convert onnx to trt...")
