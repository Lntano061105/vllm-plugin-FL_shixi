#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/Main/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/workspace/Main/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
