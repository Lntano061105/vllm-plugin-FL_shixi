/*!
 * \file chunk_gated_delta_rule.cpp
 * \brief
 */
#include "../chunk_gated_delta_rule.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRule);

const aclTensor *ChunkGatedDeltaRule(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                     const aclTensor *beta, const aclTensor *initial_state,
                                     const aclTensor *actual_seq_lengths, const aclTensor *g,
                                     float scale_value, aclTensor *final_state, aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRule, query, key, value, beta, initial_state, actual_seq_lengths, g, scale_value);

    DataType outType = DataType::DT_BF16;
    Format format = Format::FORMAT_ND;

    auto out = executor->AllocTensor(outType, format, format);

    OP_CHECK(out != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."),
             return nullptr);

    // infershape
    auto ret = INFER_SHAPE(
        ChunkGatedDeltaRule,
        OP_INPUT(query, key, value, beta, initial_state, actual_seq_lengths, g),
        OP_OUTPUT(out, final_state), OP_ATTR(scale_value));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "ChunkGatedDeltaRule InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRule,
        OP_INPUT(query, key, value, beta, initial_state, actual_seq_lengths, g),
        OP_OUTPUT(out, final_state), OP_ATTR(scale_value));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "ChunkGatedDeltaRule ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return out;
}
} // namespace l0op
