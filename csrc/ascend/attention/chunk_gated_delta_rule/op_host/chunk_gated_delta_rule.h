#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *ChunkGatedDeltaRule(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                     const aclTensor *beta, const aclTensor *initial_state,
                                     const aclTensor *actual_seq_lengths, const aclTensor *g,
                                     float scale_value, aclTensor *final_state, aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE
