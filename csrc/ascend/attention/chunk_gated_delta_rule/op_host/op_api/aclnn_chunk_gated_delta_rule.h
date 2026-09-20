#ifndef OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_H
#define OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ChunkGatedDeltaRule 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @param [in] query: 数据类型支持：bfloat16，shape (T, Nk, Dk)。
 * @param [in] key: 数据类型支持：bfloat16，shape (T, Nk, Dk)。
 * @param [in] value: 数据类型支持：bfloat16，shape (T, Nv, Dv)。
 * @param [in] beta: 数据类型支持：bfloat16，shape (T, Nv)。
 * @param [in] initial_state: 数据类型支持：bfloat16/float，shape (B, Nv, Dv, Dk)。
 * @param [in] actual_seq_lengths: 数据类型支持：int32，shape (B,)。
 * @param [in] g: 数据类型支持：float32，shape (T, Nv)，可选（nullptr 表示全 0）。
 * @param [in] scale_value: 缩放系数（float）。
 * @param [out] out: 数据类型支持：bfloat16，shape (T, Nv, Dv)。
 * @param [out] final_state: 数据类型支持：bfloat16/float，shape (B, Nv, Dv, Dk)。
 * @param [out] workspaceSize: 返回需要在 device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    const aclTensor *initial_state, const aclTensor *actual_seq_lengths, const aclTensor *g,
    float scale_value, aclTensor *out, aclTensor *final_state, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief ChunkGatedDeltaRule 的第二段接口，执行算子。
 * @param [in] workspace: device 侧 workspace 内存起址。
 * @param [in] workspaceSize: workspace 大小，由第一段接口获取。
 * @param [in] executor: op 执行器。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRule(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor,
                                                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_CHUNK_GATED_DELTA_RULE_H
