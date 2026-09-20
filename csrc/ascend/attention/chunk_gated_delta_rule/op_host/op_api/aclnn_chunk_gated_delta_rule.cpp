/*!
 * \file aclnn_chunk_gated_delta_rule.cpp
 * \brief
 */
#include <dlfcn.h>
#include "aclnn_chunk_gated_delta_rule.h"
#include "../chunk_gated_delta_rule.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t QUERY_DIM_NUM = 3;
constexpr size_t KEY_DIM_NUM = 3;
constexpr size_t VALUE_DIM_NUM = 3;
constexpr size_t BETA_DIM_NUM = 2;
constexpr size_t INITIAL_STATE_DIM_NUM = 4;
constexpr size_t SEQ_LENS_DIM_NUM = 1;

struct ChunkGatedDeltaRuleParams {
    // mandatory
    const aclTensor *query {nullptr};
    const aclTensor *key {nullptr};
    const aclTensor *value {nullptr};
    const aclTensor *beta {nullptr};
    const aclTensor *initial_state {nullptr};
    const aclTensor *actual_seq_lengths {nullptr};
    // optional
    const aclTensor *g {nullptr};
    // attrs
    float scale_value {1.0f};
    // output
    const aclTensor *out {nullptr};
    const aclTensor *final_state {nullptr};
};

// support dtype
static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BETA_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> FINAL_STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const ChunkGatedDeltaRuleParams &params)
{
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.initial_state, return false);
    OP_CHECK_NULL(params.actual_seq_lengths, return false);
    OP_CHECK_NULL(params.out, return false);
    OP_CHECK_NULL(params.final_state, return false);

    return true;
}

static inline bool CheckDtypeVaild(const ChunkGatedDeltaRuleParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.initial_state, STATE_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths, SEQ_LENS_TYPE_SUPPORT_LIST, return false);

    if (params.g != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.final_state, FINAL_STATE_TYPE_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleParams &params)
{
    CHECK_RET(CheckDtypeVaild(params), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("ChunkGatedDeltaRule check params success.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(ChunkGatedDeltaRuleParams &params)
{
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.beta->SetOriginalShape(params.beta->GetViewShape());
    params.initial_state->SetOriginalShape(params.initial_state->GetViewShape());
    params.actual_seq_lengths->SetOriginalShape(params.actual_seq_lengths->GetViewShape());

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
                                                     const aclTensor *value, const aclTensor *beta,
                                                     const aclTensor *initial_state,
                                                     const aclTensor *actual_seq_lengths, const aclTensor *g,
                                                     float scale_value, aclTensor *out, aclTensor *final_state,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRule,
                   DFX_IN(query, key, value, beta, initial_state, actual_seq_lengths, g, scale_value),
                   DFX_OUT(out, final_state));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ChunkGatedDeltaRuleParams params {query, key, value, beta, initial_state, actual_seq_lengths, g, scale_value,
                                      out, final_state};

    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());
    auto initialState_ = l0op::Contiguous(initial_state, uniqueExecutor.get());
    auto actualSeqLengths_ = l0op::Contiguous(actual_seq_lengths, uniqueExecutor.get());
    if (g != nullptr) {
        g = l0op::Contiguous(g, uniqueExecutor.get());
    }

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());

    // 调用 l0 接口
    auto outRet = l0op::ChunkGatedDeltaRule(query_, key_, value_, beta_, initialState_, actualSeqLengths_, g,
                                            scale_value, final_state, uniqueExecutor.get());
    if (outRet == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto ViewCopyResult = l0op::ViewCopy(outRet, out_, uniqueExecutor.get());
    if (ViewCopyResult == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRule);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
