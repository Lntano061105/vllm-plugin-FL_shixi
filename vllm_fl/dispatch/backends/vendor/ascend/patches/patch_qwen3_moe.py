import torch
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# 加载你的算子
torch.ops.load_library('/workspace/vllm-plugin-FL/vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so')

def apply_patch():
    """应用补丁，替换 Qwen3MoeSparseMoeBlock.forward"""
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
        original_forward = Qwen3MoeSparseMoeBlock.forward

        @wraps(original_forward)
        def patched_forward(self, hidden_states, **kwargs):
            # 调用原始 gate 得到 router_logits
            router_logits, _ = self.gate(hidden_states)
            # 调用你的算子
            if router_logits.dim() == 2:
                num_tokens, num_experts = router_logits.shape
                try:
                    y, expert_idx, out = torch.ops._C_ascend.moe_gating_top_k(
                        router_logits,
                        k=2,  # 从配置读取？暂时固定
                        k_group=1,
                        group_count=1,
                        group_select_mode=0,
                        renorm=0,
                        norm_type=0,
                        out_flag=True,
                        routed_scaling_factor=1.0,
                        eps=1e-20
                    )
                    logger.info(f"[MoePatch] 调用 moe_gating_top_k 成功，y shape: {y.shape}, expert_idx shape: {expert_idx.shape}")
                    # 继续调用原始 forward（但避免重复调用 gate）
                    # 这里直接调用 self.experts 并传入 router_logits
                    final_hidden_states = self.experts(hidden_states, router_logits)
                    return final_hidden_states
                except Exception as e:
                    logger.warning(f"[MoePatch] 调用 moe_gating_top_k 失败: {e}, 回退到原始 forward")
                    return original_forward(self, hidden_states, **kwargs)
            else:
                return original_forward(self, hidden_states, **kwargs)

        Qwen3MoeSparseMoeBlock.forward = patched_forward
        logger.info("[MoePatch] 补丁已应用，Qwen3MoeSparseMoeBlock.forward 被替换")
        return True
    except Exception as e:
        logger.error(f"[MoePatch] 应用补丁失败: {e}")
        return False
