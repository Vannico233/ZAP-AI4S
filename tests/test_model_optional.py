"""Architecture compatibility smoke test on random tiny weights, NOT Qwen scores."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def test_qwen35_lora_forward_backward_and_reload(tmp_path):
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.set_num_threads(2)
    config = Qwen3_5Config(text_config={
        "vocab_size": 256, "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_num_key_heads": 4, "linear_num_value_heads": 4,
        "layer_types": ["linear_attention", "full_attention"], "max_position_embeddings": 128,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                            "partial_rotary_factor": .5, "mrope_section": [1, 1, 2]},
        "pad_token_id": 0, "eos_token_id": 2},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 2, "out_hidden_size": 64, "num_position_embeddings": 16},
        image_token_id=253, video_token_id=254, vision_start_token_id=252)
    base = Qwen3_5ForConditionalGeneration(config)
    targets = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear) and ".language_model." in n]
    assert targets and not any("visual" in n for n in targets)
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=targets, task_type="CAUSAL_LM"))
    tokens = torch.randint(10, 100, (1, 32))
    labels = tokens.clone()
    labels[:, :20] = -100
    model.train()
    loss = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=labels, use_cache=False).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
    model.save_pretrained(tmp_path/"adapter")
    restored = PeftModel.from_pretrained(Qwen3_5ForConditionalGeneration(config), tmp_path/"adapter")
    restored.eval()
    with torch.inference_mode():
        result = restored.generate(input_ids=tokens[:, :8], max_new_tokens=2, do_sample=False,
                                   pad_token_id=0, eos_token_id=2)
    assert 8 < result.shape[1] <= 10
