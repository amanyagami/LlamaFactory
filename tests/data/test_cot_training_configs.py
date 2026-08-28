# Copyright 2025 the LlamaFactory team.
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

import re
from pathlib import Path

import yaml

from llamafactory.hparams.parser import _parse_train_args


ROOT = Path(__file__).resolve().parents[2]

COT_GEN_PATH = ROOT / "examples/train_lora/qwen3_lora_sft_cot_gen.yaml"
COT_COND_PATH = ROOT / "examples/train_lora/qwen3_lora_sft_cot_cond.yaml"
REFERENCE_PATH = ROOT / "examples/train_lora/qwen3_lora_sft.yaml"

COT_GEN = yaml.safe_load(COT_GEN_PATH.read_text())
COT_COND = yaml.safe_load(COT_COND_PATH.read_text())
REFERENCE = yaml.safe_load(REFERENCE_PATH.read_text())


def _sections(path: Path) -> list[str]:
    return re.findall(r"^### .*$", path.read_text(), flags=re.MULTILINE)


def test_configs_share_section_layout_with_reference():
    assert _sections(COT_GEN_PATH) == _sections(REFERENCE_PATH)
    assert _sections(COT_COND_PATH) == _sections(REFERENCE_PATH)


def test_configs_are_valid_yaml_mappings():
    for config in (COT_GEN, COT_COND):
        assert isinstance(config, dict)
        assert config


def test_configs_pass_argument_schema_dry_check(tmp_path):
    for config in (COT_GEN, COT_COND):
        args = dict(config)
        # bf16 and the saves/ output dir are portability artifacts of this test, not config defects:
        # HF `TrainingArguments` rejects bf16 without a CUDA device, and nothing should be written to the repo.
        args.update(output_dir=str(tmp_path), bf16=False, use_cpu=True)
        _parse_train_args(args)


def test_cot_gen_is_response_side_loss_bearing():
    assert COT_GEN["stage"] == "sft"
    assert COT_GEN["finetuning_type"] == "lora"
    assert COT_GEN["template"] == "qwen3"
    assert COT_GEN["enable_thinking"] is True
    assert "train_on_prompt" not in COT_GEN


def test_cot_cond_is_prompt_side_loss_masked():
    assert COT_COND["enable_thinking"] is False
    assert COT_COND["train_on_prompt"] is False


def test_configs_differ_only_in_paradigm_keys():
    differing = sorted(k for k in set(COT_GEN) | set(COT_COND) if COT_GEN.get(k) != COT_COND.get(k))
    assert differing == ["dataset", "enable_thinking", "output_dir", "train_on_prompt"]


def test_paper_hyperparameters():
    for config in (COT_GEN, COT_COND):
        assert config["model_name_or_path"] == "Qwen/Qwen3-8B"
        assert config["learning_rate"] == 5.0e-5
        assert config["bf16"] is True
        assert config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * 2 == 64
        assert "max_samples" not in config


def test_unstated_values_match_paper_profile_or_repo_default():
    for config in (COT_GEN, COT_COND):
        assert config["cutoff_len"] == 25000
        for key in ("lora_rank", "lora_target", "num_train_epochs", "lr_scheduler_type", "warmup_ratio"):
            assert config[key] == REFERENCE[key]
