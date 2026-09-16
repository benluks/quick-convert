# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
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
from cosyvoice.llm.llm import TransformerLM, Qwen2LM, CosyVoice3LM
from cosyvoice.flow.flow import MaskedDiffWithXvec, CausalMaskedDiffWithXvec, CausalMaskedDiffWithDiT
from cosyvoice.hifigan.generator import HiFTGenerator, CausalHiFTGenerator
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model, CosyVoice3Model
from cosyvoice.utils.transformer_classes import (
    COSYVOICE_ACTIVATION_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
)


def get_model_type(configs):
    # NOTE CosyVoice2Model inherits CosyVoiceModel
    if isinstance(configs['llm'], TransformerLM) and isinstance(configs['flow'], MaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoiceModel
    if isinstance(configs['llm'], Qwen2LM) and isinstance(configs['flow'], CausalMaskedDiffWithXvec) and isinstance(configs['hift'], HiFTGenerator):
        return CosyVoice2Model
    if isinstance(configs['llm'], CosyVoice3LM) and isinstance(configs['flow'], CausalMaskedDiffWithDiT) and isinstance(configs['hift'], CausalHiFTGenerator):
        return CosyVoice3Model
    raise TypeError('No valid model type found!')
