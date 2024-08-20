import torch
from torch import nn
from typing import Optional, Sequence, Tuple, List
from torch.nn import CrossEntropyLoss
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
import math

class KVCache():
    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]
    
    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor,torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(k)
            self.value_cache.append(v)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx],k],dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx],v],dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaConfig:

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        ropte_theta=1000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.ropte_theta = ropte_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = False

        self.vision_config = SiglipVisionConfig(**vision_config)

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2

        self.vision_config.projection_dim = projection_dim

class GemmaMLP(nn.Module):
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size,self.intermediate_size, bias =False)
        self.down_proj = nn.Linear(self.intermediate_size,self.hidden_size,bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x),approximate='tanh') * self.up_proj(x))

class GemmaAttention(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_key_value_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.ropte_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim,bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim,bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size,self.num_key_value_heads * self.head_dim,bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta
        )

        def forward(
                self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                kv_cache: Optional[KVCache] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            batch_size, q_len, _ = hidden_states.size()
            w_q = self.q_proj(hidden_states)
            w_k = self.k_proj(hidden_states)
            w_v = self.v_proj(hidden_states)

            w_q = w_q.view(batch_size,q_len,self.num_heads,self.head_dim).transpose(1,2)
            w_k = w_k.view(batch_size,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
            w_v = w_v.view(batch_size,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
            cos, sin = self.rotary_emb(w_v,position_ids,seq_len=None)
            
            w_q, w_k = apply_rotary_pos_emb(w_q,w_k,cos,sin)

            if kv_cache is not None:
                w_k,w_v = kv_cache.update(w_k,w_v,self.layer_idx)



class GemmaDecoderLayer(nn.Module):
    def __init__(self,config: GemmaConfig):
        super().__init__() 
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config,layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size,config.rms_norm_eps) 

    def forward(
        self, hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optiona[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        hidden_states = hidden_states + residual

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
    
        hidden_states = hidden_states + residual

        return hidden_states

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int,eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + self.eps)

    def forward(self,x):
        output = self._norm(x)

        output = output * (1 + self.weight.float())

        return output.type_as(x)



class GemmaModel(nn.Module):
    def __init__(self,config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(self, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, input_embeds: Optional[torch.FloatTensor] = None, kv_cache: Optional[KVCache] = None)-> torch.FloatTensor:
        hidden_states = input_embeds
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                input_embeds=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
        
        hidden_states = self.norm(hidden_states)

        return hidden_states


class GemmaForCausalLM(nn.Module):

    def __init__(self,config:GemmaConfig):
        super().__init__()
        self.config. =  config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size,
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    
    def forward(self, attention_mask:Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, input_embeds: Optional[torch.FloatTensor]=None, kv_cache: Optional[KVCache]= None) -> Dict[str,any]:
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )    
        hidden_states = outputs
        logits = self.lm_head(hidden_states)

        return_data = {
            "logits":logits,
        }   

        if kv_cache is not None:
            return_data['kv_cache'] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.projection_dim,
            bias=True,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(image_features)
        return hidden_states


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        image_features: torch.Tensor,
        input_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        kv_cache,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.tensor]:
        _, _, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size**-0.5)

        final_embedding = torch.zeros(
            batch_size, seq_len, embed_dim, dtype=dtype, device=device
        )

        text_mask = (input_ids != self.config.image_token_index) & (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index

        pad_mask = input_ids == self.pad_token_id

        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        final_embedding = torch.where(text_mask, input_embeds, final_embedding)
        final_embedding = final_embedding.masked_scatter(
            image_mask, scaled_image_features
        )
        final_embedding = torch.where(
            pad_mask, torch.zeros_like(final_embedding), final_embedding
        )

        min_dtype = torch.finfo(dtype).min

        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full(
                (barch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            assert q_len == 1
            kv_len = kv_cache.num_items()

            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

            causal_mask = causal_mask.unsqueeze(1)

            if kv_cache is not None and kv_cache.num_items() > 0:
                position_ids = attention_mask.cumsum(-1)[:, -1]
                if position_ids.dim() == 1:
                    position_ids = position_ids.unsqueeze(0)
            else:
                position_ids = (
                    (attention_mask.cumsum(-1)).masked_fill((attention_mask == 0), 1)
                ).to(device)
        return final_embedding, attention_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        image_features = self.vision_tower(pixel_values.to(input_embeds.dtype))

        image_features = self.multi_modal_projector(image_features)

        input_embeds, attention_mask, position_ids = (
            self._merge_input_ids_with_image_features(
                image_features, input_embeds, input_ids, attention_mask, kv_cache
            )
        )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
        )

        return outputs
