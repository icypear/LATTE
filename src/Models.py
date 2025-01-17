from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .helpers_fm import PerceiverResampler
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from .utils_fm import apply_with_stopping_condition
from transformers import GPT2Tokenizer
from torch.nn import init
from fvcore.nn import FlopCountAnalysis, parameter_count

class AccidentPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=2, act=torch.relu, dropout=[0, 0]):
        super(AccidentPredictor, self).__init__()
        self.act = act
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(input_dim, 64)
        self.dense2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout[0], training=self.training)
        x = self.act(self.dense1(x))
        x = F.dropout(x, self.dropout[1], training=self.training)
        x = self.dense2(x)
        return x

class Memory_Attention_Aggregation(nn.Module):
    def __init__(self, agg_dim, d_model=512, S=64):
        super(Memory_Attention_Aggregation, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(d_model, d_model))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=np.sqrt(5))

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.attn_softmax = nn.Softmax(dim=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, hiddens):

        attn = self.mk(hiddens)
        attn = self.attn_softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        hiddens = self.mv(attn)
        m = torch.tanh(hiddens)
        alpha = torch.softmax(torch.matmul(m, self.weight), 0)
        roh = torch.mul(hiddens, alpha)
        new_h = torch.sum(roh, 0)
        return new_h


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Auxiliary_Self_Attention_Aggregation(nn.Module):
    def __init__(self, agg_dim):
        super(Auxiliary_Self_Attention_Aggregation, self).__init__()
        self.agg_dim = agg_dim
        self.weight = nn.Parameter(torch.Tensor(agg_dim, 1))
        self.softmax = nn.Softmax(dim=-1)
        torch.nn.init.kaiming_normal_(self.weight, a=np.sqrt(5))
        self.dsc = DepthwiseSeparableConv(in_channels=1, out_channels=1)

    def forward(self, hiddens):
        hiddens = hiddens.unsqueeze(0)
        hiddens = hiddens.permute(1, 0, 2, 3)
        hiddens = self.dsc(hiddens)
        maxpool = torch.max(hiddens, dim=1)[0]
        avgpool = torch.mean(hiddens, dim=1)
        agg_spatial = torch.cat((avgpool, maxpool), dim=1)
        energy = torch.bmm(agg_spatial.permute([0, 2, 1]), agg_spatial)
        attention = self.softmax(energy)
        weighted_feat = torch.bmm(attention, agg_spatial.permute([0, 2, 1]))
        weight = self.weight.unsqueeze(0).repeat([hiddens.size(0), 1, 1])
        agg_feature = torch.bmm(weighted_feat.permute([0, 2, 1]), weight)
        return agg_feature.squeeze(dim=-1)

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0, 0]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = nn.Linear(hidden_dim, 64)
        self.dense2 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = F.dropout(out[:, -1], self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        return out, h

class EMSA(nn.Module):
    def __init__(self, channels, factor=5):
        super(EMSA, self).__init__()
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        concat_x = torch.cat([x_h, x_w], dim=2)
        hw = self.conv1x1(concat_x)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x_h_sigmoid = x_h.sigmoid()
        x_w_sigmoid = x_w.permute(0, 1, 3, 2).sigmoid()
        x1 = self.gn(group_x * x_h_sigmoid * x_w_sigmoid)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class LATTE(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1, n_obj=19, n_frames=100, fps=20.0, with_saa=True):
        super(LATTE, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.n_obj = n_obj
        self.n_frames = n_frames
        self.fps = fps
        self.with_saa = with_saa
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        #spatial_attention
        self.maa = Memory_Attention_Aggregation(h_dim,z_dim, n_layers, n_obj)
        self.gru_net = GRUNet(h_dim+h_dim , h_dim, 2, n_layers,dropout=[0.5, 0.0])
        self.frame_aggregation = Auxiliary_Self_Attention_Aggregation(5)
        if self.with_saa:
            # auxiliary branch
            self.predictor_aux = AccidentPredictor(h_dim + h_dim, 2, dropout=[0.5, 0.0])
            self.aaa = Auxiliary_Self_Attention_Aggregation(self.n_frames)
        # loss function
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.emssa = EMSA(channels=100)


    def forward(self, x, y, toa, hidden_in=None, nbatch=80, testing=False):
        """
        :param x, (batchsize, nFrames, nBoxes, Xdim) = (10 x 100 x 20 x 4096)
        :param y, (10 x 2)
        :param toa, (10,)
        """
        x = self.ema(x) # Model A
        losses = {'cross_entropy': 0,
                  'total_loss': 0}
        if self.with_saa:
            losses.update({'auxloss': 0})
        all_outputs, all_hidden = [], []
        all_alphas = []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(0),  self.h_dim))
        else:
            h = Variable(hidden_in)
        h = h.to(x.device)
    
        alpha = torch.zeros(self.n_frames)
        zeros_object_1 = torch.sum(x[:,:,1:self.n_obj+1,:].permute(1,2,0,3),3)
        zeros_object_2 = ~zeros_object_1.eq(0)
        zeros_object = zeros_object_2.float()

        if x.size(-1) == 4096 and self.phi_x[0].in_features != 4096:
            self.phi_x[0] = torch.nn.Linear(4096, self.h_dim).to(x.device)
        elif x.size(-1) == 512 and self.phi_x[0].in_features != 512:
            self.phi_x[0] = torch.nn.Linear(512, self.h_dim).to(x.device)

        h_list = []

        for t in range(x.size(1)):

            x_t = self.phi_x(x[:, t])
            img_embed = x_t[:, 0, :].unsqueeze(1)
            obj_embed = x_t[:, 1:, :]
            obj_embed, alphas= self.maa(obj_embed, h, t, zeros_object[t])
            x_t = torch.cat([obj_embed, img_embed], dim=-1)
            h_list.append(h)
            all_alphas.append(alphas)

            if t==2:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2]),dim=0)
                h = self.frame_aggregation(h_staked)
            elif t==3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2], h_list[t-3]),dim=0)
                h = self.frame_aggregation(h_staked)
            elif t > 3:
                h_staked = torch.stack((h_list[t],h_list[t-1], h_list[t-2],h_list[t-3],h_list[t-4]),dim=0)
                h = self.frame_aggregation(h_staked)

            output, h = self.gru_net(x_t, h)

            # computing losses
            L3 = self._exp_loss(output, y, t, toa=toa, fps=self.fps)
            losses['cross_entropy'] += L3
            all_outputs.append(output)
            all_hidden.append(h[-1])

        if self.with_saa:
            # soft attention to aggregate hidden states of all frames
            embed_video = self.aaa(torch.stack(all_hidden, dim=-1))
            dec = self.predictor_aux(embed_video)
            L4 = torch.mean(self.ce_loss(dec, y[:, 1].to(torch.long)))
            losses['auxloss'] = L4

        return losses, all_outputs, all_hidden, all_alphas


    def _exp_loss(self, pred, target, time, toa, fps=10.0):
        '''
        :param pred:
        :param target: onehot codings for binary classification
        :param time:
        :param toa:
        :param fps:
        :return:
        '''
        # positive example (exp_loss)
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(toa).to(toa.device, pred.dtype), (toa.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        # negative example
        neg_loss = self.ce_loss(pred, target_cls)

        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss

class Flamingo(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        vis_dim: int,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim

         
        self.vision_encoder = vision_encoder
        self.lang_encoder = lang_encoder

         
        for param in self.lang_encoder.parameters():
            param.requires_grad = True

        
        for param in self.vision_encoder.parameters():
            param.requires_grad = True


         
        self.lang_encoder.config.padding_side = "left"

        
        self.lang_encoder.config.pad_token_id = self.eoc_token_id

        
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing

    def _encode_vision_x(self, vision_x: torch.Tensor):
       
        with torch.no_grad():
            vision_x = self.vision_encoder.get_image_features(vision_x)
        return vision_x

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # if clear_conditioned_layers:
        #     self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        input_text: str,  
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
         
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        lang_x = tokenizer.encode(input_text, return_tensors="pt")  

        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        # Attention Mask
        if attention_mask is None:
            attention_mask = torch.ones_like(lang_x, dtype=torch.long)  


        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x) # (b, T_img, F, C, H, W)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x, 
            attention_mask=attention_mask, 
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            **kwargs,
        )

        # self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"
        # print(f"vision_x shape before rearrange: {vision_x.shape}")

        # Rearrange to pass through the vision encoder
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        # print(f"vision_x shape after rearrange: {vision_x.shape}")

        # Use CLIP to extract vision features
        with torch.no_grad():
            vision_x = self.vision_encoder.get_image_features(vision_x)  # 将 pixel_values 作为输入
        return vision_x
        


    def wrap_fsdp(self, wrapper_kwargs, device_id):
        """
        Manually wraps submodules for FSDP and move other parameters to device_id.

        Why manually wrap?
        - all parameters within the FSDP wrapper must have the same requires_grad.
            We have a mix of frozen and unfrozen parameters.
        - model.vision_encoder.visual needs to be individually wrapped or encode_vision_x errors
            See: https://github.com/pytorch/pytorch/issues/82461#issuecomment-1269136344

        The rough wrapping structure is:
        - FlamingoModel
            - FSDP(FSDP(vision_encoder))
            - FSDP(FSDP(perceiver))
            - lang_encoder
                - FSDP(FSDP(input_embeddings))
                - FlamingoLayers
                    - FSDP(FSDP(gated_cross_attn_layer))
                    - FSDP(FSDP(decoder_layer))
                - FSDP(FSDP(output_embeddings))
                - other parameters

        Known issues:
        - Our FSDP strategy is not compatible with tied embeddings. If the LM embeddings are tied,
            train with DDP or set the --freeze_lm_embeddings flag to true.
        - With FSDP + gradient ckpting, one can increase the batch size with seemingly no upper bound.
            Although the training curves look okay, we found that downstream performance dramatically
            degrades if the batch size is unreasonably large (e.g., 100 MMC4 batch size for OPT-125M).

        FAQs about our FSDP wrapping strategy:
        Why double wrap?
        As of torch==2.0.1, FSDP's _post_forward_hook and _post_backward_hook
        only free gathered parameters if the module is NOT FSDP root.

        Why unfreeze the decoder_layers?
        See https://github.com/pytorch/pytorch/issues/95805
        As of torch==2.0.1, FSDP's _post_backward_hook is only registed if the flat param
        requires_grad=True. We need the postback to fire to avoid OOM.
        To effectively freeze the decoder layers, we exclude them from the optimizer.

        What is assumed to be frozen v. unfrozen?
        We assume that the model is being trained under normal Flamingo settings
        with these lines being called in factory.py:
            ```
            # Freeze all parameters
            model.requires_grad_(False)
            assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

            # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
            model.perceiver.requires_grad_(True)
            model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
            [optional] model.lang_encoder.get_input_embeddings().requires_grad_(True)
            ```
        """
        # unfreeze the decoder layers
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):
            self.perceiver = wrap(wrap(self.perceiver))
            self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            )
            self.lang_encoder.gated_cross_attn_layers = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_encoder.gated_cross_attn_layers
            )
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            self.lang_encoder.set_input_embeddings(
                wrap(wrap(self.lang_encoder.get_input_embeddings()))
            )
            self.lang_encoder.set_output_embeddings(
                wrap(wrap(self.lang_encoder.get_output_embeddings()))
            )
            self.vision_encoder = wrap(wrap(self.vision_encoder))  # frozen

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):
            self.perceiver.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        # self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False