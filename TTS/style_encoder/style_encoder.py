import torch
from coqpit import Coqpit
from torch import nn
from TTS.style_encoder.layers.ref import ReferenceEncoder, ModifiedReferenceEncoder, BidirectionalReferenceEncoder, INReferenceEncoder
from TTS.style_encoder.layers.finegrainedre import FineGrainedReferenceEncoder
from TTS.style_encoder.layers.gst import GST
from TTS.style_encoder.layers.vae import VAEStyleEncoder
from TTS.style_encoder.layers.vqvae import VQVAEStyleEncoder
from TTS.style_encoder.layers.vaeflow import VAEFlowStyleEncoder
from TTS.style_encoder.layers.diffusion import DiffStyleEncoder
from TTS.style_encoder.layers.metastyle import MetaStyleEncoder


class StyleEncoder(nn.Module):
    def __init__(self, config:Coqpit) -> None:
        super().__init__()
        
        # Load Attributes from Style Encoder Config
        for key in config:
            setattr(self, key, config[key])
 
        if(self.use_nonlinear_proj):
            self.nl_proj = nn.Linear(self.style_embedding_dim, self.proj_dim)
            nn.init.xavier_normal_(self.nl_proj.weight) # Good init for projection

        self.dropout = nn.Dropout(p=0.5)

        if(self.use_proj_linear):
            self.proj = nn.Linear(self.style_embedding_dim, self.proj_dim)
            nn.init.xavier_normal_(self.proj.weight) # Good init for projection

        # REFERENCE PITCH ENCODER ARCHITECTURE
        if "pitch" in self.style_reference_features:
            print("> Using Pitch as Reference Feature")
            self.ref_pitch_enc = nn.Conv1d(
                1,
                self.style_embedding_dim,
                kernel_size=3,
                padding=int((3 - 1) / 2),
            )

        # REFERENCE ENERGY ENCODER ARCHITECTURE
        if "energy" in self.style_reference_features:
            print("> Using Energy as Reference Feature")
            self.ref_energy_enc = nn.Conv1d(
                1,
                self.style_embedding_dim,
                kernel_size=3,
                padding=int((3 - 1) / 2),
            ) 

        # REFERENCE MELSPECTROGRAM ENCODER ARCHITECTURE
        if "melspectrogram" in self.style_reference_features:
            print("> Using MelSpectrogram as Reference Feature")

            if self.se_type == 're':
                self.layer = ReferenceEncoder(
                    num_mel = self.num_mel,
                    embedding_dim = self.style_embedding_dim
                )
            elif self.se_type == 'inre':
                self.layer = INReferenceEncoder(
                    num_mel = self.num_mel,
                    embedding_dim = self.style_embedding_dim
                )
            elif self.se_type == 'modifiedre':
                self.layer = ModifiedReferenceEncoder(
                    num_mel = self.num_mel,
                    embedding_dim = self.style_embedding_dim
                )
            elif self.se_type == 'bidirectionalre':
                self.layer = BidirectionalReferenceEncoder(
                    num_mel = self.num_mel,
                    embedding_dim = self.style_embedding_dim
                )
            elif self.se_type == 'finegrainedre':
                self.layer = FineGrainedReferenceEncoder(
                    num_mel = self.num_mel,
                    embedding_dim = self.style_embedding_dim,
                    prosody_embedding_dim = self.prosody_embedding_dim,
                    encoder_embedding_dim = self.proj_dim,
                    fg_attention_dropout = self.fg_attention_dropout,
                    fg_attention_dim = self.fg_attention_dim
                )
            elif self.se_type == 'gst':
                self.layer = GST(
                    num_mel = self.num_mel,
                    gst_embedding_dim = self.style_embedding_dim,
                    num_heads = self.gst_num_heads,
                    num_style_tokens = self.gst_num_style_tokens,
                )
            elif self.se_type == 'vae':
                self.layer = VAEStyleEncoder(
                    num_mel = self.num_mel,
                    ref_embedding_dim = self.style_embedding_dim, # allow flexibility
                    style_embedding_dim = self.style_embedding_dim
                )
            elif self.se_type == 'vqvae':
                self.layer = VQVAEStyleEncoder(
                    nmels_ref= self.num_mel,
                    dim = self.style_embedding_dim,
                    K = self.codebook_size
                )
            elif self.se_type == 'vaeflow':
                self.layer = VAEFlowStyleEncoder(
                    num_mel = self.num_mel,
                    ref_emb_dim = self.style_embedding_dim,
                    style_emb_dim = self.style_embedding_dim,
                    intern_dim = self.vaeflow_intern_dim,
                    number_of_flows = self.vaeflow_number_of_flows
                )
            elif self.se_type == 'diffusion':
                self.layer = DiffStyleEncoder(
                    diff_num_timesteps = self.diff_num_timesteps, 
                    diff_schedule_type = self.diff_schedule_type,
                    diff_loss_type = self.diff_loss_type, 
                    ref_online = self.diff_ref_online, 
                    ref_num_mel = self.num_mel, 
                    ref_style_emb_dim = self.style_embedding_dim, 
                    den_step_dim = self.diff_step_dim,
                    den_in_out_ch = self.diff_in_out_ch, 
                    den_num_heads = self.diff_num_heads, 
                    den_hidden_channels = self.diff_hidden_channels, 
                    den_num_blocks = self.diff_num_blocks,
                    den_dropout = self.diff_dropout
                )
            elif self.se_type == 'metastyle':
                self.layer = MetaStyleEncoder(
                    num_mel = self.num_mel, 
                    embedding_dim = self.style_embedding_dim, 
                    n_spectral = 2, 
                    n_temporal = 2, n_mha = 1, 
                    mha_heads = 1, 
                    max_seq_len = 400
                )
            else:
                raise NotImplementedError

    def forward(self, **kwargs):
        inputs = kwargs['out_txt_encoder']
        out = {'style_embedding':torch.zeros(inputs.size()[0], inputs.size()[-1]).to('cuda:0'), 'styled_inputs':torch.zeros(inputs.size()).to('cuda:0')}

        if "melspectrogram" in self.style_reference_features:
            ref_melspectrogram =  kwargs['reference_features']['melspectrogram']
            if self.se_type == 'gst':
                out.update(self.gst_embedding(inputs=inputs, style_input=ref_melspectrogram))
            elif(self.se_type == 're' or self.se_type == 'inre'):
                out.update(self.re_embedding(inputs=inputs, style_input=ref_melspectrogram))
            elif self.se_type == 'modifiedre':
                out.update(self.modified_re_embedding_forward(inputs=inputs, style_input=ref_melspectrogram, speaker_embedding=kwargs['speaker_embedding']))
            elif self.se_type == 'bidirectionalre':
                out.update(self.re_embedding(inputs=inputs, style_input=ref_melspectrogram)) # Since is only the re arquitecture that is different we use the same method
            elif self.se_type == 'finegrainedre':
                out.update(self.finegrainedre_embedding(inputs=inputs, style_input=ref_melspectrogram, text_len=kwargs['text_len'], mel_len=kwargs['mel_len']))
            elif self.se_type == 'diffusion':
                out.update(self.diff_forward(inputs=inputs, style_input=ref_melspectrogram))
            elif self.se_type == 'vae':
                out.update(self.vae_forward(inputs=inputs, style_input=ref_melspectrogram))
            elif self.se_type == 'vqvae':
                out.update(self.vqvae_forward(inputs=inputs, style_input=ref_melspectrogram))
            elif self.se_type == 'vaeflow':
                out.update(self.vaeflow_forward(inputs=inputs, style_input=ref_melspectrogram))
            elif self.se_type == 'metastyle':
                out.update(self.metastyle_forward(inputs=inputs, style_input=ref_melspectrogram, mel_mask=kwargs['mel_mask']))
            else:
                raise NotImplementedError

        if "pitch" in self.style_reference_features:
            ref_pitch = kwargs['reference_features']['pitch']
            ref_pitch_out = self.ref_pitch_enc(ref_pitch)
            ref_pitch_emb = torch.sum(ref_pitch_out, dim=2) / ref_pitch.size()[-1]
            out['style_embedding'] += ref_pitch_emb
            out['styled_inputs'] = self._add_speaker_embedding(outputs = out['styled_inputs'], embedded_speakers = ref_pitch_emb.unsqueeze(1))

        if "energy" in self.style_reference_features:
            ref_energy = kwargs['reference_features']['energy']
            ref_energy_emb = self.ref_energy_enc(ref_energy)
            out['style_embedding'] += ref_energy_emb
            out['styled_inputs'] = self._add_speaker_embedding(outputs = out['styled_inputs'], embedded_speakers = ref_energy_emb.unsqueeze(1))

        return out

    def inference(self, **kwargs):
        inputs = kwargs['out_txt_encoder']

        if "melspectrogram" in self.style_reference_features:
            ref_melspectrogram =  kwargs['reference_features']['melspectrogram']
            
            if self.se_type == 'gst':
                out = self.gst_embedding(inputs=inputs, style_input=ref_melspectrogram, speaker_embedding=kwargs['d_vectors'])
            elif(self.se_type == 're' or self.se_type == 'inre'):
                out = self.re_embedding_inference(inputs=inputs, style_input=ref_melspectrogram)
            elif self.se_type == 'modifiedre':
                out = self.modified_re_embedding_inference(inputs=inputs, style_input=ref_melspectrogram, speaker_embedding=kwargs['speaker_embedding'])
            elif self.se_type == 'bidirectionalre':
                out = self.re_embedding_inference(inputs=inputs, style_input=ref_melspectrogram)# Since is only the re arquitecture that is different we use the same method
            elif self.se_type == 'finegrainedre':
                out = self.finegrainedre_embedding_inference(inputs=inputs, style_input=ref_melspectrogram, text_len=kwargs['text_len'], mel_len=kwargs['mel_len'])
            elif self.se_type == 'diffusion':
                out = self.diff_inference(inputs=inputs, style_input=ref_melspectrogram, infer_from = kwargs['diff_t'])
            elif self.se_type == 'vae':
                out = self.vae_inference(inputs=inputs, style_input=ref_melspectrogram, z = kwargs['z'])
            elif self.se_type == 'vqvae':
                out = self.vqvae_inference(inputs=inputs, style_input=ref_melspectrogram, K = kwargs['K'])
            elif self.se_type == 'vaeflow':
                out = self.vaeflow_inference(inputs=inputs, style_input=ref_melspectrogram, z = kwargs['z'])
            elif self.se_type == 'metastyle':
                out = self.metastyle_forward(inputs=inputs, style_input=ref_melspectrogram, mel_mask=kwargs['mel_mask'])
            else:
                raise NotImplementedError
            

        if "pitch" in self.style_reference_features:
            ref_pitch = kwargs['reference_features']['pitch']
            ref_pitch_emb = self.ref_pitch_enc(ref_pitch)
            out['style_embedding'] += ref_pitch_emb
            out['styled_inputs'] = self._add_speaker_embedding(outputs = out['style_inputs'], embedded_speakers = ref_pitch_emb.unsqueeze(1))

        if "energy" in self.style_reference_features:
            ref_energy = kwargs['reference_features']['energy']
            ref_energy_emb = self.ref_energy_enc(ref_energy)
            out['style_embedding'] += ref_energy_emb
            out['styled_inputs'] = self._add_speaker_embedding(outputs = out['style_inputs'], embedded_speakers = ref_energy_emb.unsqueeze(1))
            
        return out
            
    #######################
    ###### Specifics ######
    #######################
    
    # Styled Inputs = [B, L, STY_DIM]
    # Style Embedding = [B, STY_DIM]

    def metastyle_forward(self, inputs, style_input, mel_mask):
        if style_input is None:
            # ignore style token and return zero tensor
            gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
        elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
            gst_outputs = style_input
        else:
            # compute style tokens
            input_args = [style_input, mel_mask]
            gst_outputs = self.layer(mel = style_input, mask = mel_mask) 
        
            if(self.use_nonlinear_proj):
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
            
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        elif(self.agg_type == 'adain'):
            inputs = self._adain(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        else:
            inputs = self._add_speaker_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            
        return {'styled_inputs': inputs, 'style_embedding': gst_outputs}

    def modified_re_embedding_forward(self, inputs, style_input, speaker_embedding):
        if style_input is None:
            # ignore style token and return zero tensor
            gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
        elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
            gst_outputs = style_input
        else:
            # compute style tokens
            input_args = [style_input]
            # print(inputs.shape, style_input.shape, speaker_embedding.shape)
            gst_outputs = self.layer(style_input, speaker_embedding) 
        
            if(self.use_nonlinear_proj):
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
            
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        elif(self.agg_type == 'adain'):
            inputs = self._adain(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        else:
            inputs = self._add_speaker_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            
        return {'styled_inputs': inputs, 'style_embedding': gst_outputs}

    def modified_re_embedding_inference(self, inputs, style_input, speaker_embedding):
        if style_input is None:
            # ignore style token and return zero tensor
            gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
        elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
            gst_outputs = style_input
        else:
            # compute style tokens
            input_args = [style_input.unsqueeze(0)]
            gst_outputs = self.layer(style_input, speaker_embedding) 
        
            if(self.use_nonlinear_proj):
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
            
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        elif(self.agg_type == 'adain'):
            inputs = self._adain(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        else:
            inputs = self._add_speaker_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            
        return {'styled_inputs': inputs, 'style_embedding': gst_outputs}



    def re_embedding(self, inputs, style_input):
        if style_input is None:
            # ignore style token and return zero tensor
            gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
        elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
            gst_outputs = style_input
        else:
            # compute style tokens
            input_args = [style_input]
            gst_outputs = self.layer(*input_args) 
        
            if(self.use_nonlinear_proj):
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
            
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        elif(self.agg_type == 'adain'):
            inputs = self._adain(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        else:
            inputs = self._add_speaker_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            
        return {'styled_inputs': inputs, 'style_embedding': gst_outputs}

    def re_embedding_inference(self, inputs, style_input):
        if style_input is None:
            # ignore style token and return zero tensor
            gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
        elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
            gst_outputs = style_input
        else:
            # compute style tokens
            input_args = [style_input.unsqueeze(0)]
            gst_outputs = self.layer(*input_args) 
        
            if(self.use_nonlinear_proj):
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
            
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        elif(self.agg_type == 'adain'):
            inputs = self._adain(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
        else:
            inputs = self._add_speaker_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            
        return {'styled_inputs': inputs, 'style_embedding': gst_outputs}

    def finegrainedre_embedding(self, inputs, style_input, text_len, mel_len):
            if style_input is None:
                # ignore style token and return zero tensor
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
            elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
                gst_outputs = style_input
            else:
                # compute style tokens
                # input_args = [style_input]
                # print(text_len.shape, mel_len.shape, style_input.shape)
                gst_outputs, alignments = self.layer(inputs, text_len, style_input, mel_len)  # pylint: disable=not-callable
            
                if(self.use_nonlinear_proj):
                    gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                    gst_outputs = self.dropout(gst_outputs)
                
                if(self.use_proj_linear):
                    gst_outputs = self.proj(gst_outputs)

            # print(gst_outputs)
            # print(len(gst_outputs))
            # for i in range(len(gst_outputs)):
            #     print(gst_outputs[i].shape)

            inputs = inputs + gst_outputs
            return {'styled_inputs': inputs, 'style_embedding': gst_outputs} 

    def finegrainedre_embedding_inference(self, inputs, style_input, text_len, mel_len):
            if style_input is None:
                # ignore style token and return zero tensor
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
            elif style_input.shape == torch.Size([1, self.style_embedding_dim]):
                gst_outputs = style_input
            else:
                # compute style tokens
                # input_args = [style_input]
                print(text_len.shape, mel_len.shape, style_input.shape)
                gst_outputs, alignments = self.layer(inputs, text_len, style_input.unsqueeze(0), mel_len)  # pylint: disable=not-callable
            
                if(self.use_nonlinear_proj):
                    gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                    gst_outputs = self.dropout(gst_outputs)
                
                if(self.use_proj_linear):
                    gst_outputs = self.proj(gst_outputs)

            # print(gst_outputs)
            # print(len(gst_outputs))
            # for i in range(len(gst_outputs)):
            #     print(gst_outputs[i].shape)

            inputs = inputs + gst_outputs
            return {'styled_inputs': inputs, 'style_embedding': gst_outputs} 

    def gst_embedding(self, inputs, style_input, speaker_embedding=None):
            if isinstance(style_input, dict):
                # multiply each style token with a weight
                query = torch.zeros(1, 1, self.style_embedding_dim // 2).type_as(inputs)
                if speaker_embedding is not None:
                    query = torch.cat([query, speaker_embedding.reshape(1, 1, -1)], dim=-1)

                _GST = torch.tanh(self.layer.style_token_layer.style_tokens)
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
                for k_token, v_amplifier in style_input.items():
                    key = _GST[int(k_token)].unsqueeze(0).expand(1, -1, -1)
                    gst_outputs_att = self.layer.style_token_layer.attention(query, key)
                    gst_outputs = gst_outputs + gst_outputs_att * v_amplifier
            elif style_input is None:
                # ignore style token and return zero tensor
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
            else:
                # compute style tokens
                input_args = [style_input, speaker_embedding]
                gst_outputs = self.layer(*input_args)  # pylint: disable=not-callable

            if(self.use_nonlinear_proj):
                # gst_outputs = self.nl_proj(gst_outputs)
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
               
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, gst_outputs)
            else:
                inputs = self._add_speaker_embedding(inputs, gst_outputs)

            return {'styled_inputs': inputs, 'style_embedding': gst_outputs.squeeze(1)}   

    def vae_forward(self, inputs, style_input): 

        vae_output = self.layer.forward(style_input)
        vae_embedding = vae_output['z']

        if(self.use_nonlinear_proj):
            vae_embedding = torch.tanh(self.nl_proj(vae_embedding))
            vae_embedding = self.dropout(vae_embedding)

        if(self.use_proj_linear):
            vae_embedding = self.proj(vae_embedding)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(inputs, vae_embedding.unsqueeze(1))
        else:
            inputs = self._add_speaker_embedding(inputs, vae_embedding.unsqueeze(1))
    
        return {'styled_inputs': inputs, 'style_embedding': vae_embedding, 'mean': vae_output['mean'], 'log_var' : vae_output['log_var']}

    def vae_inference(self, inputs, style_input, z=None):
        # print(inputs.shape, style_input.shape)
        if(z is not None): # If an specific z is passed it uses it
            if(self.agg_type == 'concat'):
                inputs =  self._concat_embedding(inputs, z)  
            else:
                inputs =  self._add_speaker_embedding(inputs, z)
            return {'styled_inputs': inputs, 'style_embedding': z}

        else:
            vae_output = self.layer.forward(style_input.unsqueeze(0))
            vae_embedding = vae_output['z']

            if(self.use_nonlinear_proj):
                vae_embedding = torch.tanh(self.nl_proj(vae_embedding))
                vae_embedding = self.dropout(vae_embedding)
                
            if(self.use_proj_linear):
                vae_embedding = self.proj(vae_embedding)

            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, vae_embedding)
            else:
                inputs =  self._add_speaker_embedding(inputs, vae_embedding)

            return {'styled_inputs': inputs, 'style_embedding': vae_embedding, 'mean': vae_output['mean'], 'log_var' : vae_output['log_var']}

    def vqvae_forward(self, inputs, style_input): 
        vqvae_output = self.layer.forward(style_input)
        
        for key in vqvae_output:
            if(self.use_nonlinear_proj):
                vqvae_output[key] = torch.tanh(self.nl_proj(vqvae_output[key]))
                vqvae_output[key] = self.dropout(vqvae_output[key])

            if(self.use_proj_linear):
                vqvae_output[key] = self.proj(vqvae_output[key])

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(inputs, vqvae_output['z_q_x_st'].unsqueeze(1))       
        elif (self.agg_type == 'adain'):
            return NotImplementedError
        else:
            inputs = self._add_speaker_embedding(inputs, vqvae_output['z_q_x_st'].unsqueeze(1))
        
        return {'styled_inputs': inputs, 'style_embedding': vqvae_output['z_q_x_st'], 'z_e':vqvae_output['z_e'], 'z_q':vqvae_output['z_q']}

    def vqvae_inference(self, inputs, style_input, K=None):
        if(K): # If an specific K is passed it uses it
            z = self.layer.codebook.weight[K].detach()
            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, z)  
            else:
                inputs =  self._add_speaker_embedding(inputs, z)
            return {'styled_inputs': inputs, 'style_embedding': z}

        else:
            vqvae_output = self.layer.forward(style_input)

            for key in vqvae_output:
                if(self.use_nonlinear_proj):
                    vqvae_output[key] = torch.tanh(self.nl_proj(vqvae_output[key]))
                    vqvae_output[key] = self.dropout(vqvae_output[key])
                
                if(self.use_proj_linear):
                    vqvae_output[key]= self.proj(vqvae_output[key])

            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, vqvae_output['z_q_x_st'].unsqueeze(1))       
            elif (self.agg_type == 'adain'):
                return NotImplementedError
            else:
                inputs = self._add_speaker_embedding(inputs, vqvae_output['z_q_x_st'].unsqueeze(1))
        
            return {'styled_inputs': inputs, 'style_embedding': vqvae_output['z_q_x_st'], 'z_e':vqvae_output['z_e'], 'z_q':vqvae_output['z_q']}

    def vaeflow_forward(self, inputs, style_input): 
        vaeflow_output = self.layer.forward(style_input)
        vaeflow_embedding = vaeflow_output['z_T']

        if(self.use_nonlinear_proj):
            vaeflow_embedding = torch.tanh(self.nl_proj(vaeflow_embedding))
            vaeflow_embedding = self.dropout(vaeflow_embedding)

        if(self.use_proj_linear):
            vaeflow_embedding = self.proj(vaeflow_embedding)

        if(self.agg_type == 'concat'):
            inputs = self._concat_embedding(inputs, vaeflow_embedding)
        else:
            inputs = self._add_speaker_embedding(inputs, vaeflow_embedding) 
        return {'styled_inputs': inputs, 'style_embedding': vaeflow_embedding.squeeze(1), 'mean': vaeflow_output['mean'], 'log_var' : vaeflow_output['log_var'], 'z_0' : vaeflow_output['z_0'], 'z_T' : vaeflow_output['z_T']}

    def vaeflow_inference(self, inputs, style_input, z=None):
        if(z): # If an specific z is passed it uses it
            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, z)  
            else:
                inputs = self._add_speaker_embedding(inputs, z)
            return {'styled_inputs': inputs, 'style_embedding': z}

        else:
            vaeflow_output = self.layer.forward(style_input)
            vaeflow_embedding = vaeflow_output['z_T']

            if(self.use_nonlinear_proj):
                vaeflow_embedding = torch.tanh(self.nl_proj(vaeflow_embedding))
                vaeflow_embedding = self.dropout(vaeflow_embedding)
                
            if(self.use_proj_linear):
                vaeflow_embedding = self.proj(vaeflow_embedding)

            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, vaeflow_embedding)
            else:
                inputs = self._add_speaker_embedding(inputs, vaeflow_embedding)
        
            return {'styled_inputs': inputs, 'style_embedding': vaeflow_embedding.squeeze(1), 'mean': vaeflow_output['mean'], 'log_var' : vaeflow_output['log_var'], 'z_0' : vaeflow_output['z_0'], 'z_T' : vaeflow_output['z_T']}

    def diff_forward(self, inputs, style_input):
            diff_output = self.layer.forward(style_input)
            diff_embedding = diff_output['style']
            if(self.use_nonlinear_proj):
                diff_embedding = torch.tanh(self.nl_proj(diff_embedding))
                diff_embedding = self.dropout(diff_embedding)

            if(self.use_proj_linear):
                diff_embedding = self.proj(diff_embedding)

            if(self.agg_type == 'concat'):
                inputs =  self._concat_embedding(inputs, diff_embedding)
            else:
                inputs =  self._add_speaker_embedding(inputs, diff_embedding)
            return {'styled_inputs': inputs, 'style_embedding': diff_embedding.squeeze(1), 'noises': diff_output['noises']}

    def diff_inference(self, inputs, style_input, infer_from):
            diff_output = self.layer.inference(style_input, infer_from)
            diff_embedding = diff_output['style']
            if(self.use_nonlinear_proj):
                diff_embedding = torch.tanh(self.nl_proj(diff_embedding))
                diff_embedding = self.dropout(diff_embedding)

            if(self.use_proj_linear):
                diff_embedding = self.proj(diff_embedding)

            if(self.agg_type == 'concat'):
                inputs =  self._concat_embedding(inputs, diff_embedding)
            else:
                inputs =  self._add_speaker_embedding(inputs, diff_embedding)
            return {'styled_inputs': inputs, 'style_embedding': diff_embedding.squeeze(1), 'noises': diff_output['noises']}

    ##########################
    ###### Aggregations ######
    ##########################

    # For this two below, remember if B is batch size, L the sequence length, E is the embedding dim and D the style embed dim
    # for tacotron2 the encoder outputs comes [B,L,E] and faspitch comes [B,E,L]
    # @classmethod
    def _concat_embedding(self, outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embedded_speakers_], dim=-1)
        return outputs

    # @classmethod
    def _add_speaker_embedding(self, outputs, embedded_speakers):
        # Fixed to the forwardtts, now, for adding we normalize by l2 norm
        if(self.agg_norm == True):
            embedded_speakers_ = nn.functional.normalize(embedded_speakers).expand(outputs.size(0), outputs.size(1), -1)
        else:
            embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = outputs + embedded_speakers_
        return outputs

    # we assume that the encoder "outputs" will get the shape [B,L,E], so we mean over L and apply adain in it
    def _adain(self, outputs, embedded_speakers):

        mean_content = torch.mean(outputs, dim= [-1])
        std_content = torch.std(outputs, dim= [-1]) + 1e-5

        # print('embed_speakers shape : ' , embedded_speakers.shape)

        # embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), outputs.size(2))
        
        # print('embed_speakers_ shape : ' , embedded_speakers_.shape)

        mean_style = torch.mean(embedded_speakers, dim= [-1])
        std_style = torch.std(embedded_speakers, dim= [-1]) + 1e-5

        # print(mean_style.shape, std_style.shape, mean_style, std_style)

        # print(mean_content.shape, std_content.shape, mean_content, std_content)
        
        mean_style = mean_style.unsqueeze(1).expand(outputs.size(0), outputs.size(1), outputs.size(2))
        std_style = std_style.unsqueeze(1).expand(outputs.size(0), outputs.size(1), outputs.size(2))

        mean_content = mean_content.unsqueeze(2).expand(outputs.size(0), outputs.size(1), outputs.size(2))
        std_content = std_content.unsqueeze(2).expand(outputs.size(0), outputs.size(1), outputs.size(2))

        # print(mean_style.shape, std_style.shape, mean_style, std_style)

        # print(mean_content.shape, std_content.shape, mean_content, std_content)

        # add verbose to debug nan errors: the hypothesis is that the mean becomes high
        # print(mean_content, std_content, mean_style, std_style) # Apparently the error was dividing by 0, added 1e-5

        return (outputs - mean_content)/std_content*std_style + mean_style
