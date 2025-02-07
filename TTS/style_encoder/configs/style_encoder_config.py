from dataclasses import asdict, dataclass
from coqpit import Coqpit, check_argument
from typing import List

@dataclass
class StyleEncoderConfig(Coqpit):
    """Defines the Generic Style Encoder Config

    Args:    
        embedding_dim (int):
            Defines the size of the embedding vector dimensions. Defaults to 256.
        
        gst_style_input_weights (dict):
            Defines the weights for each style token used at inference. Defaults to None.

        gst_num_heads (int):
            Number of attention heads used by the multi-head attention. Defaults to 4.

        gst_num_style_tokens (int):
            Number of style token vectors. Defaults to 10.

        se_type (str):
            Style encoder type \in {diffusion, vae, vaeflow, gst, re}
    """
    
    #########################
    #### General Configs ####
    #########################
    
    # Style Encoder Type
    se_type: str = "re"
    start_loss_at: int = 0 # Currently only used on NON-RE-GST architectures
    use_supervised_style: bool = False # Enables supervision (Look-Up or style class)
        
    # Style Encoder Inputs
    num_mel: int = 80
    decompose_ref_mel: bool = False
    style_reference_features: List[str] = None # Entries can be pitch, energy or melspectrogram

    # Style Encoder Outputs
    style_embedding_dim: int = 128
    proj_dim: int = 384 # Projection dim, often the encoder output (512 is the tacotron2 default encoder output, 384 for fastpitch)
    use_nonlinear_proj: bool = False # Whether use or not a linear (last_dim, last_dim) + tanh before agg in TTS encoder outputs

    # Timbre perturbation
    use_timbre_perturbation: bool = False

    # Normalization
    use_melstat_norm: bool = False # If want to use pre computed melspec statistic only in input of style encoder

    # Aggregation
    agg_type: str = "concat" # Can be concat, sum, or adain
    agg_norm: bool = False # If agg_type == sum, you can rather than normalizing or not
    use_proj_linear: bool = False # Whether use linear projection to decoder dim or not (specifcally useful for sum agg_style)
    agg_spk_emb_decoder: bool = False # If you want to agg speaker embedding only in decoder input (and not in varianca adaptors)
    agg_stl_emb_adaptors: bool = False # If you want to agg style embedding only in varianca adaptors (and not in decoder input)

    # Style Classifier
    use_guided_style: bool = False # Whether use guided style encoder training
    guided_style_loss_type: str = 'ce' # or focal_loss
    
    # Speaker Classifier + GRL
    use_grl_on_speakers_in_style_embedding: bool = False # Whether use or not GRL in style embedding output avoinding speaker information
    grl_loss_type: str = 'ce' # or focal loss
    use_inverter: bool = False # whether use gri instead of grl
    grl_alpha: float = 1 # GRL alpha, still one for all GRL's
        
    # Orthogonal losses   
    orthogonal_loss: bool = False  
    orthogonal_loss_alpha: float = 1.0
    content_orthogonal_loss: bool = False # whether use othogonal loss between style and content embeddings
    speaker_orthogonal_loss: bool = False # whether use othogonal loss between speaker and content embeddings
        
    # Residual Speaker
    use_residual_speaker_disentanglement: bool = False # If true, decomposes style embedding in F and SE-F, where SE-F will represent speakers

    # Cycle consistency
    use_cycle_consistency: bool = False 
    cycle_consistency_alpha: float = 1
    start_cycle_at: int = 1 # Step to start back propagating cycle loss
        
    # CLIP loss configs
    use_clip_loss: bool = False 
    clip_alpha_loss: float = 1
    clip_temperature: float = 1
    
    # Style ditortion configs
    use_style_distortion_loss: bool = False
    style_distortion_alpha_loss: float = 1
        
    #######################################
    #### Architecture Specific Configs ####
    #######################################
    
    # Finegrained RE 
    fg_attention_dropout: float = 0.1
    fg_attention_dim: int = 384
    prosody_embedding_dim: int = 3 #Default to align with text, according to reference repo, by my first look it should be = proj_dim = encoder_output_dim

    # GST
    gst_style_input_weights: dict = None
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    # VAE
    use_cyclical_annealing: bool = True # Whether use or not annealing (recommended true), only linear implemented
    vae_loss_alpha: float = 1.0 # Default alpha value (term of KL loss)
    vae_cycle_period: int = 5000 # iteration period to apply a new annealing cycle

    # VQVAE
    codebook_size: int = 16 # Codebook size
    vqvae_commitment_beta: float = 0.25 # Depends on the scale of the reconstruction loss

    # VAEFLOW
    vaeflow_intern_dim: int = 300
    vaeflow_number_of_flows: int = 16

    # Diffusion
    diff_num_timesteps: int = 25 
    diff_schedule_type: str = 'cosine'
    diff_loss_type: str = 'l1' 
    diff_ref_online: bool = True
    diff_step_dim: int = 128
    diff_in_out_ch: int = 1 
    diff_num_heads: int = 1 
    diff_hidden_channels: int = 128 
    diff_num_blocks: int = 5
    diff_dropout: float = 0.1
    diff_loss_alpha: float = 0.75

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        super().check_values()

        check_argument("se_type", c, restricted=True, enum_list=["lookup", "gst", "re","vae","vqvae", "diffusion", "vaeflow","finegrainedre","modifiedre", "bidirectionalre","metastyle", "inre"])
        check_argument("agg_type", c, restricted=True, enum_list=["sum", "concat", "adain"])
        check_argument("num_mel", c, restricted=False)
        check_argument("style_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)
        check_argument("vaeflow_intern_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("vaeflow_number_of_flows", c, restricted=True, min_val=0, max_val=1000)
        check_argument("diff_num_timesteps", c, restricted=True, min_val=0, max_val=5000)
        check_argument("diff_schedule_type", c, restricted=True, enum_list=["cosine", "linear"])
        check_argument("diff__step", c, restricted=True, min_val=0, max_val=self.num_timesteps)
        check_argument("diff_loss_type", c, restricted=True, enum_list=["l1", "mse"])
        check_argument("diff_ref_online", c, restricted=True, enum_list=[True, False])
        check_argument("diff_step_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("diff_in_out_ch", c, restricted=True, min_val=1, max_val=1)
        check_argument("diff_num_heads", c, restricted=False)
        check_argument("diff_hidden_channels", c, restricted=True, min_val=0, max_val=2048)
        check_argument("diff_num_blocks", c, restricted=True, min_val=0, max_val=100)
        check_argument("diff_dropout", c, restricted=False)
        check_argument("diff_loss_alpha", c, restricted=False)





