# Motus Policy for RoboTwin

# for save csv
import csv

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

# # =========================
# # Test-Time Scaling Defaults
# # =========================
# DEFAULT_TTS_ENABLE = False
# DEFAULT_TTS_NUM_SAMPLES = 1
# DEFAULT_TTS_LOG_ACTIONS = True
# DEFAULT_TTS_SAVE_FULL_ACTIONS = True

# =========================
# Test-Time Scaling Defaults
# =========================
DEFAULT_TTS_ENABLE = False
DEFAULT_TTS_NUM_SAMPLES = 8

# Methods:
#   global_medoid: average-L2/global-medoid selection
#   keystone:      unimodality guard + kmeans + largest-cluster medoid
#   rank_softmax:  rank-based stochastic selection, P(i)=softmax(-rank_i/tau)
DEFAULT_TTS_METHOD = "global_medoid"

# # KeyStone-style defaults
# DEFAULT_TTS_NUM_CLUSTERS = 2
# DEFAULT_TTS_TAU = 0.3
# DEFAULT_TTS_KMEANS_ITERS = 10

# TTS defaults
DEFAULT_TTS_NUM_CLUSTERS = 2

# tau meaning:
#   keystone:     unimodality guard threshold
#   rank_softmax: rank-softmax temperature
DEFAULT_TTS_TAU = 0.3

DEFAULT_TTS_KMEANS_ITERS = 10

# Logging
DEFAULT_TTS_LOG_ACTIONS = True

# Deprecated/no-op: kept only for backward compatibility with old scripts.
# New implementation does not save .npz files.
DEFAULT_TTS_SAVE_FULL_ACTIONS = False

# tts add
def _as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in ["1", "true", "yes", "y", "on"]

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        config_path: str, 
        wan_path: str, 
        vlm_path: str, 
        device: str = "cuda", 
        log_dir: Optional[str] = None, 
        task_name: Optional[str] = None,
        # tts_enable: bool = DEFAULT_TTS_ENABLE,
        # tts_num_samples: int = DEFAULT_TTS_NUM_SAMPLES,
        # tts_log_actions: bool = DEFAULT_TTS_LOG_ACTIONS,
        # tts_save_full_actions: bool = DEFAULT_TTS_SAVE_FULL_ACTIONS,   
        tts_enable: bool = DEFAULT_TTS_ENABLE,
        tts_num_samples: int = DEFAULT_TTS_NUM_SAMPLES,
        tts_method: str = DEFAULT_TTS_METHOD,
        tts_num_clusters: int = DEFAULT_TTS_NUM_CLUSTERS,
        tts_tau: float = DEFAULT_TTS_TAU,
        tts_kmeans_iters: int = DEFAULT_TTS_KMEANS_ITERS,
        tts_log_actions: bool = DEFAULT_TTS_LOG_ACTIONS,
        tts_save_full_actions: bool = DEFAULT_TTS_SAVE_FULL_ACTIONS,             
    ):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load normalization stats
        self._load_normalization_stats()
        
        # Initialize image saving
        self.save_images = True
        
        # base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        # task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        # self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        # self.save_dir.mkdir(parents=True, exist_ok=True)
        # self.episode_count = 0
        # self.step_count = 0

        # logger.info("Motus Policy initialized successfully")
        
        
        # tts add
        
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"

        self.log_dir = Path(base_log_dir)
        self.task_dir_name = task_dir_name

        self.save_dir = self.log_dir / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # self.tts_enable = bool(tts_enable)
        # self.tts_num_samples = max(1, int(tts_num_samples))
        # self.tts_log_actions = bool(tts_log_actions)
        # self.tts_save_full_actions = bool(tts_save_full_actions)
        
        # self.tts_enable = _as_bool(tts_enable)
        # self.tts_num_samples = max(1, int(tts_num_samples))
        # self.tts_log_actions = _as_bool(tts_log_actions)
        # self.tts_save_full_actions = _as_bool(tts_save_full_actions)
        
        # self.tts_dir = self.log_dir / "tts" / task_dir_name

        # if self.tts_enable and self.tts_save_full_actions:
        #     self.tts_dir.mkdir(parents=True, exist_ok=True)

        # self.episode_count = 0
        # self.step_count = 0

        # print(
        #     f"[TTS] enable={self.tts_enable}, "
        #     f"num_samples={self.tts_num_samples}, "
        #     f"log_actions={self.tts_log_actions}, "
        #     f"save_full_actions={self.tts_save_full_actions}, "
        #     f"tts_dir={self.tts_dir}"
        # )
        
        self.tts_enable = _as_bool(tts_enable)
        self.tts_num_samples = max(1, int(tts_num_samples))

        self.tts_method = str(tts_method).strip().lower()
        # allowed_tts_methods = {"global_medoid", "keystone"}
        allowed_tts_methods = {"global_medoid", "keystone", "rank_softmax"}
        if self.tts_method not in allowed_tts_methods:
            raise ValueError(
                f"Unknown tts_method={self.tts_method}. "
                f"Allowed methods: {sorted(allowed_tts_methods)}"
            )

        self.tts_num_clusters = max(1, int(tts_num_clusters))
        self.tts_tau = float(tts_tau)
        self.tts_kmeans_iters = max(1, int(tts_kmeans_iters))

        self.tts_log_actions = _as_bool(tts_log_actions)

        # Deprecated/no-op. Kept so old command lines still parse.
        self.tts_save_full_actions = _as_bool(tts_save_full_actions)

        self.tts_dir = self.log_dir / "tts" / task_dir_name

        if self.tts_enable and self.tts_log_actions:
            self.tts_dir.mkdir(parents=True, exist_ok=True)

        self.episode_count = 0
        self.step_count = 0

        print(
            f"[TTS] enable={self.tts_enable}, "
            f"num_samples={self.tts_num_samples}, "
            f"method={self.tts_method}, "
            f"num_clusters={self.tts_num_clusters}, "
            f"tau={self.tts_tau}, "
            f"kmeans_iters={self.tts_kmeans_iters}, "
            f"log_actions={self.tts_log_actions}, "
            f"save_full_actions={self.tts_save_full_actions} (deprecated/no-op), "
            f"tts_dir={self.tts_dir}"
        )

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

    # tts add
    def _run_single_inference(self, current_frame, t5_list, vlm_inputs, num_inference_steps):
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )
        return predicted_frames, predicted_actions

    # # tts add
    # def _select_tts_medoid(self, actions_stack: torch.Tensor):
    #     # actions_stack: [N, H, D], already on CPU
    #     n = actions_stack.shape[0]
    #     flat_actions = actions_stack.reshape(n, -1)
    #     pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)
    #     avg_l2 = pairwise_l2.sum(dim=1) / (n - 1)
    #     best_idx = int(torch.argmin(avg_l2).item())
    #     return best_idx, pairwise_l2, avg_l2

    # # # tts add
    # # def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, best_idx):
    # #     if not self.tts_log_actions:
    # #         return

    # #     avg_list = [round(float(x), 6) for x in avg_l2]
    # #     print(
    # #         f"[TTS] episode={self.episode_count} "
    # #         f"step={self.step_count} "
    # #         f"samples={actions_stack.shape[0]} "
    # #         f"best={best_idx} "
    # #         f"avg_l2={avg_list}"
    # #     )

    # #     pairwise_np = pairwise_l2.numpy()
    # #     print(
    # #         "[TTS] pairwise_l2=\n"
    # #         + np.array2string(pairwise_np, precision=4, suppress_small=True)
    # #     )

    # #     if self.tts_save_full_actions:
    # #         save_path = self.tts_dir / f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.npz"
    # #         np.savez_compressed(
    # #             save_path,
    # #             actions=actions_stack.numpy(),
    # #             pairwise_l2=pairwise_l2.numpy(),
    # #             avg_l2=avg_l2.numpy(),
    # #             best_idx=np.array(best_idx, dtype=np.int64),
    # #         )
    # #         print(f"[TTS] saved full action chunks to {save_path}")

    # # tts add
    # def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, best_idx):
    #     if not self.tts_log_actions:
    #         return

    #     self.tts_dir.mkdir(parents=True, exist_ok=True)

    #     avg_l2_np = avg_l2.numpy()
    #     pairwise_l2_np = pairwise_l2.numpy()
    #     actions_np = actions_stack.numpy()

    #     best_avg_l2 = float(avg_l2_np[best_idx])
    #     min_avg_l2 = float(avg_l2_np.min())
    #     max_avg_l2 = float(avg_l2_np.max())
    #     mean_avg_l2 = float(avg_l2_np.mean())

    #     avg_list = [round(float(x), 6) for x in avg_l2_np]

    #     print(
    #         f"[TTS] episode={self.episode_count} "
    #         f"step={self.step_count} "
    #         f"samples={actions_stack.shape[0]} "
    #         f"best={best_idx} "
    #         f"best_avg_l2={best_avg_l2:.6f} "
    #         f"min_avg_l2={min_avg_l2:.6f} "
    #         f"max_avg_l2={max_avg_l2:.6f}"
    #     )

    #     summary_path = self.tts_dir / "summary.csv"
    #     write_header = not summary_path.exists()

    #     with open(summary_path, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         if write_header:
    #             writer.writerow([
    #                 "episode",
    #                 "step",
    #                 "samples",
    #                 "best_idx",
    #                 "best_avg_l2",
    #                 "min_avg_l2",
    #                 "max_avg_l2",
    #                 "mean_avg_l2",
    #                 "avg_l2",
    #                 "npz_path",
    #             ])

    #         npz_path = ""
    #         if self.tts_save_full_actions:
    #             npz_path = str(self.tts_dir / f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.npz")

    #         writer.writerow([
    #             self.episode_count,
    #             self.step_count,
    #             actions_stack.shape[0],
    #             best_idx,
    #             f"{best_avg_l2:.8f}",
    #             f"{min_avg_l2:.8f}",
    #             f"{max_avg_l2:.8f}",
    #             f"{mean_avg_l2:.8f}",
    #             avg_list,
    #             npz_path,
    #         ])

    #     if self.tts_save_full_actions:
    #         save_path = self.tts_dir / f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.npz"
    #         np.savez_compressed(
    #             save_path,
    #             actions=actions_np,
    #             pairwise_l2=pairwise_l2_np,
    #             avg_l2=avg_l2_np,
    #             best_idx=np.array(best_idx, dtype=np.int64),
    #         )
    #         # print(f"[TTS] saved full action chunks to {save_path}")

    # # tts add
    # def _select_tts_action(self, actions_stack: torch.Tensor):
    #     """
    #     Dispatch TTS selector according to self.tts_method.

    #     actions_stack: [K, H, D], CPU tensor.
    #     """
    #     if self.tts_method == "global_medoid":
    #         return self._select_tts_global_medoid(actions_stack)

    #     if self.tts_method == "keystone":
    #         return self._select_tts_keystone(actions_stack)

    #     raise ValueError(f"Unknown tts_method={self.tts_method}")
    
    # tts v2 add
    def _select_tts_action(self, actions_stack: torch.Tensor):
        """
        Dispatch TTS selector according to self.tts_method.

        actions_stack: [K, H, D], CPU tensor.
        """
        if self.tts_method == "global_medoid":
            return self._select_tts_global_medoid(actions_stack)

        if self.tts_method == "keystone":
            return self._select_tts_keystone(actions_stack)

        if self.tts_method == "rank_softmax":
            return self._select_tts_rank_softmax(actions_stack)

        raise ValueError(f"Unknown tts_method={self.tts_method}")    

    # ttsv2 add
    def _build_tts_ranks(self, avg_l2: torch.Tensor):
        """
        Build ranks from avg_l2.

        avg_l2: lower is better.
        return:
            ranks: [K], rank 0 is best / smallest avg_l2.
            order: [K], candidate indices sorted by avg_l2 ascending.
        """
        order = torch.argsort(avg_l2, descending=False)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(len(avg_l2), dtype=order.dtype, device=avg_l2.device)
        return ranks, order

    # tts add
    def _select_tts_global_medoid(self, actions_stack: torch.Tensor):
        """
        Current baseline method:
        flatten each action chunk, compute pairwise L2, select the global medoid.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            best_idx = 0
        else:
            avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)
            best_idx = int(torch.argmin(avg_l2).item())

        # ttsv2 add
        ranks, order = self._build_tts_ranks(avg_l2)
        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[best_idx] = 1.0

        # metrics = {
        #     "tts_method": "global_medoid",
        #     "selection_stage": "global_medoid",
        #     "global_medoid_idx": best_idx,
        #     "unimodal": True,
        #     "s_score": "",
        #     "num_clusters": 1,
        #     "selected_cluster": -1,
        #     "selected_cluster_size": k,
        #     "cluster_counts": "",
        #     "cluster_ids": "",
        # }
        
        # ttsv2 add
        metrics = {
            "tts_method": "global_medoid",
            "selection_stage": "global_medoid",
            "global_medoid_idx": best_idx,
            "unimodal": True,
            "s_score": "",
            "num_clusters": 1,
            "selected_cluster": -1,
            "selected_cluster_size": k,
            "cluster_counts": "",
            "cluster_ids": "",
            "selected_idx": best_idx,
            "selected_rank": int(ranks[best_idx].item()),
            "selected_prob": "1.00000000",
            "rank_temperature": "",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return best_idx, pairwise_l2, avg_l2, metrics

    # ttsv2 add
    def _select_tts_rank_softmax(self, actions_stack: torch.Tensor):
        """
        Rank-softmax selector:
        1. Flatten each action chunk.
        2. Compute pairwise L2.
        3. Compute avg_l2 for each candidate.
        4. Rank candidates by avg_l2.
        5. Sample selected_idx from softmax(-rank / tau).

        tau is self.tts_tau for this method.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            selected_idx = 0
            global_medoid_idx = 0
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)
            selected_rank = 0
            selected_prob = 1.0
        else:
            avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)

            ranks, order = self._build_tts_ranks(avg_l2)
            global_medoid_idx = int(order[0].item())

            tau = max(1e-8, float(self.tts_tau))

            logits = -ranks.to(torch.float32) / tau
            selection_probs = torch.softmax(logits, dim=0)

            selected_idx = int(torch.multinomial(selection_probs, num_samples=1).item())
            selected_rank = int(ranks[selected_idx].item())
            selected_prob = float(selection_probs[selected_idx].item())

        metrics = {
            "tts_method": "rank_softmax",
            "selection_stage": "rank_softmax_sample",
            "global_medoid_idx": global_medoid_idx,
            "unimodal": "",
            "s_score": "",
            "num_clusters": 1,
            "selected_cluster": -1,
            "selected_cluster_size": k,
            "cluster_counts": "",
            "cluster_ids": "",
            "selected_idx": selected_idx,
            "selected_rank": selected_rank,
            "selected_prob": f"{selected_prob:.8f}",
            "rank_temperature": f"{float(self.tts_tau):.8f}",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return selected_idx, pairwise_l2, avg_l2, metrics    

    # tts add
    def _compute_global_medoid_and_guard(
        self,
        flat_actions: torch.Tensor,
        pairwise_l2: torch.Tensor,
    ):
        """
        KeyStone-style unimodality guard.

        s_score = ||mean(candidate_chunks) - global_medoid|| / median_pairwise_distance

        If s_score < tau, treat candidates as one cluster and use global medoid.
        """
        k = flat_actions.shape[0]

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            return 0, avg_l2, 0.0

        avg_l2 = pairwise_l2.sum(dim=1) / (k - 1)
        global_medoid_idx = int(torch.argmin(avg_l2).item())

        iu = torch.triu_indices(k, k, offset=1, device=pairwise_l2.device)
        median_d = pairwise_l2[iu[0], iu[1]].median()

        eps = 1e-8
        s_score = (
            torch.norm(flat_actions.mean(dim=0) - flat_actions[global_medoid_idx], p=2)
            / (median_d + eps)
        )

        return global_medoid_idx, avg_l2, float(s_score.item())

    # tts add
    def _kmeans_small(
        self,
        x: torch.Tensor,
        num_clusters: int,
        max_iter: int,
    ):
        """
        Small deterministic k-means for K action chunks.

        x: [K, D], CPU tensor.
        return: labels [K].

        Initialization:
        - first center = global medoid
        - next centers = farthest-first
        This avoids consuming random seeds during evaluation.
        """
        k = x.shape[0]
        c = min(max(1, int(num_clusters)), k)

        if c == 1:
            return torch.zeros(k, dtype=torch.long, device=x.device)

        dists = torch.cdist(x, x, p=2)
        first_idx = int(torch.argmin(dists.sum(dim=1)).item())

        selected = [first_idx]
        centers = [x[first_idx].clone()]

        for _ in range(1, c):
            center_tensor = torch.stack(centers, dim=0)
            dist_to_centers = torch.cdist(x, center_tensor, p=2).min(dim=1).values

            for idx in selected:
                dist_to_centers[idx] = -1.0

            next_idx = int(torch.argmax(dist_to_centers).item())
            selected.append(next_idx)
            centers.append(x[next_idx].clone())

        centers = torch.stack(centers, dim=0)
        labels = torch.full((k,), -1, dtype=torch.long, device=x.device)

        for _ in range(max_iter):
            new_labels = torch.cdist(x, centers, p=2).argmin(dim=1)

            if torch.equal(new_labels, labels):
                break

            labels = new_labels

            for cid in range(c):
                mask = labels == cid
                if mask.any():
                    centers[cid] = x[mask].mean(dim=0)
                else:
                    # Empty cluster: re-seed with point farthest from current centers.
                    dist_to_centers = torch.cdist(x, centers, p=2).min(dim=1).values
                    centers[cid] = x[int(torch.argmax(dist_to_centers).item())].clone()

        return labels

    # ttsv2 add
    def _select_tts_keystone(self, actions_stack: torch.Tensor):
        """
        KeyStone-style selector:
        1. Compute global medoid.
        2. Compute unimodality score s_score.
        3. If s_score < tau: return global medoid.
        4. Else: k-means into C clusters.
        5. Select the largest cluster.
        6. Return the medoid inside the largest cluster.
        """
        k = actions_stack.shape[0]
        flat_actions = actions_stack.reshape(k, -1)

        pairwise_l2 = torch.cdist(flat_actions, flat_actions, p=2)

        if k <= 1:
            avg_l2 = torch.zeros(k, dtype=pairwise_l2.dtype, device=pairwise_l2.device)
            
            ranks = torch.zeros(k, dtype=torch.long, device=pairwise_l2.device)
            selection_probs = torch.ones(k, dtype=torch.float32, device=pairwise_l2.device)
            
            metrics = {
                "tts_method": "keystone",
                "selection_stage": "single_sample",
                "global_medoid_idx": 0,
                "unimodal": True,
                "s_score": "",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": 0,
                "selected_rank": 0,
                "selected_prob": "1.00000000",
                "rank_temperature": "",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return 0, pairwise_l2, avg_l2, metrics

        global_medoid_idx, avg_l2, s_score = self._compute_global_medoid_and_guard(
            flat_actions=flat_actions,
            pairwise_l2=pairwise_l2,
        )

        # Unimodality guard: if candidates look like one cluster,
        # do not force k-means to split them.
        if s_score < self.tts_tau or self.tts_num_clusters <= 1:
            ranks, order = self._build_tts_ranks(avg_l2)
            selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
            selection_probs[global_medoid_idx] = 1.0
            
            metrics = {
                "tts_method": "keystone",
                "selection_stage": "guard_global_medoid",
                "global_medoid_idx": global_medoid_idx,
                "unimodal": True,
                "s_score": f"{s_score:.8f}",
                "num_clusters": 1,
                "selected_cluster": -1,
                "selected_cluster_size": k,
                "cluster_counts": "",
                "cluster_ids": "",
                "selected_idx": global_medoid_idx,
                "selected_rank": int(ranks[global_medoid_idx].item()),
                "selected_prob": "1.00000000",
                "rank_temperature": "",
                "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
                "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
            }
            return global_medoid_idx, pairwise_l2, avg_l2, metrics

        c = min(self.tts_num_clusters, k)
        cluster_ids = self._kmeans_small(
            flat_actions,
            num_clusters=c,
            max_iter=self.tts_kmeans_iters,
        )

        cluster_counts = torch.bincount(cluster_ids, minlength=c)
        selected_cluster = int(torch.argmax(cluster_counts).item())

        mask = cluster_ids == selected_cluster
        idxs = mask.nonzero(as_tuple=True)[0]

        # Medoid inside the largest cluster, not the centroid itself.
        sub_dists = pairwise_l2[mask][:, mask]
        local_idx = int(torch.argmin(sub_dists.sum(dim=1)).item())
        best_idx = int(idxs[local_idx].item())
        
        ranks, order = self._build_tts_ranks(avg_l2)
        selection_probs = torch.zeros(k, dtype=torch.float32, device=avg_l2.device)
        selection_probs[best_idx] = 1.0

        metrics = {
            "tts_method": "keystone",
            "selection_stage": "cluster_medoid",
            "global_medoid_idx": global_medoid_idx,
            "unimodal": False,
            "s_score": f"{s_score:.8f}",
            "num_clusters": c,
            "selected_cluster": selected_cluster,
            "selected_cluster_size": int(cluster_counts[selected_cluster].item()),
            "cluster_counts": "|".join(map(str, cluster_counts.detach().cpu().tolist())),
            "cluster_ids": "|".join(map(str, cluster_ids.detach().cpu().tolist())),
            "selected_idx": best_idx,
            "selected_rank": int(ranks[best_idx].item()),
            "selected_prob": "1.00000000",
            "rank_temperature": "",
            "rank_ids": "|".join(map(str, ranks.detach().cpu().tolist())),
            "selection_probs": "|".join(f"{float(x):.8f}" for x in selection_probs.detach().cpu().tolist()),
        }

        return best_idx, pairwise_l2, avg_l2, metrics

    # # tts add
    # # def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, best_idx, metrics):
    # def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, selected_idx, metrics):
    #     """
    #     Log TTS selector information.

    #     Important:
    #     - Fixed CSV schema for all methods.
    #     - No .npz saving.
    #     - Unsupported fields for a method are filled with empty string or -1.
    #     """
    #     if not self.tts_log_actions:
    #         return

    #     self.tts_dir.mkdir(parents=True, exist_ok=True)

    #     avg_l2_np = avg_l2.detach().cpu().numpy()

    #     best_avg_l2 = float(avg_l2_np[best_idx])
    #     min_avg_l2 = float(avg_l2_np.min())
    #     max_avg_l2 = float(avg_l2_np.max())
    #     mean_avg_l2 = float(avg_l2_np.mean())

    #     avg_list = "|".join(f"{float(x):.6f}" for x in avg_l2_np)

    #     print(
    #         f"[TTS] episode={self.episode_count} "
    #         f"step={self.step_count} "
    #         f"method={metrics.get('tts_method', self.tts_method)} "
    #         f"stage={metrics.get('selection_stage', '')} "
    #         f"samples={actions_stack.shape[0]} "
    #         f"C={self.tts_num_clusters} "
    #         f"tau={self.tts_tau} "
    #         f"kmeans_iters={self.tts_kmeans_iters} "
    #         f"best={best_idx} "
    #         f"global_medoid={metrics.get('global_medoid_idx', '')} "
    #         f"unimodal={metrics.get('unimodal', '')} "
    #         f"s_score={metrics.get('s_score', '')} "
    #         f"selected_cluster={metrics.get('selected_cluster', '')} "
    #         f"cluster_size={metrics.get('selected_cluster_size', '')} "
    #         f"cluster_counts={metrics.get('cluster_counts', '')} "
    #         f"best_avg_l2={best_avg_l2:.6f} "
    #         f"min_avg_l2={min_avg_l2:.6f} "
    #         f"max_avg_l2={max_avg_l2:.6f}"
    #     )

    #     summary_path = self.tts_dir / "summary.csv"
    #     write_header = not summary_path.exists()

    #     with open(summary_path, "a", newline="") as f:
    #         writer = csv.writer(f)

    #         if write_header:
    #             writer.writerow([
    #                 "episode",
    #                 "step",
    #                 "samples",
    #                 "tts_method",
    #                 "selection_stage",
    #                 "best_idx",
    #                 "global_medoid_idx",
    #                 "unimodal",
    #                 "s_score",
    #                 "num_clusters",
    #                 "tau",
    #                 "kmeans_iters",
    #                 "selected_cluster",
    #                 "selected_cluster_size",
    #                 "cluster_counts",
    #                 "cluster_ids",
    #                 "best_avg_l2",
    #                 "min_avg_l2",
    #                 "max_avg_l2",
    #                 "mean_avg_l2",
    #                 "avg_l2",
    #             ])

    #         writer.writerow([
    #             self.episode_count,
    #             self.step_count,
    #             actions_stack.shape[0],
    #             metrics.get("tts_method", self.tts_method),
    #             metrics.get("selection_stage", ""),
    #             best_idx,
    #             metrics.get("global_medoid_idx", ""),
    #             metrics.get("unimodal", ""),
    #             metrics.get("s_score", ""),
    #             metrics.get("num_clusters", self.tts_num_clusters),
    #             self.tts_tau,
    #             self.tts_kmeans_iters,
    #             metrics.get("selected_cluster", ""),
    #             metrics.get("selected_cluster_size", ""),
    #             metrics.get("cluster_counts", ""),
    #             metrics.get("cluster_ids", ""),
    #             f"{best_avg_l2:.8f}",
    #             f"{min_avg_l2:.8f}",
    #             f"{max_avg_l2:.8f}",
    #             f"{mean_avg_l2:.8f}",
    #             avg_list,
    #         ])
    
    # ttsv2 add
    def _log_tts_result(self, actions_stack, pairwise_l2, avg_l2, selected_idx, metrics):
        """
        Log TTS selector information.

        Important:
        - Fixed CSV schema for all methods.
        - No .npz saving.
        - Unsupported fields for a method are filled with empty string or -1.
        """
        if not self.tts_log_actions:
            return

        self.tts_dir.mkdir(parents=True, exist_ok=True)

        avg_l2_np = avg_l2.detach().cpu().numpy()

        selected_avg_l2 = float(avg_l2_np[selected_idx])
        min_avg_l2 = float(avg_l2_np.min())
        max_avg_l2 = float(avg_l2_np.max())
        mean_avg_l2 = float(avg_l2_np.mean())

        avg_list = "|".join(f"{float(x):.6f}" for x in avg_l2_np)

        global_medoid_idx = metrics.get("global_medoid_idx", "")
        global_medoid_avg_l2 = ""
        if global_medoid_idx != "":
            global_medoid_avg_l2 = f"{float(avg_l2_np[int(global_medoid_idx)]):.8f}"

        print(
            f"[TTS] episode={self.episode_count} "
            f"step={self.step_count} "
            f"method={metrics.get('tts_method', self.tts_method)} "
            f"stage={metrics.get('selection_stage', '')} "
            f"samples={actions_stack.shape[0]} "
            f"C={self.tts_num_clusters} "
            f"tau={self.tts_tau} "
            f"kmeans_iters={self.tts_kmeans_iters} "
            f"selected={selected_idx} "
            f"selected_rank={metrics.get('selected_rank', '')} "
            f"selected_prob={metrics.get('selected_prob', '')} "
            f"global_medoid={global_medoid_idx} "
            f"unimodal={metrics.get('unimodal', '')} "
            f"s_score={metrics.get('s_score', '')} "
            f"selected_cluster={metrics.get('selected_cluster', '')} "
            f"cluster_size={metrics.get('selected_cluster_size', '')} "
            f"cluster_counts={metrics.get('cluster_counts', '')} "
            f"selected_avg_l2={selected_avg_l2:.6f} "
            f"global_medoid_avg_l2={global_medoid_avg_l2} "
            f"min_avg_l2={min_avg_l2:.6f} "
            f"max_avg_l2={max_avg_l2:.6f}"
        )

        summary_path = self.tts_dir / "summary.csv"
        write_header = not summary_path.exists()

        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow([
                    "episode",
                    "step",
                    "samples",
                    "tts_method",
                    "selection_stage",

                    "selected_idx",
                    "selected_rank",
                    "selected_prob",

                    "global_medoid_idx",
                    "global_medoid_avg_l2",

                    "unimodal",
                    "s_score",
                    "num_clusters",
                    "tau",
                    "kmeans_iters",

                    "selected_cluster",
                    "selected_cluster_size",
                    "cluster_counts",
                    "cluster_ids",

                    "rank_temperature",
                    "rank_ids",
                    "selection_probs",

                    "selected_avg_l2",
                    "min_avg_l2",
                    "max_avg_l2",
                    "mean_avg_l2",
                    "avg_l2",
                ])

            writer.writerow([
                self.episode_count,
                self.step_count,
                actions_stack.shape[0],
                metrics.get("tts_method", self.tts_method),
                metrics.get("selection_stage", ""),

                selected_idx,
                metrics.get("selected_rank", ""),
                metrics.get("selected_prob", ""),

                global_medoid_idx,
                global_medoid_avg_l2,

                metrics.get("unimodal", ""),
                metrics.get("s_score", ""),
                metrics.get("num_clusters", self.tts_num_clusters),
                self.tts_tau,
                self.tts_kmeans_iters,

                metrics.get("selected_cluster", ""),
                metrics.get("selected_cluster_size", ""),
                metrics.get("cluster_counts", ""),
                metrics.get("cluster_ids", ""),

                metrics.get("rank_temperature", ""),
                metrics.get("rank_ids", ""),
                metrics.get("selection_probs", ""),

                f"{selected_avg_l2:.8f}",
                f"{min_avg_l2:.8f}",
                f"{max_avg_l2:.8f}",
                f"{mean_avg_l2:.8f}",
                avg_list,
            ])

    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)


        # # Run inference
        # num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        # with torch.no_grad():
        #     predicted_frames, predicted_actions = self.model.inference_step(
        #         first_frame=current_frame,
        #         state=self.current_state,
        #         num_inference_steps=num_inference_steps,
        #         language_embeddings=t5_list,
        #         vlm_inputs=[vlm_inputs],
        #     )
        
        # tts add 
        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']

        if self.tts_enable and self.tts_num_samples > 1:
            action_candidates = []
            frame_candidates = []

            for sample_idx in range(self.tts_num_samples):
                cand_frames, cand_actions = self._run_single_inference(
                    current_frame=current_frame,
                    t5_list=t5_list,
                    vlm_inputs=vlm_inputs,
                    num_inference_steps=num_inference_steps,
                )

                action_candidates.append(cand_actions.squeeze(0).detach().float().cpu())
                frame_candidates.append(cand_frames.detach().float().cpu() if cand_frames is not None else None)

            # actions_stack = torch.stack(action_candidates, dim=0)
            # best_idx, pairwise_l2, avg_l2 = self._select_tts_medoid(actions_stack)
            # self._log_tts_result(actions_stack, pairwise_l2, avg_l2, best_idx)

            # predicted_actions = actions_stack[best_idx].unsqueeze(0)
            # predicted_frames = frame_candidates[best_idx]
            
            # actions_stack = torch.stack(action_candidates, dim=0)
            # best_idx, pairwise_l2, avg_l2, metrics = self._select_tts_action(actions_stack)
            # self._log_tts_result(actions_stack, pairwise_l2, avg_l2, best_idx, metrics)

            # predicted_actions = actions_stack[best_idx].unsqueeze(0)
            # predicted_frames = frame_candidates[best_idx]
            
            actions_stack = torch.stack(action_candidates, dim=0)
            selected_idx, pairwise_l2, avg_l2, metrics = self._select_tts_action(actions_stack)
            self._log_tts_result(actions_stack, pairwise_l2, avg_l2, selected_idx, metrics)

            predicted_actions = actions_stack[selected_idx].unsqueeze(0)
            predicted_frames = frame_candidates[selected_idx]
        else:
            predicted_frames, predicted_actions = self._run_single_inference(
                current_frame=current_frame,
                t5_list=t5_list,
                vlm_inputs=vlm_inputs,
                num_inference_steps=num_inference_steps,
            )
        # Run inference end

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                # condition_frame_viz = current_frame.squeeze(0)
                # predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                condition_frame_viz = current_frame.squeeze(0).detach().cpu()
                predicted_frames_viz = predicted_frames_viz.squeeze(0).detach().cpu()
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


def encode_obs(observation):
    """Post-Process Observation"""
    return observation


def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # # tts add
    # tts_enable = _as_bool(usr_args.get("tts_enable", DEFAULT_TTS_ENABLE))
    # tts_num_samples = int(usr_args.get("tts_num_samples", DEFAULT_TTS_NUM_SAMPLES))
    # tts_log_actions = _as_bool(usr_args.get("tts_log_actions", DEFAULT_TTS_LOG_ACTIONS))
    # tts_save_full_actions = _as_bool(usr_args.get("tts_save_full_actions", DEFAULT_TTS_SAVE_FULL_ACTIONS))
    
    # tts add
    tts_enable = _as_bool(usr_args.get("tts_enable", DEFAULT_TTS_ENABLE))
    tts_num_samples = int(usr_args.get("tts_num_samples", DEFAULT_TTS_NUM_SAMPLES))

    tts_method = str(usr_args.get("tts_method", DEFAULT_TTS_METHOD))
    tts_num_clusters = int(usr_args.get("tts_num_clusters", DEFAULT_TTS_NUM_CLUSTERS))
    tts_tau = float(usr_args.get("tts_tau", DEFAULT_TTS_TAU))
    tts_kmeans_iters = int(usr_args.get("tts_kmeans_iters", DEFAULT_TTS_KMEANS_ITERS))

    tts_log_actions = _as_bool(usr_args.get("tts_log_actions", DEFAULT_TTS_LOG_ACTIONS))

    # Deprecated/no-op; kept for old shell commands.
    tts_save_full_actions = _as_bool(
        usr_args.get("tts_save_full_actions", DEFAULT_TTS_SAVE_FULL_ACTIONS)
    )
    
    # policy = MotusPolicy(
    #     checkpoint_path=checkpoint_path,
    #     wan_path=wan_path,
    #     vlm_path=vlm_path,
    #     config_path=str(config_path),
    #     device=device,
    #     log_dir=usr_args.get('log_dir'),
    #     task_name=usr_args.get('task_name')
    # )
    
    # tts add
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
        # tts_enable=tts_enable,
        # tts_num_samples=tts_num_samples,
        # tts_log_actions=tts_log_actions,
        # tts_save_full_actions=tts_save_full_actions,
        tts_enable=tts_enable,
        tts_num_samples=tts_num_samples,
        tts_method=tts_method,
        tts_num_clusters=tts_num_clusters,
        tts_tau=tts_tau,
        tts_kmeans_iters=tts_kmeans_iters,
        tts_log_actions=tts_log_actions,
        tts_save_full_actions=tts_save_full_actions,        
    )
    
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')


def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")