"""
Curriculum learning scheduler for spatial attention in WanVAE.
"""

import torch


class attnScheduler:
    """
    Schedules spatial attention window size during training.
    
    The curriculum starts with exact spatial correspondence (window=0) where each
    query token can only attend to the reference token at the same spatial position.
    It gradually expands to include nearby pixels, and finally enables full attention.
    
    Args:
        start_window (int): Initial window size (0 = only corresponding position)
        max_window (int): Maximum spatial window before going full attention
        warmup_steps (int): Number of steps to stay at start_window
        transition_steps (int): Number of steps to gradually expand from start to max
        full_attention_at_end (bool): Whether to enable full attention after transition
        enabled (bool): Whether curriculum is enabled (for easy on/off toggling)
    """
    
    def __init__(
        self,
        start_window=0,
        max_window=8,
        warmup_steps=1000,
        transition_steps=10000,
        full_attention_at_end=True,
        enabled=True,
    ):
        self.start_window = start_window
        self.max_window = max_window
        self.warmup_steps = warmup_steps
        self.transition_steps = transition_steps
        self.full_attention_at_end = full_attention_at_end
        self.enabled = enabled
        
        print(f"\n{'='*60}")
        print("CURRICULUM SCHEDULER INITIALIZED")
        print(f"{'='*60}")
        print(f"  Enabled:              {self.enabled}")
        print(f"  Start window:         {self.start_window}")
        print(f"  Max window:           {self.max_window}")
        print(f"  Warmup steps:         {self.warmup_steps:,}")
        print(f"  Transition steps:     {self.transition_steps:,}")
        print(f"  Full attention:       {self.full_attention_at_end}")
        print(f"  Total curriculum:     {self.warmup_steps + self.transition_steps:,} steps")
        print(f"{'='*60}\n")
    
    def get_window_size(self, step):
        """
        Get spatial window size for current training step.
        
        Args:
            step (int): Current training step
            
        Returns:
            int: Window size (0 = diagonal, -1 = full attention)
        """
        if not self.enabled:
            return -1  # Full attention when disabled
        
        # Phase 1: Warmup - stay at start window
        if step < self.warmup_steps:
            return self.start_window
        
        # Phase 2: Transition - gradually expand window
        progress = min(1.0, (step - self.warmup_steps) / self.transition_steps)
        
        # Phase 3: Full attention (if enabled)
        if self.full_attention_at_end and progress >= 1.0:
            return -1  # Full attention
        
        # Exponential growth feels more natural for spatial expansion
        # window = start + (max - start) * progress^2
        window = self.start_window + (self.max_window - self.start_window) * (progress ** 2)
        return int(window)
    
    def get_phase_info(self, step):
        """
        Get human-readable phase information for logging.
        
        Args:
            step (int): Current training step
            
        Returns:
            dict: Phase information
        """
        window = self.get_window_size(step)
        
        if not self.enabled:
            phase = "disabled"
            description = "Full attention (curriculum disabled)"
        elif step < self.warmup_steps:
            phase = "warmup"
            description = f"Warmup phase (window={window})"
        elif step < self.warmup_steps + self.transition_steps:
            phase = "transition"
            progress = (step - self.warmup_steps) / self.transition_steps * 100
            description = f"Transition phase (window={window}, {progress:.1f}% complete)"
        else:
            phase = "full_attention"
            description = "Full attention phase"
        
        return {
            "phase": phase,
            "window_size": window,
            "description": description,
            "step": step,
        }
    
    def update_model(self, model, step):
        """
        Update all attention blocks in the model with current window size.

        Args:
            model: The model (WanDecoderTransformer, AutoencoderKLWan, or wrapper)
            step (int): Current training step

        Returns:
            int: Current window size
        """
        window_size = self.get_window_size(step)

        # Handle wrapped model
        if hasattr(model, 'ae'):
            ae_model = model.ae
        else:
            ae_model = model

        # Update decoder attention blocks (for autoencoder models)
        if hasattr(ae_model, 'decoder') and hasattr(ae_model.decoder, 'up_attn_blocks'):
            for module in ae_model.decoder.up_attn_blocks:
                if hasattr(module, 'attention_window'):
                    module.attention_window = window_size

        # Update transformer blocks (for WanDecoderTransformer)
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
            for block in model.transformer.blocks:
                if hasattr(block, 'attention_window'):
                    block.attention_window = window_size

        return window_size


def visualize_spatial_mask(
    q_time=5, 
    k_time=1, 
    height=8, 
    width=8, 
    window_size=0,
    save_path=None
):
    """
    Visualize what the spatial attention mask looks like for debugging.
    
    Args:
        q_time: Number of query frames
        k_time: Number of reference frames
        height: Spatial height
        width: Spatial width
        window_size: Current window size
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    q_len = q_time * height * width
    k_len = k_time * height * width
    
    # Create mask
    mask = torch.zeros(q_len, k_len, dtype=torch.bool)
    
    for q_idx in range(q_len):
        q_w = q_idx % width
        q_h = (q_idx // width) % height
        q_t = q_idx // (width * height)
        
        for kv_idx in range(k_len):
            k_w = kv_idx % width
            k_h = (kv_idx // width) % height
            k_t = kv_idx // (width * height)
            
            h_dist = abs(q_h - k_h)
            w_dist = abs(q_w - k_w)
            spatial_dist = h_dist + w_dist
            
            mask[q_idx, kv_idx] = spatial_dist <= window_size
    
    # Calculate sparsity
    total_connections = q_len * k_len
    allowed_connections = mask.sum().item()
    sparsity = (1 - allowed_connections / total_connections) * 100
    
    # Create plot
    plt.figure(figsize=(12, 10))
    plt.imshow(mask.numpy(), cmap='Blues', interpolation='nearest', aspect='auto')
    plt.xlabel(f'Reference tokens (k_time={k_time}, h={height}, w={width})', fontsize=12)
    plt.ylabel(f'Query tokens (q_time={q_time}, h={height}, w={width})', fontsize=12)
    plt.title(f'Spatial Attention Mask (window_size={window_size})\n'
              f'Sparsity: {sparsity:.1f}% | Allowed connections: {allowed_connections}/{total_connections}',
              fontsize=14)
    
    cbar = plt.colorbar(label='Attention allowed')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Blocked', 'Allowed'])
    
    # Add grid lines to show frame boundaries
    for t in range(1, q_time):
        plt.axhline(y=t * height * width - 0.5, color='red', linestyle='--', 
                   linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention mask visualization to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test the scheduler
    print("Testing Curriculum Scheduler\n")
    
    scheduler = SpatialAttentionCurriculumScheduler(
        start_window=0,
        max_window=8,
        warmup_steps=1000,
        transition_steps=10000,
        full_attention_at_end=True,
        enabled=True,
    )
    
    # Test at various steps
    test_steps = [0, 500, 1000, 3000, 5000, 8000, 11000, 15000]
    
    print("Step-by-step progression:")
    print(f"{'Step':>8} | {'Window':>8} | {'Phase':>15} | Description")
    print("-" * 80)
    
    for step in test_steps:
        info = scheduler.get_phase_info(step)
        print(f"{step:>8,} | {info['window_size']:>8} | {info['phase']:>15} | {info['description']}")
    
    # Visualize masks
    print("\nGenerating mask visualizations...")
    for window in [0, 1, 2, 4, 8]:
        visualize_spatial_mask(
            q_time=5,
            k_time=1,
            height=8,
            width=8,
            window_size=window,
            save_path=f"mask_window_{window}.png"
        )
