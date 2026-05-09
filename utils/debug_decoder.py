import torch
import torch.nn.functional as F

class MinimalDecoderTracker:
    """Ultra-minimal version tracking only mean/std."""
    
    def __init__(self, decoder):
        self.decoder = decoder
        self.stats = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        layers_to_track = [
            ('conv_in', self.decoder.conv_in),
            ('mid_block', self.decoder.mid_block),
            ('conv_out', self.decoder.conv_out),
        ]
        
        # Add up_blocks
        for i, block in enumerate(self.decoder.up_blocks):
            layers_to_track.append((f'up_block_{i}', block))
        
        # Add attention blocks if present
        if hasattr(self.decoder, 'up_attn_blocks'):
            for i, block in enumerate(self.decoder.up_attn_blocks):
                layers_to_track.append((f'attn_{i}', block))
        
        for name, layer in layers_to_track:
            self.hooks.append(
                layer.register_forward_hook(self._make_hook(name))
            )
    
    def _make_hook(self, name):
        def hook(module, input, output):
            with torch.no_grad():
                self.stats[f"{name}_mean"] = output.mean().item()
                self.stats[f"{name}_std"] = output.std().item()
        return hook
    
    def get_stats(self):
        return self.stats.copy()
    
    def clear(self):
        self.stats = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()