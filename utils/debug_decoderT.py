import torch
import torch.nn.functional as F


class MinimalDecoderTracker:
    """Ultra-minimal version tracking only mean/std."""

    def __init__(self, decoder, transformer=None):
        self.decoder = self._resolve_decoder(decoder)
        self.transformer = self._resolve_transformer(transformer)
        self.stats = {}
        self.hooks = []
        self._register_hooks()

    def _resolve_decoder(self, decoder):
        """
        Accept decoder modules directly or indirect references like the parent
        autoencoder or its bound decode method.
        """
        if decoder is None:
            return None

        if hasattr(decoder, "mid_block") or hasattr(decoder, "up_blocks"):
            return decoder

        if hasattr(decoder, "decoder"):
            return decoder.decoder

        bound = getattr(decoder, "__self__", None)
        if bound is not None and hasattr(bound, "decoder"):
            return bound.decoder

        return decoder

    def _resolve_transformer(self, transformer):
        """Mirror decoder resolution for the optional transformer component."""
        if transformer is None:
            return None

        if hasattr(transformer, "transformer_blocks") or hasattr(transformer, "blocks"):
            return transformer

        if hasattr(transformer, "transformer"):
            inner = transformer.transformer
            if hasattr(inner, "transformer_blocks") or hasattr(inner, "blocks"):
                return inner

        bound = getattr(transformer, "__self__", None)
        if bound is not None:
            return self._resolve_transformer(bound)

        return transformer

    def _iter_transformer_blocks(self):
        transformer = self.transformer
        if transformer is None:
            return []

        blocks = None
        if hasattr(transformer, "transformer_blocks"):
            blocks = transformer.transformer_blocks
        elif hasattr(transformer, "blocks"):
            blocks = transformer.blocks

        if blocks is None:
            return []

        if isinstance(blocks, dict):
            return list(blocks.items())
        return list(enumerate(blocks))

    def _register_hooks(self):
        layers_to_track = []

        if self.decoder is not None:
            if hasattr(self.decoder, "mid_block"):
                layers_to_track.append(("mid_block", self.decoder.mid_block))
            if hasattr(self.decoder, "conv_out"):
                layers_to_track.append(("conv_out", self.decoder.conv_out))

            if hasattr(self.decoder, "up_blocks"):
                for i, block in enumerate(self.decoder.up_blocks):
                    layers_to_track.append((f"up_block_{i}", block))

            # Add attention blocks if present
            if hasattr(self.decoder, "up_attn_blocks"):
                for i, block in enumerate(self.decoder.up_attn_blocks):
                    layers_to_track.append((f"attn_{i}", block))

        for identifier, block in self._iter_transformer_blocks():
            name = identifier if isinstance(identifier, str) else f"transformer_{identifier}"
            layers_to_track.append((name, block))

        for name, layer in layers_to_track:
            if layer is None:
                continue
            self.hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def _extract_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    tensor = _extract_tensor(item)
                    if tensor is not None:
                        return tensor
                return None
            sample = getattr(obj, "sample", None)
            if isinstance(sample, torch.Tensor):
                return sample
            return None

        def hook(module, input, output):
            with torch.no_grad():
                tensor = _extract_tensor(output)
                if tensor is None:
                    return
                self.stats[f"{name}_mean"] = tensor.mean().item()
                self.stats[f"{name}_std"] = tensor.std().item()

        return hook

    def get_stats(self):
        return self.stats.copy()

    def clear(self):
        self.stats = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

class MinimalDecoderTrackerWanModel:
    """Ultra-minimal version tracking only mean/std."""

    def __init__(self, decoder, transformer=None):
        self.decoder = self._resolve_decoder(decoder)
        self.transformer = self._resolve_transformer(transformer)
        self.stats = {}
        self.hooks = []
        self._register_hooks()

    def _resolve_decoder(self, decoder):
        """
        Accept decoder modules directly or indirect references like the parent
        autoencoder or its bound decode method.
        """
        if decoder is None:
            return None

        if hasattr(decoder, "mid_block") or hasattr(decoder, "up_blocks"):
            return decoder

        if hasattr(decoder, "decoder"):
            return decoder.decoder

        bound = getattr(decoder, "__self__", None)
        if bound is not None and hasattr(bound, "decoder"):
            return bound.decoder

        return decoder

    def _resolve_transformer(self, transformer):
        """
        Resolve transformer, handling the new WanDecoderTransformer -> WanModel structure.
        
        New structure:
        - WanDecoderTransformer has .transformer (which is WanModel or PEFT-wrapped WanModel)
        - WanModel has .blocks (ModuleList of WanAttentionBlock)
        """
        if transformer is None:
            return None

        # Check if it's already a WanModel (has .blocks directly)
        if hasattr(transformer, "blocks") and not hasattr(transformer, "transformer"):
            return transformer

        # Check if it's WanDecoderTransformer (has .transformer attribute)
        if hasattr(transformer, "transformer"):
            inner = transformer.transformer
            # The inner transformer is WanModel (or PEFT-wrapped WanModel)
            if hasattr(inner, "blocks") or hasattr(inner, "base_model"):
                return inner

        # Legacy support
        if hasattr(transformer, "transformer_blocks"):
            return transformer

        bound = getattr(transformer, "__self__", None)
        if bound is not None:
            return self._resolve_transformer(bound)

        return transformer

    def _iter_transformer_blocks(self):
        """
        Iterate through transformer blocks, handling PEFT wrapping.
        """
        transformer = self.transformer
        if transformer is None:
            return []

        blocks = None
        
        # Try to get blocks directly
        if hasattr(transformer, "blocks"):
            blocks = transformer.blocks
            
            # If blocks is PEFT-wrapped, access underlying ModuleList
            if hasattr(blocks, "base_model"):
                try:
                    blocks = blocks.base_model.model
                except AttributeError:
                    pass
        
        # Legacy support
        elif hasattr(transformer, "transformer_blocks"):
            blocks = transformer.transformer_blocks

        if blocks is None:
            return []

        if isinstance(blocks, dict):
            return list(blocks.items())
        return list(enumerate(blocks))

    def _register_hooks(self):
        layers_to_track = []

        # Track decoder layers
        if self.decoder is not None:
            if hasattr(self.decoder, "mid_block"):
                layers_to_track.append(("mid_block", self.decoder.mid_block))
            if hasattr(self.decoder, "conv_out"):
                layers_to_track.append(("conv_out", self.decoder.conv_out))

            if hasattr(self.decoder, "up_blocks"):
                for i, block in enumerate(self.decoder.up_blocks):
                    layers_to_track.append((f"up_block_{i}", block))

        # Track transformer blocks
        for identifier, block in self._iter_transformer_blocks():
            name = identifier if isinstance(identifier, str) else f"transformer_{identifier}"
            layers_to_track.append((name, block))

        # Register hooks
        for name, layer in layers_to_track:
            if layer is None:
                continue
            self.hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def _extract_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    tensor = _extract_tensor(item)
                    if tensor is not None:
                        return tensor
                return None
            sample = getattr(obj, "sample", None)
            if isinstance(sample, torch.Tensor):
                return sample
            return None

        def hook(module, input, output):
            with torch.no_grad():
                tensor = _extract_tensor(output)
                if tensor is None:
                    return
                self.stats[f"{name}_mean"] = tensor.mean().item()
                self.stats[f"{name}_std"] = tensor.std().item()

        return hook

    def get_stats(self):
        """Return a copy of current statistics."""
        return self.stats.copy()

    def clear(self):
        """Clear collected statistics."""
        self.stats = {}

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class MinimalVideoVaePlusRefTracker:
    """
    Tracker for AutoencoderKL2plus1D_1dcnn (videovaeplus_ref).

    Tracks mean/std of outputs from:
      - decoder: mid block, up blocks, conv_out
      - transformer (WanDecoderTransformer → WanModel): per-block
      - ref_conv_in: patch_embedding, proj
    """

    def __init__(self, model):
        self.decoder = self._resolve_decoder(model)
        self.transformer = self._resolve_transformer(model)
        self.ref_conv_in = self._resolve_ref_conv_in(model)
        self.stats = {}
        self.hooks = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_decoder(self, model):
        """Return the Decoder2plus1D instance."""
        if model is None:
            return None
        # Full model
        if hasattr(model, "decoder"):
            return model.decoder
        # Already a Decoder2plus1D (has .mid and .up)
        if hasattr(model, "mid") and hasattr(model, "up"):
            return model
        return None

    def _resolve_transformer(self, model):
        """
        Return the inner WanModel (has .blocks) from WanDecoderTransformer.
        Handles PEFT wrapping.
        """
        if model is None:
            return None

        candidate = None

        # Full AutoencoderKL2plus1D_1dcnn
        if hasattr(model, "transformer"):
            candidate = model.transformer

        # Already WanDecoderTransformer or WanModel
        if candidate is None and (hasattr(model, "blocks") or hasattr(model, "transformer")):
            candidate = model

        if candidate is None:
            return None

        # Unwrap WanDecoderTransformer → WanModel
        if hasattr(candidate, "transformer"):
            inner = candidate.transformer
            # Handle PEFT base_model wrapping
            if hasattr(inner, "base_model"):
                try:
                    inner = inner.base_model.model
                except AttributeError:
                    pass
            candidate = inner

        # Handle PEFT on the WanModel itself
        if hasattr(candidate, "base_model"):
            try:
                candidate = candidate.base_model.model
            except AttributeError:
                pass

        return candidate if hasattr(candidate, "blocks") else None

    def _resolve_ref_conv_in(self, model):
        """Return the RefConvIn instance."""
        if model is None:
            return None
        if hasattr(model, "ref_conv_in"):
            return model.ref_conv_in
        # Passed directly
        if hasattr(model, "patch_embedding") and hasattr(model, "proj"):
            return model
        return None

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        layers_to_track = []

        # ---- Decoder ----
        dec = self.decoder
        if dec is not None:
            # mid block (Decoder2plus1D uses .mid, not .mid_block)
            if hasattr(dec, "mid"):
                layers_to_track.append(("dec_mid", dec.mid))
            # up blocks (.up is a ModuleList in Decoder2plus1D)
            if hasattr(dec, "up"):
                for i, block in enumerate(dec.up):
                    layers_to_track.append((f"dec_up_{i}", block))
            if hasattr(dec, "conv_out"):
                layers_to_track.append(("dec_conv_out", dec.conv_out))

        # ---- Transformer blocks ----
        tx = self.transformer
        if tx is not None and hasattr(tx, "blocks"):
            blocks = tx.blocks
            # Unwrap PEFT ModuleList wrapping if needed
            if hasattr(blocks, "base_model"):
                try:
                    blocks = blocks.base_model.model
                except AttributeError:
                    pass
            if isinstance(blocks, dict):
                items = list(blocks.items())
            else:
                items = list(enumerate(blocks))
            for identifier, block in items:
                name = identifier if isinstance(identifier, str) else f"tx_{identifier}"
                layers_to_track.append((name, block))

        # ---- ref_conv_in ----
        ref = self.ref_conv_in
        if ref is not None:
            if hasattr(ref, "patch_embedding"):
                layers_to_track.append(("ref_patch_embedding", ref.patch_embedding))
            if hasattr(ref, "proj"):
                layers_to_track.append(("ref_proj", ref.proj))

        for name, layer in layers_to_track:
            if layer is None:
                continue
            self.hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def _extract_tensor(obj):
            if isinstance(obj, torch.Tensor):
                return obj
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    t = _extract_tensor(item)
                    if t is not None:
                        return t
                return None
            sample = getattr(obj, "sample", None)
            if isinstance(sample, torch.Tensor):
                return sample
            return None

        def hook(module, input, output):
            with torch.no_grad():
                tensor = _extract_tensor(output)
                if tensor is None:
                    return
                self.stats[f"{name}_mean"] = tensor.mean().item()
                self.stats[f"{name}_std"] = tensor.std().item()

        return hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self):
        return self.stats.copy()

    def clear(self):
        self.stats = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def __del__(self):
        self.remove_hooks()