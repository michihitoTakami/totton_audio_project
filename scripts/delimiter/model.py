from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class DelimiterDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class DelimiterWeights:
    config_json: Path
    state_dict_pth: Path


def load_weights_from_dir(weights_dir: Path, target: str = "all") -> DelimiterWeights:
    return DelimiterWeights(
        config_json=weights_dir / f"{target}.json",
        state_dict_pth=weights_dir / f"{target}.pth",
    )


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_delimiter_model(
    weights: DelimiterWeights,
    device: "Any",
    *,
    strict: bool = True,
) -> tuple["Any", dict[str, Any]]:
    """Build and load De-limiter model.

    This is a small vendored subset based on the upstream MIT-licensed repo:
    `https://github.com/jeonchangbin49/De-limiter`.

    Returns (model, config_dict).
    """

    try:
        import torch
        from asteroid_filterbanks import make_enc_dec
        from asteroid.masknn import TDConvNet

        # Optional but required by our wrapper model.
        from einops import rearrange
        from asteroid.models.base_models import (
            BaseEncoderMaskerDecoder,
            _shape_reconstructed,
            _unsqueeze_to_3d,
        )
        from asteroid.utils.torch_utils import jitable_shape, pad_x_to_y
    except Exception as e:  # pragma: no cover
        raise DelimiterDependencyError(
            "De-limiter PoC dependencies are missing. Install with: uv sync --extra delimiter"
        ) from e

    cfg = _load_json(weights.config_json)
    args = cfg.get("args", {})

    data_params = args.get("data_params", {})
    conv = args.get("conv_tasnet_params", {})
    model_loss = args.get("model_loss_params", {})

    sample_rate = int(data_params.get("sample_rate", 44100))
    nb_channels = int(data_params.get("nb_channels", 2))
    architecture = str(model_loss.get("architecture", "conv_tasnet_mask_on_output"))

    # --- Vendored minimal wrappers from upstream models/base_models.py ---
    # Keeping them local avoids importing the upstream utils/train code.

    class BaseEncoderMaskerDecoderWithConfigsMaskOnOutput(BaseEncoderMaskerDecoder):
        def __init__(
            self,
            encoder,
            masker,
            decoder,
            encoder_activation=None,
            **kwargs,
        ):
            super().__init__(encoder, masker, decoder, encoder_activation)
            self.use_encoder = kwargs.get("use_encoder", True)
            self.use_decoder = kwargs.get("use_decoder", True)
            self.nb_channels = kwargs.get("nb_channels", 2)
            decoder_activation = kwargs.get("decoder_activation", "sigmoid")
            if decoder_activation == "sigmoid":
                self.act_after_dec = torch.nn.Sigmoid()
            elif decoder_activation == "relu":
                self.act_after_dec = torch.nn.ReLU()
            elif decoder_activation == "relu6":
                self.act_after_dec = torch.nn.ReLU6()
            elif decoder_activation == "tanh":
                self.act_after_dec = torch.nn.Tanh()
            elif decoder_activation == "none":
                self.act_after_dec = torch.nn.Identity()
            else:
                self.act_after_dec = torch.nn.Sigmoid()

        def forward(self, wav):
            shape = jitable_shape(wav)
            wav = _unsqueeze_to_3d(wav)  # (batch, n_channels, time)

            tf_rep = self.forward_encoder(wav) if self.use_encoder else wav

            if self.nb_channels == 2:
                tf_rep = rearrange(tf_rep, "b c f t -> b (c f) t")

            est_masks = self.forward_masker(tf_rep)

            if self.use_decoder:
                if self.nb_channels == 2:
                    est_masks = rearrange(est_masks, "b 1 f t -> b f t")
                est_masks_decoded = self.forward_decoder(est_masks)
                est_masks_decoded = pad_x_to_y(
                    est_masks_decoded, wav
                )  # (batch, 1, time)
                est_masks_decoded = self.act_after_dec(est_masks_decoded)
                decoded = wav * est_masks_decoded

                return est_masks_decoded, _shape_reconstructed(decoded, shape)

            return (est_masks,)

    class BaseEncoderMaskerDecoderWithConfigsMultiChannelAsteroid(
        BaseEncoderMaskerDecoder
    ):
        def __init__(
            self,
            encoder,
            masker,
            decoder,
            encoder_activation=None,
            **kwargs,
        ):
            super().__init__(encoder, masker, decoder, encoder_activation)
            self.use_encoder = kwargs.get("use_encoder", True)
            self.apply_mask = kwargs.get("apply_mask", True)
            self.use_decoder = kwargs.get("use_decoder", True)
            self.nb_channels = kwargs.get("nb_channels", 2)
            decoder_activation = kwargs.get("decoder_activation", "none")
            if decoder_activation == "sigmoid":
                self.act_after_dec = torch.nn.Sigmoid()
            elif decoder_activation == "relu":
                self.act_after_dec = torch.nn.ReLU()
            elif decoder_activation == "relu6":
                self.act_after_dec = torch.nn.ReLU6()
            elif decoder_activation == "tanh":
                self.act_after_dec = torch.nn.Tanh()
            elif decoder_activation == "none":
                self.act_after_dec = torch.nn.Identity()
            else:
                self.act_after_dec = torch.nn.Sigmoid()

        def forward(self, wav):
            shape = jitable_shape(wav)
            wav = _unsqueeze_to_3d(wav)

            tf_rep = self.forward_encoder(wav) if self.use_encoder else wav

            if self.nb_channels == 2:
                tf_rep_flat = rearrange(tf_rep, "b c f t -> b (c f) t")
            else:
                tf_rep_flat = tf_rep

            est_masks = self.forward_masker(tf_rep_flat)

            if self.nb_channels == 2:
                tf_rep = rearrange(
                    tf_rep_flat, "b (c f) t -> b c f t", c=self.nb_channels
                )

            masked_tf_rep = est_masks * tf_rep if self.apply_mask else est_masks

            if self.use_decoder:
                decoded = self.forward_decoder(masked_tf_rep)
                decoded = pad_x_to_y(decoded, wav)
                decoded = self.act_after_dec(decoded)
                return masked_tf_rep, _shape_reconstructed(decoded, shape)

            return masked_tf_rep

    # --- Build model from config ---

    norm_type = conv.get("norm_type") or "gLN"
    encoder_activation = conv.get("encoder_activation", "relu")
    decoder_activation = conv.get("decoder_activation", "sigmoid")

    encoder, decoder = make_enc_dec(
        "free",
        n_filters=int(conv.get("n_filters", 512)),
        kernel_size=int(conv.get("kernel_size", 128)),
        stride=int(conv.get("stride", 64)),
        sample_rate=sample_rate,
    )

    in_chan = encoder.n_feats_out * nb_channels

    masker = TDConvNet(
        in_chan=in_chan,
        n_src=1
        if architecture == "conv_tasnet_mask_on_output"
        else int(conv.get("n_src", 1)),
        out_chan=encoder.n_feats_out,
        n_blocks=int(conv.get("n_blocks", 5)),
        n_repeats=int(conv.get("n_repeats", 2)),
        bn_chan=int(conv.get("bn_chan", 128)),
        hid_chan=int(conv.get("hid_chan", 512)),
        skip_chan=int(conv.get("skip_chan", 128)),
        norm_type=norm_type,
        mask_act=str(conv.get("mask_act", "relu")),
    )

    if architecture == "conv_tasnet_mask_on_output":
        model = BaseEncoderMaskerDecoderWithConfigsMaskOnOutput(
            encoder,
            masker,
            decoder,
            encoder_activation=encoder_activation,
            use_encoder=True,
            use_decoder=True,
            nb_channels=nb_channels,
            decoder_activation=decoder_activation,
        )
    elif architecture == "conv_tasnet":
        model = BaseEncoderMaskerDecoderWithConfigsMultiChannelAsteroid(
            encoder,
            masker,
            decoder,
            encoder_activation=encoder_activation,
            use_encoder=True,
            apply_mask=True,
            use_decoder=True,
            nb_channels=nb_channels,
            decoder_activation=decoder_activation,
        )
    else:
        raise ValueError(f"Unsupported De-limiter architecture: {architecture}")

    model = model.to(device)

    ckpt = torch.load(weights.state_dict_pth, map_location=device)
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

    return model, cfg


DelimiterBackend = Literal["delimiter", "bypass"]
