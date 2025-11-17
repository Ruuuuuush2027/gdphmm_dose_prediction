import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from modeling.mednext import MedNeXt, MedNeXtOutBlock, MedNeXtUpBlock, MedNeXtBlock

class MedNeXt_D2(MedNeXt):
    def __init__(self, **kwargs):
        # Force init_filters = 16
        kwargs["init_filters"] = 16
        super().__init__(**kwargs)

        spatial_dims = kwargs.get("spatial_dims", 3)
        spatial_dims_str = f"{spatial_dims}d"
        init_filters = kwargs["init_filters"]

        # Define a full independent decoder for HaN
        self.han_up_blocks = nn.ModuleList([
            MedNeXtUpBlock(
                in_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i)),
                out_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i - 1)),
                expansion_ratio=kwargs['decoder_expansion_ratio'][i],
                kernel_size=kwargs['kernel_size'],
                use_residual_connection=True,
                norm_type="group",
                dim=spatial_dims_str
            )
            for i in range(len(kwargs['blocks_up']))
        ])

        self.han_dec_stages = nn.ModuleList([
            nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i - 1)),
                        out_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i - 1)),
                        expansion_ratio=kwargs['decoder_expansion_ratio'][i],
                        kernel_size=kwargs['kernel_size'],
                        use_residual_connection=True,
                        norm_type="group",
                        dim=spatial_dims_str
                    )
                    for _ in range(kwargs['blocks_up'][i])
                ]
            )
            for i in range(len(kwargs['blocks_up']))
        ])

        # Define a full independent decoder for Lung
        self.lung_up_blocks = nn.ModuleList([
            MedNeXtUpBlock(
                in_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i)),
                out_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i - 1)),
                expansion_ratio=kwargs['decoder_expansion_ratio'][i],
                kernel_size=kwargs['kernel_size'],
                use_residual_connection=True,
                norm_type="group",
                dim=spatial_dims_str
            )
            for i in range(len(kwargs['blocks_up']))
        ])

        self.lung_dec_stages = nn.ModuleList([
            nn.Sequential(
                *[
                    MedNeXtBlock(
                        in_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i - 1)),
                        out_channels=init_filters * (2 ** (len(kwargs['blocks_up']) - i - 1)),
                        expansion_ratio=kwargs['decoder_expansion_ratio'][i],
                        kernel_size=kwargs['kernel_size'],
                        use_residual_connection=True,
                        norm_type="group",
                        dim=spatial_dims_str
                    )
                    for _ in range(kwargs['blocks_up'][i])
                ]
            )
            for i in range(len(kwargs['blocks_up']))
        ])

        # Define two independent output heads
        self.out_han = MedNeXtOutBlock(
            in_channels=init_filters,
            n_classes=1,
            dim=spatial_dims_str
        )
        self.out_lung = MedNeXtOutBlock(
            in_channels=init_filters,
            n_classes=1,
            dim=spatial_dims_str
        )

    def forward(self, x, task: str):
        # Stem + Encoder
        x = self.stem(x)
        enc_outputs = []
        for enc_stage, down_block in zip(self.enc_stages, self.down_blocks):
            x = enc_stage(x)
            enc_outputs.append(x)
            x = down_block(x)

        x = self.bottleneck(x)

        # Decoder branch according to task
        if task == "HaN":
            for up_block, dec_stage, enc_output in zip(self.han_up_blocks, self.han_dec_stages, reversed(enc_outputs)):
                x = up_block(x)
                x = x + enc_output
                x = dec_stage(x)
            x = self.out_han(x)

        elif task == "Lung":
            for up_block, dec_stage, enc_output in zip(self.lung_up_blocks, self.lung_dec_stages, reversed(enc_outputs)):
                x = up_block(x)
                x = x + enc_output
                x = dec_stage(x)
            x = self.out_lung(x)

        else:
            raise ValueError(f"Unknown task: {task}")

        return x


def create_mednext_d2(spatial_dims=3, in_channels=7, lr=1e-4, including_optimizer=True):
    model = MedNeXt_D2(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=1,  # Although output heads are separate, the base model uses out_channels=2
        kernel_size=3,
        deep_supervision=False,
        init_filters=16,  #! Force smaller init filters
        encoder_expansion_ratio=(2, 3, 4, 4),
        decoder_expansion_ratio=(4, 4, 3, 2),
        bottleneck_expansion_ratio=4,
        blocks_down=(2, 2, 2, 2),
        blocks_bottleneck=2,
        blocks_up=(2, 2, 2, 2),
        use_residual_connection=True,
        norm_type="group",
        global_resp_norm=False
    )

    if including_optimizer:
     
        # Define parameter groups for HaN and Lung
        shared_params = list(model.stem.parameters()) + \
                        list(model.enc_stages.parameters()) + \
                        list(model.down_blocks.parameters()) + \
                        list(model.bottleneck.parameters())

        optimizer_han = torch.optim.AdamW(
            [
                {"params": shared_params, "initial_lr": lr},
                {"params": model.han_up_blocks.parameters(), "initial_lr": lr},
                {"params": model.han_dec_stages.parameters(), "initial_lr": lr},
                {"params": model.out_han.parameters(), "initial_lr": lr}
            ],
            lr=lr,
            weight_decay=1e-4
        )

        optimizer_lung = torch.optim.AdamW(
            [
                {"params": shared_params, "initial_lr": lr},
                {"params": model.lung_up_blocks.parameters(), "initial_lr": lr},
                {"params": model.lung_dec_stages.parameters(), "initial_lr": lr},
                {"params": model.out_lung.parameters(), "initial_lr": lr}
            ],
            lr=lr,
            weight_decay=1e-4
        )

        return model, optimizer_han, optimizer_lung

    else:
        return model

if __name__ == "__main__":
    
    # model, optimizer_han, optimizer_lung = create_mednext_d2()
    model = create_mednext_d2(including_optimizer=False)
    print(model)

