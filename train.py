import torch
import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloading.efficient_data_loader_v2 import GDPDataset_v2
from evaluation.dose_metric_online import dose_MAE_metric, cropped2ori
from modeling.mednext_d2 import create_mednext_d2
import numpy as np
import argparse
from os.path import join
import random
import yaml
import os


class L_Trainer(object):

    def __init__(self, num_gpu=1, precision="32"):
        self.devices = num_gpu

        if precision == "16":
            self.precision = "bf16-mixed"
            print(f"Pytorch Lightning Fabric using {self.precision} precision")
        elif precision == "32":
            self.precision = "32"
        else:
            raise ValueError("Precision must be 16 or 32")

        self.fabric = L.Fabric(
            accelerator="gpu",
            strategy="auto",
            devices=self.devices,
            num_nodes=1,
            precision=self.precision,
        )

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_epoch_time = 0

    def fit(
        self,
        model: torch.nn.Module,
        cfig: dict = None,
        input_type: str = None,
        optimizer: list = None,
        lr_scheduler: list = None,
        train_loader: list = None,
        val_loader: list = None,
        criterion=None,
        max_epoch=None,
        save_checkpoint_folder=None,
    ):
        self.cfig = cfig
        self.input_type = input_type
        self.max_epoch = max_epoch
        self.save_checkpoint_folder = save_checkpoint_folder

        assert len(train_loader) == 2
        assert len(val_loader) == 2
        assert len(optimizer) == 2
        assert len(lr_scheduler) == 2

        self.train_HaN_loader = self.fabric.setup_dataloaders(
            train_loader[0], move_to_device=True
        )
        self.train_Lung_loader = self.fabric.setup_dataloaders(
            train_loader[1], move_to_device=True
        )
        self.val_HaN_loader = self.fabric.setup_dataloaders(
            val_loader[0], move_to_device=True, use_distributed_sampler=False
        )
        self.val_Lung_loader = self.fabric.setup_dataloaders(
            val_loader[1], move_to_device=True, use_distributed_sampler=False
        )

        self.HaN_optimizer = optimizer[0]
        self.Lung_optimizer = optimizer[1]
        self.HaN_lr_scheduler = lr_scheduler[0]
        self.Lung_lr_scheduler = lr_scheduler[1]

        self.best_dose_score = 1e9
        self.best_dose_score_epoch = 0

        if not os.path.exists(self.save_checkpoint_folder):
            os.makedirs(self.save_checkpoint_folder, exist_ok=True)
            print(f"Created checkpoint folder: {self.save_checkpoint_folder}")

        self.model, self.HaN_optimizer, self.Lung_optimizer = self.fabric.setup(
            model, self.HaN_optimizer, self.Lung_optimizer
        )
        if self.cfig.get("torch_compile", False):
            print("Compiling model with torch.compile()...")
            if hasattr(self.model, "module"):
                # model.module is DDP/FSDP wrapped inner model
                self.model.module = torch.compile(self.model.module)
            else:
                self.model = torch.compile(self.model)
        self.criterion = criterion

        for i in range(self.current_epoch, self.max_epoch):
            self.train_loop()
            self.val_loop()

    def save_best_checkpoint(self, path):
        # Save state_dicts: more portable and re-compile on load if needed.
        module = self.model.module if hasattr(self.model, "module") else self.model
        state = {
            "model_state_dict": module.state_dict(),
            "HaN_optimizer_state": self.HaN_optimizer.state_dict(),
            "Lung_optimizer_state": self.Lung_optimizer.state_dict(),
            "epoch": self.current_epoch,
        }
        self.fabric.save(path, state)

    def train_loop(self):
        self.model.train()
        train_loss = []
        epoch_total_steps = len(self.train_HaN_loader) * 2
        HaN_iter = iter(self.train_HaN_loader)
        Lung_iter = iter(self.train_Lung_loader)

        for step in tqdm(range(epoch_total_steps), desc=f"Epoch {self.current_epoch}"):
            if step % 2 == 0:
                try:
                    batch_data = next(HaN_iter)
                except StopIteration:
                    HaN_iter = iter(self.train_HaN_loader)
                    batch_data = next(HaN_iter)

                with self.fabric.autocast():
                    outputs = self.model(batch_data["data"], task="HaN")
                    if self.cfig["act_sig"]:
                        outputs = torch.sigmoid(outputs)
                    outputs = outputs * self.cfig["scale_out"]
                    loss = self.criterion(outputs, batch_data["label"])
                self.HaN_optimizer.zero_grad()
                self.fabric.backward(loss)
                self.HaN_optimizer.step()
                if self.HaN_lr_scheduler:
                    self.HaN_lr_scheduler.step()
            else:
                try:
                    batch_data = next(Lung_iter)
                except StopIteration:
                    Lung_iter = iter(self.train_Lung_loader)
                    batch_data = next(Lung_iter)

                with self.fabric.autocast():
                    outputs = self.model(batch_data["data"], task="Lung")
                    if self.cfig["act_sig"]:
                        outputs = torch.sigmoid(outputs)
                    outputs = outputs * self.cfig["scale_out"]
                    loss = self.criterion(outputs, batch_data["label"])
                self.Lung_optimizer.zero_grad()
                self.fabric.backward(loss)
                self.Lung_optimizer.step()
                if self.Lung_lr_scheduler:
                    self.Lung_lr_scheduler.step()

            train_loss.append(loss.item())
            if (step + 1) % 20 == 0:
                torch.cuda.empty_cache()

        self.current_epoch += 1
        print(f"Epoch: {self.current_epoch}, tr_loss: {np.mean(train_loss):.4f}")

    def val_loop(self):
        val_current_epoch = False
        if self.current_epoch <= 100:
            if self.current_epoch % 10 == 0:
                val_current_epoch = True
        else:
            if self.current_epoch % 4 == 0:
                val_current_epoch = True

        if not val_current_epoch:
            return

        self.model.eval()
        dose_score_list = []
        Lung_score_list = []
        HaN_score_list = []
        step = 0

        with torch.no_grad():
            for batch_data in self.val_Lung_loader:
                outputs = self.model(batch_data["data"], task="Lung")
                if self.cfig["act_sig"]:
                    outputs = torch.sigmoid(outputs)
                outputs = outputs * self.cfig["scale_out"]
                for index in range(len(outputs)):
                    crop_data = outputs[index][0].cpu().numpy()
                    ori_size = batch_data["ori_img_size"][index].cpu().numpy().tolist()
                    isocenter = (
                        batch_data["ori_isocenter"][index].cpu().numpy().tolist()
                    )
                    pred2orisize = (
                        cropped2ori(
                            crop_data,
                            ori_size,
                            isocenter,
                            self.cfig["loader_params"]["in_size"],
                        )
                        * self.cfig["loader_params"]["dose_div_factor"]
                    )
                    score = dose_MAE_metric(
                        pred_dose=pred2orisize, pred_id=batch_data["id"][index]
                    )
                    dose_score_list.append(score)
                    Lung_score_list.append(score)
                step += 1
                if step % 20 == 0:
                    torch.cuda.empty_cache()

            for batch_data in self.val_HaN_loader:
                outputs = self.model(batch_data["data"], task="HaN")
                if self.cfig["act_sig"]:
                    outputs = torch.sigmoid(outputs)
                outputs = outputs * self.cfig["scale_out"]
                for index in range(len(outputs)):
                    crop_data = outputs[index][0].cpu().numpy()
                    ori_size = batch_data["ori_img_size"][index].cpu().numpy().tolist()
                    isocenter = (
                        batch_data["ori_isocenter"][index].cpu().numpy().tolist()
                    )
                    pred2orisize = (
                        cropped2ori(
                            crop_data,
                            ori_size,
                            isocenter,
                            self.cfig["loader_params"]["in_size"],
                        )
                        * self.cfig["loader_params"]["dose_div_factor"]
                    )
                    score = dose_MAE_metric(
                        pred_dose=pred2orisize, pred_id=batch_data["id"][index]
                    )
                    dose_score_list.append(score)
                    HaN_score_list.append(score)
                step += 1
                if step % 20 == 0:
                    torch.cuda.empty_cache()

        mean_dose_score = np.mean(dose_score_list)
        mean_Lung_score = np.mean(Lung_score_list)
        mean_HaN_score = np.mean(HaN_score_list)

        if mean_dose_score < self.best_dose_score:
            self.best_dose_score = mean_dose_score
            self.best_dose_score_epoch = self.current_epoch
            if self.best_dose_score < 2.30 and self.current_epoch > 0:
                self.save_best_checkpoint(
                    join(self.save_checkpoint_folder, f"{self.current_epoch}.pth")
                )
        else:
            if (
                self.current_epoch - self.best_dose_score_epoch >= 50
                or self.current_epoch == self.max_epoch
            ):
                print("Early stopping or training complete")
                self.save_best_checkpoint(
                    join(self.save_checkpoint_folder, "latest.pth")
                )

        print(
            f"Epoch: {self.current_epoch}, dose_score: {mean_dose_score:.4f}, Lung: {mean_Lung_score:.4f}, HaN: {mean_HaN_score:.4f}, Best: {self.best_dose_score:.4f} @ epoch {self.best_dose_score_epoch}"
        )


if __name__ == "__main__":
    
    torch.set_float32_matmul_precision("medium")
    random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    np.random.seed(2025)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project", type=str, required=True)
    parser.add_argument("-c", required=True, type=str)
    parser.add_argument("--fp", type=str, default="32")
    args = parser.parse_args()

    exp_name = args.c.split(".yaml")[0] + "/" + args.project
    print("exp_name", exp_name)
    cfig = yaml.load(open(join("config_files", args.c)), Loader=yaml.FullLoader)


    save_checkpoint_folder = os.path.join(
        os.environ.get("Result", "./results"), exp_name
    )
    print("save_checkpoint_folder", save_checkpoint_folder)

    if cfig["loader_params"]["dataloader"] == "v2":
        train_HaN_dataset = GDPDataset_v2(
            cfig["loader_params"], phase="train", site="HaN", debug=False
        )
        val_HaN_dataset = GDPDataset_v2(
            cfig["loader_params"], phase="val", site="HaN", debug=False
        )
        train_Lung_dataset = GDPDataset_v2(
            cfig["loader_params"], phase="train", site="Lung", debug=False
        )
        val_Lung_dataset = GDPDataset_v2(
            cfig["loader_params"], phase="val", site="Lung", debug=False
        )
    else:
        raise NotImplementedError(
            f"Data loader {cfig['loader_params']['dataloader']} is not implemented"
        )

    train_HaN_loader = DataLoader(
        train_HaN_dataset,
        batch_size=cfig["loader_params"]["train_bs"],
        shuffle=True,
        num_workers=cfig["loader_params"]["num_workers"],
        pin_memory=True,
        prefetch_factor=cfig["loader_params"].get("prefetch_factor", 2),
        persistent_workers=True,
    )

    val_HaN_loader = DataLoader(
        val_HaN_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True
    )
    train_Lung_loader = DataLoader(
        train_Lung_dataset,
        batch_size=cfig["loader_params"]["train_bs"],
        shuffle=True,
        num_workers=cfig["loader_params"]["num_workers"],
        pin_memory=True,
        prefetch_factor=cfig["loader_params"].get("prefetch_factor", 2),
        persistent_workers=True,
    )
    val_Lung_loader = DataLoader(
        val_Lung_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True
    )

    if cfig["loss"] == "l1":
        criterion = torch.nn.L1Loss()

    else:
        raise NotImplementedError(f"Loss {cfig['loss']} is not implemented")

    total_steps = len(train_HaN_loader) * cfig["num_epochs"] * 2
    print("total_steps", total_steps)

    model, HaN_optimizer, Lung_optimizer = create_mednext_d2(
        spatial_dims=3, in_channels=7, lr=cfig["lr"]
    )

    HaN_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        HaN_optimizer, T_max=total_steps, last_epoch=-1
    )
    Lung_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        Lung_optimizer, T_max=total_steps, last_epoch=-1
    )

    trainer = L_Trainer(num_gpu=1, precision=args.fp)
    trainer.fit(
        model=model,
        cfig=cfig,
        input_type=None,
        optimizer=[HaN_optimizer, Lung_optimizer],
        lr_scheduler=[HaN_lr_scheduler, Lung_lr_scheduler],
        train_loader=[train_HaN_loader, train_Lung_loader],
        val_loader=[val_HaN_loader, val_Lung_loader],
        criterion=criterion,
        save_checkpoint_folder=save_checkpoint_folder,
        max_epoch=cfig["num_epochs"],
    )
