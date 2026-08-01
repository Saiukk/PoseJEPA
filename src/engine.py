from __future__ import annotations

import logging
import os
import time

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _safe_len(x):
    """Safe len() for IterableDataset/DataLoader without __len__."""
    try:
        return len(x)
    except TypeError:
        return None


def _output_dir() -> str:
    """
    Current run's output dir, read from Hydra instead of os.getcwd():
    the latter depends on hydra.job.chdir (may be off depending on
    version/config), while HydraConfig.get().runtime.output_dir is
    always correct regardless of cwd changes.
    """
    try:
        return HydraConfig.get().runtime.output_dir
    except ValueError:
        # HydraConfig not initialized (e.g. class used outside @hydra.main)
        return os.getcwd()


class Engine:
    """ The brain  """

    def __init__(self, cfg: DictConfig, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
        self.device = torch.device(cfg.get("device", "cuda:0"))

        # Single source of truth for which components are trainable,
        # computed once and reused everywhere (build, optimizer, train mode).
        self.trainable_components = list(cfg.train.get("trainable_components", ["predictor"]))

        self.models = self._build_models()
        self.criterion = nn.L1Loss()
        self.loader = self._build_dataloader(split="train")

        # only set if is_train=True, otherwise stay None
        self.optimizer = None
        self.epochs = None
        self.log_every = None
        self.ckpt_every = None
        self.val_every = None
        self.start_epoch = 0
        self.val_loader = None
        self.best_val_loss = float("inf")
        self.wandb_run = None
        self._resumed_ckpt = None

        if self.is_train:
            self.val_loader = self._build_dataloader(split="val")
            self._setup_training()
            self._setup_wandb()
        else:
            checkpoint = self.cfg.get("checkpoint", None)
            if checkpoint:
                self.load_checkpoint(checkpoint, strict=True)
            for m in self.models.values():
                m.eval()

    # ------------------------------------------------------------------ #
    # Build
    # ------------------------------------------------------------------ #
    def _build_models(self) -> dict[str, nn.Module]:
        encoder2d = hydra.utils.instantiate(self.cfg.encoder2d).to(self.device)
        encoder3d = hydra.utils.instantiate(self.cfg.encoder3d).to(self.device)
        predictor = hydra.utils.instantiate(
            self.cfg.Predictor,
            context_dim=encoder3d.output_dim[-1],
            output_dim=encoder2d.output_dim[-1],
        ).to(self.device)
        models = {"encoder2d": encoder2d, "encoder3d": encoder3d, "predictor": predictor}

        for name, module in models.items():
            is_trainable = name in self.trainable_components
            for p in module.parameters():
                p.requires_grad_(is_trainable)
            module.train(is_trainable)
            log.info("Model '%s' -> trainable=%s", name, is_trainable)

        return models

    def _build_dataloader(self, split: str = "train"):
        dataset = hydra.utils.instantiate(
            self.cfg.data, dataset_source=self.cfg.gso.root, split=split
        )

        n_shards = len(getattr(dataset, "shard_paths", []))
        if n_shards == 0:
            if split == "val":
                log.warning("no shard assigned for validation")
                return None
            raise ValueError(f"No shard found for split='{split}'")

        # WebDataset splits shards across workers: more workers than shards
        # leaves some workers empty and crashes. Cap it.
        num_workers = min(self.cfg.train.loader.num_workers, n_shards)
        if num_workers != self.cfg.train.loader.num_workers:
            log.warning(
                f"num_workers={self.cfg.train.loader.num_workers} reduced to {num_workers} "
                f"because only {n_shards} shards available for split='{split}'"
            )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.train.loader.batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        log.info("[%s] Dataloader: %d shard, %d worker", split, n_shards, num_workers)
        return loader

    def _setup_training(self):
        trainer_cfg = self.cfg.train.get("trainer", {})

        params = []
        for name in self.trainable_components:
            params += list(self.models[name].parameters())
        if not params:
            raise ValueError("No trainable parameters: check trainable_components")

        self.optimizer = torch.optim.AdamW(
            params,
            lr=trainer_cfg.get("lr", 3e-4),
            weight_decay=trainer_cfg.get("weight_decay", 0.05),
        )
        self.epochs = trainer_cfg.get("epochs", 100)
        self.log_every = trainer_cfg.get("log_every", 10)
        self.ckpt_every = trainer_cfg.get("ckpt_every", 10)
        self.val_every = trainer_cfg.get("val_every", 1)

        resume = trainer_cfg.get("resume", None)
        if resume:
            ckpt = self.load_checkpoint(resume)
            self._resumed_ckpt = ckpt
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt.get("epoch", 0) + 1

            # best_val_loss must come from best.pt, not from the checkpoint
            # we're resuming: `resume` may point to a periodic checkpoint
            # (not the best one), so ckpt["val_loss"] would only reflect
            # that epoch, risking overwriting best.pt with a worse one.
            best_path = os.path.join(os.path.dirname(resume), "best.pt")
            if os.path.isfile(best_path):
                best_ckpt = torch.load(best_path, map_location="cpu")
                self.best_val_loss = best_ckpt.get("val_loss", float("inf")) or float("inf")
                log.info("best_val_loss restored from best.pt: %.4f", self.best_val_loss)
            else:
                self.best_val_loss = ckpt.get("val_loss", float("inf")) or float("inf")

    def _setup_wandb(self):
        wandb_cfg = self.cfg.get("wandb", None)
        if not wandb_cfg or not wandb_cfg.get("enable", False):
            self.wandb_run = None
            return

        run_id = None
        if self._resumed_ckpt is not None:
            run_id = self._resumed_ckpt.get("wandb_run_id", None)

        self.wandb_run = wandb.init(
            project=wandb_cfg.get("project", "jepa-pose"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("name", None),
            mode=wandb_cfg.get("mode", "online"),
            tags=wandb_cfg.get("tags", None),
            dir=_output_dir(),  # same dir as the Hydra run, regardless of job.chdir
            config=OmegaConf.to_container(self.cfg, resolve=True),
            id=run_id,
            resume="allow" if run_id else None,
        )
        log.info("wandb initialized: dir=%s run_id=%s", _output_dir(), self.wandb_run.id)

    # ------------------------------------------------------------------ #
    # Public entry point: single method, behaves differently based on
    # is_train. Alternative: explicit fit()/predict(), caller's choice
    # (see train.py/infer.py below).
    # ------------------------------------------------------------------ #
    def run(self):
        if self.is_train:
            return self.fit()
        return self.predict()

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self):
        assert self.is_train, "fit() requires Engine(cfg, is_train=True)"
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(epoch)

            val_metrics = {}
            if self.val_loader is not None and epoch % self.val_every == 0:
                val_metrics = self.val_epoch(epoch)

            log.info("[epoch %d] train=%s val=%s", epoch, train_metrics, val_metrics)

            if self.wandb_run is not None:
                payload = {"epoch": epoch}
                payload.update({f"train/{k}": v for k, v in train_metrics.items()})
                payload.update({f"val/{k}": v for k, v in val_metrics.items()})
                wandb.log(payload, step=epoch)

            is_best = False
            if val_metrics:
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    is_best = True

            if epoch % self.ckpt_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss=val_metrics.get("loss"), is_best=is_best)

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def train_epoch(self, epoch: int) -> dict:
        self._set_train_mode()
        running_loss, n_batches = 0.0, 0
        start_time = time.time()
        total_steps = _safe_len(self.loader)

        for step, batch in enumerate(self.loader):
            batch = self._to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred, target = self._forward(batch)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            if step % self.log_every == 0:
                if total_steps:
                    log.info("epoch %d step %d/%d loss=%.4f", epoch, step, total_steps, loss.item())
                else:
                    log.info("epoch %d step %d loss=%.4f", epoch, step, loss.item())

        elapsed = time.time() - start_time
        return {"loss": running_loss / max(n_batches, 1), "time_s": round(elapsed, 1)}

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        """Same L1 loss as training, computed on unseen shards."""
        if self.val_loader is None:
            return {}

        for m in self.models.values():
            m.eval()

        running_loss, n_batches = 0.0, 0
        start_time = time.time()

        for batch in self.val_loader:
            batch = self._to_device(batch)
            pred, target = self._forward(batch)
            loss = self.criterion(pred, target)
            running_loss += loss.item()
            n_batches += 1

        elapsed = time.time() - start_time
        self._set_train_mode()  # restore train mode for trainable modules
        return {"loss": running_loss / max(n_batches, 1), "time_s": round(elapsed, 1)}

    def _to_device(self, batch: dict) -> dict:
        """Move every tensor in the batch to self.device; leave non-tensor fields (strings) as-is."""
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _forward(self, batch: dict):
        """Run encoder2d (frozen, target) and predictor (trainable, prediction)."""
        img = batch["image"]
        pose = batch["rot_matrix"].flatten(1)   # (B, 3, 3) -> (B, 9)
        m = batch["mask_tokens"]                # (B, 256)
        context = batch["context_3d"]           # (B, N, context_dim)

        with torch.no_grad():
            target = self.models["encoder2d"](img)   # frozen, no grad needed

        pred = self.models["predictor"](context, pose, m)   # trainable, must track grad

        return pred, target

    def _set_train_mode(self):
        for name, m in self.models.items():
            m.train(name in self.trainable_components)

    def save_checkpoint(self, epoch: int, val_loss: float | None = None, is_best: bool = False):
        ckpt_dir = os.path.join(_output_dir(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        payload = {
            "epoch": epoch,
            "models": {k: m.state_dict() for k, m in self.models.items()},
            "optimizer": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "wandb_run_id": self.wandb_run.id if self.wandb_run is not None else None,
        }

        path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
        torch.save(payload, path)
        log.info("Checkpoint saved: %s", path)

        if is_best:
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(payload, best_path)
            log.info("New best checkpoint (val_loss=%.4f): %s", val_loss, best_path)

    def load_checkpoint(self, path: str, strict: bool = False) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        for name, module in self.models.items():
            if name in ckpt["models"]:
                module.load_state_dict(ckpt["models"][name], strict=strict)
            elif strict:
                raise KeyError(f"Checkpoint has no state for '{name}'")
        log.info("Checkpoint loaded from %s (epoch %s)", path, ckpt.get("epoch", "?"))
        return ckpt

    def predict(self):
        raise NotImplementedError("predict() not yet implemented")