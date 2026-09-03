from __future__ import annotations

import logging
import math
import os
import time

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _safe_len(x):
    try:
        return len(x)
    except TypeError:
        return None


def _output_dir() -> str:
    try:
        return HydraConfig.get().runtime.output_dir
    except ValueError:
        return os.getcwd()


class Engine:
    """ The brain """

    def __init__(self, cfg: DictConfig, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
        self.device = torch.device(cfg.get("device", "cuda:0"))

        self.trainable_components = list(cfg.train.get("trainable_components", ["predictor"]))

        self.models = self._build_models()

        # Peso del termine di occupancy rispetto al content loss.
        # occupancy_loss e content_loss vivono su scale diverse (BCE vs L1 su
        # feature ad alta dimensionalita'), quindi lambda va tarato guardando
        # i due termini separatamente nei log, non assunto = 1 a scatola chiusa.
        self.lambda_occ = float(cfg.train.get("lambda_occ", 1.0))

        # NOTA: mask_threshold non entra piu' nella loss (ne' content ne' occupancy,
        # entrambe usano mask_tokens come segnale continuo/soft). Resta solo come
        # soglia di discretizzazione per metriche interpretabili (IoU) in validazione.
        self.mask_threshold = float(cfg.train.get("mask_threshold", 0.5))

        self.loader = self._build_dataloader(split="train")

        self.optimizer = None
        self.scheduler = None
        self.epochs = None
        self.steps_per_epoch = None
        self.val_steps = None
        self.log_every = None
        self.ckpt_every = None
        self.val_every = None
        self.start_epoch = 0
        self.global_step = 0
        self.val_loader = None
        self.best_val_loss = float("inf")
        self.early_stop_patience = None
        self.early_stop_min_delta = 0.0
        self.epochs_without_improvement = 0
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

    # Build..

    def _build_models(self) -> dict[str, nn.Module]:
        encoder2d = hydra.utils.instantiate(self.cfg.encoder2d).to(self.device)
        encoder3d = hydra.utils.instantiate(self.cfg.encoder3d).to(self.device)
        predictor = hydra.utils.instantiate(
            self.cfg.Predictor,
            context_dim=encoder3d.output_dim[0][-1],
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
            persistent_workers=num_workers > 0,
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

        self.epochs = trainer_cfg.get("epochs", 50)
        self.steps_per_epoch = trainer_cfg.get("steps_per_epoch", 2000)
        self.val_steps = trainer_cfg.get("val_steps", 100)
        self.log_every = trainer_cfg.get("log_every", 20)
        self.ckpt_every = trainer_cfg.get("ckpt_every", 5)
        self.val_every = trainer_cfg.get("val_every", 1)

        self.early_stop_patience = trainer_cfg.get("early_stop_patience", None)
        self.early_stop_min_delta = float(trainer_cfg.get("early_stop_min_delta", 0.0))

        self.total_steps = self.epochs * self.steps_per_epoch
        warmup_steps = max(1, int(self.total_steps * trainer_cfg.get("warmup_frac", 0.03)))
        min_lr_ratio = float(trainer_cfg.get("min_lr_ratio", 0.01))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, self.total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        log.info(
            "Schedule: %d epochs x %d steps = %d total steps (warmup %d, val %d batches)",
            self.epochs, self.steps_per_epoch, self.total_steps, warmup_steps, self.val_steps,
        )

        resume = trainer_cfg.get("resume", None)
        if resume:
            ckpt = self.load_checkpoint(resume)
            self._resumed_ckpt = ckpt
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("global_step", self.start_epoch * self.steps_per_epoch)
            self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)

            if ckpt.get("scheduler") is not None:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            else:
                for _ in range(self.global_step):
                    self.scheduler.step()

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
            dir=_output_dir(),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            id=run_id,
            resume="allow" if run_id else None,
        )
        log.info("wandb initialized: dir=%s run_id=%s", _output_dir(), self.wandb_run.id)

    ## PUBLIC ENTRY POINTS

    def run(self):
        if self.is_train:
            return self.fit()
        return self.predict()

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
                wandb.log(payload, step=self.global_step)

            is_best = False
            if val_metrics:
                if val_metrics["loss"] < self.best_val_loss - self.early_stop_min_delta:
                    self.best_val_loss = val_metrics["loss"]
                    self.epochs_without_improvement = 0
                    is_best = True
                else:
                    self.epochs_without_improvement += 1

            if epoch % self.ckpt_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss=val_metrics.get("loss"), is_best=is_best)

            if self._should_early_stop(epoch):
                break

        if self.wandb_run is not None:
            self.wandb_run.finish()

    def _should_early_stop(self, epoch: int) -> bool:
        if not self.early_stop_patience:
            return False
        if self.epochs_without_improvement < self.early_stop_patience:
            return False
        log.info(
            "Early stopping at epoch %d: no improvement for %d validations (best=%.4f)",
            epoch, self.epochs_without_improvement, self.best_val_loss,
        )
        return True

    # --- loss: content (pesata, continua) + occupancy (BCE soft, esplicita) ---
    #
    # mask_tokens (B, N) e' una frazione continua in [0,1] (viene da un resize
    # BOX sulla maschera binaria, quindi e' gia' un soft-label naturale). Non
    # usiamo piu' una soglia per decidere "dentro/fuori" dalla loss:
    # - il CONTENUTO pesa ogni token in proporzione alla sua copertura reale
    #   (un bordo al 30% conta poco, uno al 95% conta quasi come pieno):
    #   niente piu' cutoff binario che amplifica il rumore del segmentatore.
    # - l'OCCUPANCY e' un head separato, allenato a rigenerare mask_tokens
    #   stesso: e' il predittore che ora deve imparare "dove" va l'oggetto
    #   sotto una data R, non piu' un'informazione che riceve gratis in input.

    @staticmethod
    def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """L1 per-token pesata da weight (continuo, non binario). pred/target: (B, N, D), weight: (B, N)."""
        per_token = F.l1_loss(pred, target, reduction="none").mean(dim=-1)  # (B, N)
        denom = weight.sum().clamp(min=1e-6)
        return (per_token * weight).sum() / denom

    @staticmethod
    def _weighted_cos(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Cosine similarity media, pesata da weight (continuo)."""
        per_token = F.cosine_similarity(pred, target, dim=-1)  # (B, N)
        denom = weight.sum().clamp(min=1e-6)
        return (per_token * weight).sum() / denom

    @staticmethod
    def _baseline_l1(target: torch.Tensor, weight: torch.Tensor) -> float:
        """L1 del miglior predittore costante (mediana pesata approssimata con
        soglia solo qui, per avere una baseline scalare semplice da confrontare)."""
        valid = (weight > 0.5)
        target_valid = target[valid] if valid.any() else target.reshape(-1, target.shape[-1])
        median = target_valid.median(dim=0, keepdim=True).values
        return F.l1_loss(median.expand_as(target_valid), target_valid).item()

    def _compute_losses(self, pred_content, pred_occ_logits, target, mask_tokens):
        """
        pred_content: (B, N, D) - output del predictor, spazio DINOv2
        pred_occ_logits: (B, N) - logit dell'occupancy head (pre-sigmoid)
        target: (B, N, D) - DINOv2 frozen, MAI mascherato/alterato
        mask_tokens: (B, N) - frazione continua di copertura oggetto, GT
        """
        content_loss = self._weighted_l1(pred_content, target, mask_tokens)

        # soft-BCE: mask_tokens come probabilita' target, non come cutoff
        occupancy_loss = F.binary_cross_entropy_with_logits(pred_occ_logits, mask_tokens)

        total = content_loss + self.lambda_occ * occupancy_loss
        return total, content_loss, occupancy_loss

    def train_epoch(self, epoch: int) -> dict:
        self._set_train_mode()
        self.loader.dataset._epoch_counter.value = epoch
        running = {"loss": 0.0, "content_loss": 0.0, "occ_loss": 0.0, "cos": 0.0}
        n_batches = 0
        start_time = time.time()
        total_steps = self.steps_per_epoch

        for step, batch in enumerate(self.loader):

            if step >= self.steps_per_epoch:
                break

            batch = self._to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)
            pred_content, pred_occ_logits, target = self._forward(batch)

            mask_tokens = batch["mask_tokens"]
            loss, content_loss, occ_loss = self._compute_losses(
                pred_content, pred_occ_logits, target, mask_tokens
            )

            with torch.no_grad():
                cos = self._weighted_cos(pred_content, target, mask_tokens).item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            running["loss"] += loss.item()
            running["content_loss"] += content_loss.item()
            running["occ_loss"] += occ_loss.item()
            running["cos"] += cos
            n_batches += 1

            if step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                log.info(
                    "epoch %d step %d/%d loss=%.4f content=%.4f occ=%.4f cos=%.4f lr=%.2e",
                    epoch, step, total_steps, loss.item(), content_loss.item(),
                    occ_loss.item(), cos, lr,
                )
                if self.wandb_run is not None:
                    wandb.log({
                        "train/loss_step": loss.item(),
                        "train/content_loss_step": content_loss.item(),
                        "train/occ_loss_step": occ_loss.item(),
                        "train/cos_step": cos,
                        "train/lr": lr,
                    }, step=self.global_step)

        elapsed = time.time() - start_time
        seen = n_batches * self.cfg.train.loader.batch_size
        n = max(n_batches, 1)
        return {
            "loss": running["loss"] / n,
            "content_loss": running["content_loss"] / n,
            "occ_loss": running["occ_loss"] / n,
            "cos": running["cos"] / n,
            "time_s": round(elapsed, 1),
            "samples_per_s": round(seen / max(elapsed, 1e-6), 1),
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        if self.val_loader is None:
            return {}

        for m in self.models.values():
            m.eval()

        agg = {
            "loss": 0.0, "loss_perm": 0.0,
            "cos": 0.0, "cos_perm": 0.0,
            "occ_iou": 0.0, "occ_iou_perm": 0.0,
            "baseline_l1": 0.0,
        }
        n_batches = 0
        start_time = time.time()

        for batch in self.val_loader:

            if n_batches >= self.val_steps:
                break

            batch = self._to_device(batch)
            mask_tokens = batch["mask_tokens"]

            pred_content, pred_occ_logits, target = self._forward(batch)
            loss, _, _ = self._compute_losses(pred_content, pred_occ_logits, target, mask_tokens)
            agg["loss"] += loss.item()
            agg["cos"] += self._weighted_cos(pred_content, target, mask_tokens).item()
            agg["baseline_l1"] += self._baseline_l1(target, mask_tokens)

            # IoU tra occupancy predetta (con R vera) e GT, entrambe discretizzate
            # solo qui per avere una metrica interpretabile in [0,1].
            occ_pred_bin = (torch.sigmoid(pred_occ_logits) > self.mask_threshold)
            occ_gt_bin = (mask_tokens > self.mask_threshold)
            inter = (occ_pred_bin & occ_gt_bin).float().sum(dim=1)
            union = (occ_pred_bin | occ_gt_bin).float().sum(dim=1).clamp(min=1e-6)
            agg["occ_iou"] += (inter / union).mean().item()

            # --- controllo: rotazione permutata nel batch ---
            B = target.shape[0]
            perm = torch.randperm(B, device=target.device)
            if B > 1 and (perm == torch.arange(B, device=target.device)).all():
                perm = perm.roll(1)

            R_mat = batch["rot_matrix"]
            pred_content_perm, pred_occ_logits_perm, _ = self._forward(batch, override_R=R_mat[perm])

            loss_perm, _, _ = self._compute_losses(
                pred_content_perm, pred_occ_logits_perm, target, mask_tokens
            )
            agg["loss_perm"] += loss_perm.item()
            agg["cos_perm"] += self._weighted_cos(pred_content_perm, target, mask_tokens).item()

            occ_pred_perm_bin = (torch.sigmoid(pred_occ_logits_perm) > self.mask_threshold)
            inter_p = (occ_pred_perm_bin & occ_gt_bin).float().sum(dim=1)
            union_p = (occ_pred_perm_bin | occ_gt_bin).float().sum(dim=1).clamp(min=1e-6)
            agg["occ_iou_perm"] += (inter_p / union_p).mean().item()

            n_batches += 1

        elapsed = time.time() - start_time
        self._set_train_mode()

        n = max(n_batches, 1)
        loss_true = agg["loss"] / n
        loss_perm = agg["loss_perm"] / n
        cos_true = agg["cos"] / n
        cos_perm = agg["cos_perm"] / n
        baseline = agg["baseline_l1"] / n

        return {
            "loss": loss_true,
            "loss_perm": loss_perm,
            "gap_R": (loss_perm - loss_true) / max(loss_perm, 1e-8),
            "cos": cos_true,
            "cos_perm": cos_perm,
            "gap_cos": cos_true - cos_perm,
            # nuova metrica: quanto l'occupancy predetta segue la vera R
            # rispetto a una R sbagliata (permutata). Isola "sa dove va
            # l'oggetto" da "sa cosa c'e' dentro" (gap_cos).
            "occ_iou": agg["occ_iou"] / n,
            "occ_iou_perm": agg["occ_iou_perm"] / n,
            "gap_occ_iou": (agg["occ_iou"] - agg["occ_iou_perm"]) / n,
            "baseline_l1": baseline,
            "loss_ratio": loss_true / max(baseline, 1e-8),
            "time_s": round(elapsed, 1),
        }

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    @staticmethod
    def _generate_rays(K, R, trans, grid_size=16, patch_size=14):
        B = K.shape[0]
        device = K.device

        y = torch.arange(grid_size, device=device, dtype=torch.float32)
        x = torch.arange(grid_size, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        u = grid_x * patch_size + (patch_size / 2.0)
        v = grid_y * patch_size + (patch_size / 2.0)

        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3)
        uv1 = uv1.unsqueeze(0).expand(B, -1, -1)

        K_inv = torch.linalg.inv(K)
        d_c = torch.bmm(uv1, K_inv.transpose(1, 2))

        R_inv = R.transpose(1, 2)
        d_o = F.normalize(torch.bmm(d_c, R_inv.transpose(1, 2)), p=2, dim=-1)   # (B, N, 3)

        O = -torch.bmm(R_inv, trans.unsqueeze(-1)).squeeze(-1)                 # (B, 3)
        O_exp = O.unsqueeze(1).expand(-1, d_o.shape[1], -1)
        moment = torch.cross(O_exp, d_o, dim=-1)                               # (B, N, 3)

        return d_o, moment

    def _forward(self, batch, override_R=None):
        img = batch["image"]
        K = batch["K"]
        trans = batch["trans"]
        context_features = batch["context_3d"]
        context_centers = batch["center_3d"]

        R_mat = override_R if override_R is not None else batch["rot_matrix"]
        ray_dirs, ray_moments = self._generate_rays(K, R_mat, trans, grid_size=16, patch_size=14)

        with torch.no_grad():
            target = self.models["encoder2d"](img)

        pred_content, pred_occ_logits = self.models["predictor"](
            ray_dirs=ray_dirs,
            ray_moments=ray_moments,
            context_features=context_features,
            context_centers=context_centers,
        )

        return pred_content, pred_occ_logits, target

    def _set_train_mode(self):
        for name, m in self.models.items():
            m.train(name in self.trainable_components)

    def save_checkpoint(self, epoch: int, val_loss: float | None = None, is_best: bool = False):
        ckpt_dir = os.path.join(_output_dir(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "models": {k: m.state_dict() for k, m in self.models.items()},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "val_loss": val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "wandb_run_id": self.wandb_run.id if self.wandb_run is not None else None,
        }

        if is_best:
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(payload, best_path)
            log.info("New best checkpoint (val_loss=%.4f): %s", val_loss, best_path)
        else:
            path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            torch.save(payload, path)
            log.info("Checkpoint saved: %s", path)

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