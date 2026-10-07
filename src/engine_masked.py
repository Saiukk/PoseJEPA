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
from torch.utils.data import DataLoader, IterableDataset

log = logging.getLogger(__name__)


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

        self.context_mode = "aggregator" #cfg.data.get("context_mode", "aggregator")
        if self.context_mode not in ("uni3d", "aggregator"):
            raise ValueError(f"context_mode deve essere 'uni3d' o 'aggregator', ricevuto: {self.context_mode}")

        self.trainable_components = list(cfg.train.get("trainable_components", ["predictor"]))

        self.models = self._build_models()

        # NOTA: la loss ora e' sempre masked L1 punto-per-punto (vedi _masked_l1_per_sample),
        # non serve piu' un nn.L1Loss non mascherato come criterion generico.

        self.loader = self._build_dataloader(split="train")

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.grad_clip = None
        self._trainable_params: list[nn.Parameter] = []
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

        logging.info("Engine2 initialized.")
        
    # Build..

    def _build_models(self) -> dict[str, nn.Module]:
        encoder2d = hydra.utils.instantiate(self.cfg.encoder2d).to(self.device)

        # L'Aggregator riceve le feature DINOv2 per vista: in_dim = dimensione di encoder2d
        extra = {"in_dim": encoder2d.output_dim[-1]} if self.context_mode == "aggregator" else {}
        encoder3d = hydra.utils.instantiate(self.cfg.encoder3d, **extra).to(self.device)

        predictor = hydra.utils.instantiate(
            self.cfg.Predictor,
            context_dim=encoder3d.output_dim[0][-1],
            output_dim=encoder2d.output_dim[-1],
        ).to(self.device)
        models = {"encoder2d": encoder2d, "encoder3d": encoder3d, "predictor": predictor}

        if self.context_mode == "aggregator" and "encoder3d" not in self.trainable_components:
            log.warning(
                "context_mode=aggregator ma 'encoder3d' non e' in trainable_components: "
                "l'Aggregator resta a pesi random!"
            )
        if self.context_mode == "uni3d" and "encoder3d" in self.trainable_components:
            log.warning("context_mode=uni3d: 'encoder3d' resta frozen (rimosso da trainable_components)")
            self.trainable_components.remove("encoder3d")

        for name, module in models.items():
            is_trainable = name in self.trainable_components
            for p in module.parameters():
                p.requires_grad_(is_trainable)
            module.train(is_trainable)
            log.info("Model '%s' -> trainable=%s", name, is_trainable)

        return models

    def _build_dataloader(self, split: str = "train", epoch: int | None = None):
        dataset = hydra.utils.instantiate(
            self.cfg.data, dataset_source=self.cfg.gso.root, split=split
        )

        is_streaming = isinstance(dataset, IterableDataset)
        if is_streaming and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch if epoch is not None else 0)

        n_shards = len(getattr(dataset, "shard_paths", []))
        if n_shards == 0:
            if split == "val":
                log.warning("no shard assigned for validation")
                return None
            raise ValueError(f"No shard found for split='{split}'")

        loader_cfg = self.cfg.train.loader
        requested_workers = loader_cfg.num_workers
        if split == "val":
            # opzionale: meno worker per la validation (ogni loader persistente tiene i suoi processi vivi)
            requested_workers = loader_cfg.get("val_num_workers", requested_workers)

        num_workers = min(requested_workers, n_shards)
        if num_workers != requested_workers:
            log.warning(
                f"num_workers={requested_workers} reduced to {num_workers} "
                f"because only {n_shards} shards available for split='{split}'"
            )

        loader = DataLoader(
            dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            # Worker persistenti anche per lo streaming: l'epoca e' un mp.Value condiviso,
            # quindi set_epoch() nel padre e' visibile ai worker gia' avviati.
            persistent_workers=(num_workers > 0),
            prefetch_factor=loader_cfg.get("prefetch_factor", 1) if num_workers > 0 else None,  #<-- Keep the GPU busy
            multiprocessing_context="fork" if num_workers > 0 else None,
        )
        log.info(
            "[%s] Dataloader: %d shard, %d worker%s%s",
            split, n_shards, num_workers,
            " (persistent)" if num_workers > 0 else "",
            " (streaming)" if is_streaming else "",
        )
        return loader

    @staticmethod
    def _set_dataset_epoch(loader, epoch: int, split: str) -> None:
        """Aggiorna l'epoca del dataset SENZA ricostruire il DataLoader (worker persistenti)."""
        ds = loader.dataset
        if hasattr(ds, "_epoch_counter"):
            ds._epoch_counter.value = epoch
        elif hasattr(ds, "set_epoch"):
            ds.set_epoch(epoch)
        else:
            raise AttributeError(
                f"Dataset {type(ds)} ({split}) non supporta ne' "
                f"_epoch_counter ne' set_epoch: aggiungi il supporto per l'epoca."
            )

    def _setup_training(self):
        trainer_cfg = self.cfg.train.get("trainer", {})

        params = []
        for name in self.trainable_components:
            params += list(self.models[name].parameters())
        if not params:
            raise ValueError("No trainable parameters: check trainable_components")
        self._trainable_params = params

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
        self.grad_clip = trainer_cfg.get("grad_clip", None)     # es. 1.0; None = nessun clipping

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

        self.scaler = torch.amp.GradScaler()

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

    # ---------- loss mascherata (punto per punto, sui token con oggetto visibile) ----------

    @staticmethod
    def _masked_l1_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """L1 per token"""
        err = F.l1_loss(pred, target, reduction="none").mean(dim=-1)  # (B, N)
        valid = mask.float()
        return (err * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _masked_cosine_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cosine per token."""
        cos = F.cosine_similarity(pred, target, dim=-1)  # (B, N)
        valid = mask.float()
        return (cos * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _masked_baseline_l1(target: torch.Tensor, mask: torch.Tensor) -> float:

        valid = mask.float()
        losses = []
        for b in range(target.shape[0]):
            idx = valid[b] > 0
            if idx.sum() == 0:
                losses.append(torch.zeros((), device=target.device))
                continue
            flat = target[b][idx]  # (n_valid, D)
            median = flat.median(dim=0, keepdim=True).values
            losses.append(F.l1_loss(median.expand_as(flat), flat))
        return torch.stack(losses).mean().item()

    def _grad_norms(self) -> dict[str, float]:
        """Norma del gradiente per componente trainabile (da chiamare dopo unscale_)."""
        out = {}
        for name in self.trainable_components:
            gs = [p.grad.detach().float().norm() for p in self.models[name].parameters() if p.grad is not None]
            out[name] = torch.stack(gs).norm().item() if gs else 0.0
        return out

    def train_epoch(self, epoch: int) -> dict:
        self._set_train_mode()

        # Aggiorna l'epoca sul dataset: niente ricostruzione del loader, i worker restano vivi
        self._set_dataset_epoch(self.loader, epoch, "train")

        running_loss = torch.tensor(0.0, device=self.device)
        running_cos = torch.tensor(0.0, device=self.device)
        n_batches = 0
        start_time = time.time()
        total_steps = self.steps_per_epoch

        for step, batch in enumerate(self.loader):

            if step >= self.steps_per_epoch:
                break

            batch = self._to_device(batch)
            mask = batch["mask_tokens"].float()  # (B, N): 1=oggetto visibile, 0=sfondo/occluso
            self.optimizer.zero_grad(set_to_none=True)

            # FORWARD: Aggregator (se attivo) -> raggi -> predictor, tutto nel grafo autograd
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_content, target = self._forward(batch)
                loss = self._masked_l1_per_sample(pred_content, target, mask).mean()
                # Calcola il coseno SENZA chiamare .item() per non bloccare la GPU
                cos = self._masked_cosine_per_sample(pred_content, target, mask).mean()

            # BACKWARD: loss -> predictor -> Aggregator (encoder3d) grazie a requires_grad
            # su entrambi. Il target (encoder2d) e' sotto no_grad e non riceve nulla.
            self.scaler.scale(loss).backward()

            is_log_step = step % self.log_every == 0
            grad_norms: dict[str, float] = {}
            if self.grad_clip or is_log_step:
                self.scaler.unscale_(self.optimizer)          # gradienti in scala reale
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self._trainable_params, self.grad_clip)
                if is_log_step:
                    grad_norms = self._grad_norms()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()
            self.global_step += 1

            running_loss += loss.detach()
            running_cos += cos.detach()
            n_batches += 1

            if is_log_step:
                loss_val = loss.item()
                cos_val = cos.item()
                lr = self.optimizer.param_groups[0]["lr"]
                log.info(
                    "epoch %d step %d/%d loss=%.4f cos=%.4f lr=%.2e grad=%s",
                    epoch, step, total_steps, loss_val, cos_val, lr,
                    {k: round(v, 4) for k, v in grad_norms.items()},
                )
                if self.context_mode == "aggregator" and grad_norms.get("encoder3d", 1.0) == 0.0:
                    log.warning("Gradiente dell'Aggregator = 0: non e' collegato al grafo (controlla il forward).")
                if self.wandb_run is not None:
                    payload = {
                        "train/loss_step": loss_val,
                        "train/cos_step": cos_val,
                        "train/lr": lr,
                    }
                    payload.update({f"grad/{k}": v for k, v in grad_norms.items()})
                    wandb.log(payload, step=self.global_step)

        if n_batches < self.steps_per_epoch:
            log.warning(
                "epoch %d: solo %d/%d batch disponibili (il dataset e' finito prima). "
                "Se streaming: controlla che ci siano abbastanza sample pre-renderizzati "
                "per questa epoca su HF, o riduci steps_per_epoch.",
                epoch, n_batches, self.steps_per_epoch,
            )

        elapsed = time.time() - start_time
        seen = n_batches * self.cfg.train.loader.batch_size
        n = max(n_batches, 1)
        return {
            "loss": (running_loss / n).item(),
            "cos": (running_cos / n).item(),
            "time_s": round(elapsed, 1),
            "samples_per_s": round(seen / max(elapsed, 1e-6), 1),
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        if self.val_loader is None:
            return {}

        self._set_dataset_epoch(self.val_loader, epoch, "val")

        for m in self.models.values():
            m.eval()

        agg = {
            "loss": 0.0, "loss_perm": 0.0,
            "cos": 0.0, "cos_perm": 0.0,
            "baseline_l1": 0.0,
        }
        n_batches = 0
        start_time = time.time()

        for batch in self.val_loader:

            if n_batches >= self.val_steps:
                break

            batch = self._to_device(batch)
            mask = batch["mask_tokens"].float()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                ctx = self._encode_context(batch)              # una volta sola, riusato per il test permutato
                pred_content, target = self._forward(batch, ctx=ctx)
                loss = self._masked_l1_per_sample(pred_content, target, mask).mean()
                cos = self._masked_cosine_per_sample(pred_content, target, mask).mean()

            agg["loss"] += loss.item()
            agg["cos"] += cos.item()
            agg["baseline_l1"] += self._masked_baseline_l1(target, mask)

            # --- sanity check: rotazione permutata nel batch ---
            # NOTA: si usa ancora la mask ORIGINALE (non permutata), perche' indica
            # dove si trova l'oggetto nell'immagine corrente, indipendentemente
            # dalla posa (sbagliata) data in input al predictor.

            B = target.shape[0]
            perm = torch.randperm(B, device=target.device)
            if B > 1 and (perm == torch.arange(B, device=target.device)).all():
                perm = perm.roll(1)

            R_mat = batch["rot_matrix"]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred_content_perm, _ = self._forward(batch, override_R=R_mat[perm], ctx=ctx)
                loss_perm = self._masked_l1_per_sample(pred_content_perm, target, mask).mean()
                cos_perm = self._masked_cosine_per_sample(pred_content_perm, target, mask).mean()

            agg["loss_perm"] += loss_perm.item()
            agg["cos_perm"] += cos_perm.item()

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
    def _generate_rays(K, R, trans, centroid, scale, grid_size=16, patch_size=16):
        """
        Plucker rays, normalized by centroid and scale of the 3D model.
        """
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

        # trans_norm = (trans + R @ centroid) / scale
        # (derivazione: O_norm = (O_world - centroid) / scale, con O_world = -R^T trans)
        centroid_col = centroid.unsqueeze(-1)                                            # (B, 3, 1)
        trans_norm = (trans.unsqueeze(-1) + torch.bmm(R, centroid_col)).squeeze(-1)      # (B, 3)
        trans_norm = trans_norm / scale.unsqueeze(-1)                                    # (B, 3)

        O = -torch.bmm(R_inv, trans_norm.unsqueeze(-1)).squeeze(-1)                      # (B, 3)
        O_exp = O.unsqueeze(1).expand(-1, d_o.shape[1], -1)
        moment = torch.cross(O_exp, d_o, dim=-1)                                         # (B, N, 3)

        return d_o, moment

    def _encode_context(self, batch):
        """Contesto 3D per il predictor: (B, P, D) feature + (B, P, 3) centri.

        - uni3d:      gia' calcolato dal dataset, nessun parametro trainabile.
        - aggregator: cross-attention sulle viste (encoder3d), CON gradiente.
                      I tensori arrivano dal dataset al dtype su disco: il cast
                      avviene qui, sulla GPU.
        """
        if self.context_mode == "aggregator":
            return self.models["encoder3d"].extractFeatures(
                batch["view_feats"].float(),     # (B, P, N, D)
                batch["view_dirs"].float(),      # (B, P, N, 3)
                batch["view_mask"].bool(),       # (B, P, N) bool
                batch["point_xyz"].float(),      # (B, P, 3)
            )
        return batch["context_3d"], batch["center_3d"]

    def _forward(self, batch, override_R=None, ctx=None):
        img = batch["image"]
        B = img.shape[0]
        K = batch["K"].float()
        trans = batch["trans"].float()
        # il dataset salva centroid come (1,3) -> per campione serve (3,)
        centroid = batch["centroid"].reshape(B, 3).float()
        scale = batch["scale"].reshape(B).float()
        R_mat = (override_R if override_R is not None else batch["rot_matrix"]).float()

        # Raggi in fp32: sotto autocast bmm girerebbe in fp16 e la posa perderebbe precisione
        with torch.autocast(device_type="cuda", enabled=False):
            ray_dirs, ray_moments = self._generate_rays(
                K, R_mat, trans, centroid, scale, grid_size=16, patch_size=16
            )

        # Aggregator (o contesto cached). NON dentro no_grad: serve il gradiente
        context_features, context_centers = ctx if ctx is not None else self._encode_context(batch)

        with torch.no_grad():
            target = self.models["encoder2d"](img)

        pred_content = self.models["predictor"](
            ray_dirs=ray_dirs,
            ray_moments=ray_moments,
            context_features=context_features,
            context_centers=context_centers,
        )

        return pred_content, target

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