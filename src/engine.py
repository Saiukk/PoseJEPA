

# The IDEA
# self.encoder2d = hydra.utils.instantiate(cfg.encoder2d)
# self.encoder3d = hydra.utils.instantiate(cfg.encoder3d)

# fusion_dim = self.encoder2d.output_dim + self.encoder3d.output_dim
# self.predictor = hydra.utils.instantiate(cfg.Predictor, input_dim=fusion_dim)