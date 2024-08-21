
def build_dataloader(cfg):
    name = cfg.dataset_name
    if name == "logo":
        from dataloader.dataloader_logo import get_dataloader
        return get_dataloader(cfg)
    elif name == "gym":
        from dataloader.dataloader_gymnastic import get_dataloader
        return get_dataloader(cfg)
    elif name == "fisv":
        from dataloader.dataloader_fisv import get_dataloader
        return get_dataloader(cfg)
    else:
        raise ValueError(f'Unsupported dataset name [{cfg.dataset}]')