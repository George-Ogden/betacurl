import wandb

def pytest_configure(config):
    wandb.init("test run", mode="disabled")