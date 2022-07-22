import hydra
from hydra.utils import get_original_cwd

from src import Mimic3Pipeline,Mimic3Dataset
  
@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg=None):
    owd = get_original_cwd()
    if cfg.preprocess.do:
        pipeline = Mimic3Pipeline(owd)
        pipeline.run()
        return
    dataset = Mimic3Dataset(owd)
    pass

if __name__ == "__main__":
    main()