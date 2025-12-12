import yaml

def load_config(path="conf/project.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
