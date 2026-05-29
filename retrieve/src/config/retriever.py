import pydantic
import yaml

from .base import EnvYaml

class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int

class RetrieverYaml(pydantic.BaseModel):
    topic_pe: bool
    DDE_kwargs: DDEYaml
    use_structure_encoder: bool = True
    use_semantic_encoder: bool = False
    use_relation_query_sim: bool = True
    use_triple_query_sim: bool = True

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

# FIXED(QARAG): Config for query-aware weak supervision.
# TODO(QARAG): Later add optional aggregation modes, path diversity, and
# flow-prior switches here instead of hardcoding them inside the dataset.
class WeakSupervisionYaml(pydantic.BaseModel):
    strategy: str = 'query_path_soft'
    near_shortest_delta: int = 0
    top_m_paths: int = 10
    max_paths_per_pair: int = 50
    negative_confidence: float = 0.05
    score_temperature: float = 1.0

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    optimizer: OptimizerYaml
    eval: EvalYaml
    weak_supervision: WeakSupervisionYaml = pydantic.Field(
        default_factory=WeakSupervisionYaml)
    train: RetrieverExpYaml

def load_yaml(config_file):
    with open(config_file) as f:
        yaml_data = yaml.load(f, Loader=yaml.loader.SafeLoader)

    task = yaml_data.pop('task')
    assert task == 'retriever'
    
    config = RetrieverTrainYaml(**yaml_data).model_dump()
    config['eval']['k_list'] = [
        int(k) for k in config['eval']['k_list'].split(',')]

    return config
