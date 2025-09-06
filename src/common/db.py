from peewee import Model, Proxy, CharField, FloatField, SqliteDatabase, IntegerField
from .types import EmbeddingConfig
from playhouse.shortcuts import ThreadSafeDatabaseMetadata
from hydra.utils import get_original_cwd
from os.path import join

proxy = Proxy()

class BaseModel(Model):
    class Meta:
        database = proxy
        model_metadata_class = ThreadSafeDatabaseMetadata


class EmbeddingMeta(BaseModel):
    dataset = CharField()
    embedder = CharField()
    embedding_loc = CharField()
    embedding_time = FloatField()
    

class ClusterizationReport(BaseModel):
    dataset = CharField()
    embedder = CharField()
    
    rand_score = FloatField(null=True)
    davies_bouldin_score = FloatField(null=True)
    noise_perc = FloatField(null=True)


class ClassificationReport(BaseModel):
    dataset = CharField()
    task = CharField()
    embedder = CharField()
    
    model = CharField()
    hyperparams = CharField()
    library_hash = CharField()

    cv_metric_name = CharField()
    cv_metric = FloatField()

    test_metric_name = CharField()
    test_metric = FloatField()


class Runtime(BaseModel):
    dataset = CharField()
    embedder = CharField()
    device = CharField()
    
    mean_runtime = FloatField()
    std_runtime = FloatField()
    n_samples = IntegerField()


def close_db():
    proxy.close()
    

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DbContex(metaclass=Singleton):
    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._database = None
        
    def __enter__(self):
        self._database = SqliteDatabase(join(get_original_cwd(), self._config.database))
        proxy.initialize(self._database)
        proxy.create_tables([EmbeddingMeta, Runtime, ClassificationReport, ClusterizationReport], safe=True)
        return None

    def __exit__(self, *args, **kwargs):
        proxy.close()
