""" Custom types """
from typing import TypeVar

DataFrame = TypeVar('pandas.core.frame.DataFrame')
Series = TypeVar('pandas.core.series.Series')

PsycopgConnection = TypeVar('psycopg2._psycopg.connection')
PsycopgCursor = TypeVar('psycopg2._psycopg.cursor')

SKLearnPipeline = TypeVar('sklearn.pipeline.Pipeline')
