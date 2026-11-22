import pytest
from threadpoolctl import threadpool_limits

from broadway.spark_pipeline import start_spark


@pytest.fixture(scope="session", autouse=True)
def limit_native_threads():
    with threadpool_limits(limits=2):
        yield


@pytest.fixture(scope="session")
def spark():
    session = start_spark()
    yield session
    session.stop()
