import os
import collections
import pytest
import datastack.common.settings as settings

import datastack.ml as ml

# Configuration class as namedtuple
MLConfig = collections.namedtuple('MLConfig', 'TEST_DATA')

TEST_DATA = "/"+os.path.join(*ml.__file__.split("/")[:-1])+'/testdata'

def makeConfig(test_data_dir):
    new_config = MLConfig(TEST_DATA=test_data_dir)
    return new_config


@pytest.fixture(scope="session")
def ml_config_fixture(request):

    injectable_config = makeConfig(TEST_DATA)

    # boilerplate
    def teardown():
        pass
    request.addfinalizer(teardown)

    return injectable_config

if __name__ == '__main__':
    pytest.main()
