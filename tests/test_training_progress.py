import sys
import unittest
from unittest.mock import patch

from wmal.training.progress import progress_bar


class TrainingProgressTests(unittest.TestCase):
    def test_disabled_progress_does_not_require_optional_tqdm(self):
        with patch.dict(sys.modules, {'tqdm': None}):
            with progress_bar(2, 'epochs', enabled=False) as progress:
                progress.set_postfix(loss=0.1)
                progress.update()


if __name__ == '__main__':
    unittest.main()
