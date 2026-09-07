import unittest


class ImportTest(unittest.TestCase):
    def test_linorobot2_gazebo_imports(self):
        import linorobot2_gazebo

        self.assertIsNotNone(linorobot2_gazebo)
