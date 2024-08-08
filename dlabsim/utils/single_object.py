import numpy as np

class SingleObject:
    name = ""
    position = np.zeros(3)
    quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    lazy_update = False
    is_pose_dirty = False

    def __init__(self, name, lazy_update=True) -> None:
        self.name = name
        self.lazy_update = lazy_update

    def updatePose(self, position, quat_wxyz):
        if not np.allclose(self.position, position, atol=1e-3) or not np.allclose(self.quat_wxyz, quat_wxyz, atol=1e-3):
            self.position = position.copy()
            self.quat_wxyz = quat_wxyz.copy()
            self.is_pose_dirty = True
