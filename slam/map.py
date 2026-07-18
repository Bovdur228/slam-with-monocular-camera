# Класс MapPoint и его вспомогательные классы --------------------------------------------------------------------

# Класс Observation хранит связь: конкретная MapPoint была замечена в конкретном KeyFrame <-> под конкретным keypoint index
class Observation:

    def __init__(
        self,
        keyframe,
        keypoint_idx
    ):

        self.keyframe = keyframe
        self.keypoint_idx = keypoint_idx

# Класс, описывающий 3D точку в мире
class MapPoint:

    def __init__(
        self,
        position,
        descriptor
    ):

        self.position = position
        self.descriptor = descriptor

        # список из объектов класса Observation, где хранятся KeyFrames, которые наблюдали эту Map Point и их keypoint, под которым она наблюдалась
        self.observations = []
        self.num_observations = 1

        self.id = None


# Класс, описывающий строение KeyFrame
class KeyFrame:

    def __init__(
        self,
        R,
        t,
        kp,
        des
    ):

        self.R = R
        self.t = t

        self.kp = kp
        self.des = des

        self.id = None

        self.map_points = []


# Временное хранилище наблюдений текущего кадра
class FrameObservation:

    def __init__(
        self,
        point_world,
        descriptor,
        kp_idx
    ):
        self.point_world = point_world
        self.descriptor = descriptor
        self.kp_idx = kp_idx