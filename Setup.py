import carla

class CarlaSetup:
    def __init__(self):
        self.actor_list = [] # Simulasyona eklenen nesneleri tutar
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print("World loaded:", self.world)
        self.blueprint_library = self.world.get_blueprint_library()

    def spawnPoints(self):
        return self.world.get_map().get_spawn_points() # Simulasyondaki spawn noktalarını getiriyor

    def addVehicle(self, carName, spawn_point): # Simülasyona istenilen noktaya araç spawnlar
        bp = self.blueprint_library.filter(carName) # Simulasyondaki istenilen araba modelini getiriyor
        if not bp:
            raise ValueError(f"{carName} bulunamadı!")
        vehicle_bp = bp[0] 
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point) # Spawn noktasına aracı spawnlıyoruz 
        self.actor_list.append(vehicle)  # Actor_list'e araba nesnesini ekliyoruz 
        print("Vehicle spawned.")
        return vehicle

    def addCamera(self, camName, width, height, fov, x1, z1, vehicle): # Simulasyona kamera spawnlar
        camera_bp = self.blueprint_library.find(camName)
        camera_bp.set_attribute('image_size_x', f'{width}')
        camera_bp.set_attribute('image_size_y', f'{height}')
        camera_bp.set_attribute('fov', f'{fov}')
        camera_transform = carla.Transform(carla.Location(x=x1, z=z1))
        camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self.actor_list.append(camera)
        print("Camera spawned.")
        return camera