# carla_yolop.py
import carla
import cv2
import numpy as np
import random
import time
import queue
import sys
from pathlib import Path
import torch
import os

# Ensure CARLA PythonAPI egg is in sys.path (adjust path as needed)
# sys.path.append('C:/path/to/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-win-amd64.egg')

sys.path.append(str(Path(__file__).parent))
from yolop_inference import YOLOPModel
from utils.Lane import Lane

OUTPUT_VIDEO_PATH = "output_test2.mp4"
yolop_model = None

def wait_for_carla(host='127.0.0.1', port=2000, timeout=90):
    import socket, time
    print("⏳ Waiting for CARLA to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("✅ Connected to CARLA server!")
                return True
        except:
            time.sleep(1)
    raise TimeoutError(f"❌ CARLA not available after {timeout} seconds.")

def get_client(host='127.0.0.1', port=2000, timeout=90.0):
    wait_for_carla()
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client

def camera_callback(image, image_queue):
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        if array.size != image.height * image.width * 4:
            print("Warning: RGB buffer size mismatch")
            return
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Extract RGB channels
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image_queue.put(array)
    except Exception as e:
        print(f"Error in camera_callback: {e}")

def depth_callback(image, depth_queue):
    depth_queue.put(image)

def depth_to_array(image):
    try:
        img = np.frombuffer(image.raw_data, dtype=np.uint8)
        if img.size != image.height * image.width * 4:
            print("Warning: Depth buffer size mismatch")
            return np.zeros((image.height, image.width), dtype=np.float32)
        img = img.reshape((image.height, image.width, 4))
        img = img[:, :, :3].astype(np.float32)
        normalized = (img[:, :, 0] + img[:, :, 1] * 256 + img[:, :, 2] * 256**2) / (256**3 - 1)
        depth_meters = 1000.0 * normalized
        return depth_meters
    except Exception as e:
        print(f"Error in depth_to_array: {e}")
        return np.zeros((image.height, image.width), dtype=np.float32)

def annotate_vehicle_zones(img, detections, depth_array, lane_module):
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        cx = (x1 + x2) // 2
        cy = y2 - int((y2 - y1) * 0.1)
        zone = lane_module.get_zone_at(cx, cy)
        zone_text = f"Lane {zone}" if zone in [1, 2, 3] else "Lane ?"
        colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}
        color = colors.get(zone, (255, 255, 255))

        cv2.putText(img, zone_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        patch = depth_array[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6]
        valid_depths = patch[(patch > 0.1) & (patch < 100.0)]
        if valid_depths.size > 0:
            distance = float(np.percentile(valid_depths, 30))
            if 0.5 < distance < 100.0:
                label = f"{distance:.2f}m"
                cv2.putText(img, label, (x1, max(10, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def run_simulation(client, args):
    world = client.get_world()
    # Force load Town03
    if 'Town03' not in world.get_map().name:
        print("🔄 Loading Town03...")
        client.load_world('Town03')
        world = client.get_world()
        time.sleep(10.0)  # Wait for map to load
        print("✅ Town03 loaded.")
        # Set clear weather to improve lane detection
        weather = carla.WeatherParameters.ClearNoon
        world.set_weather(weather)
        print("✅ Weather set to ClearNoon.")

    
    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()

    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    pedestrian_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Error: No spawn points available in the current map.")
        raise RuntimeError("No spawn points available.")
    start_point = spawn_points[308] if len(spawn_points) > 308 else spawn_points[0]

    vehicle = world.spawn_actor(vehicle_bp, start_point)
    vehicle.set_autopilot(True)

    pedestrian_transform = random.choice(spawn_points)
    pedestrian = world.spawn_actor(pedestrian_bp, pedestrian_transform)
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=pedestrian)
    walker_controller.start()
    walker_controller.go_to_location(world.get_random_location_from_navigation())
    walker_controller.set_max_speed(1.2)

    npc_vehicle_bp = blueprint_library.filter('vehicle.*')[1]
    npc_spawn_points = random.sample(spawn_points, 5)
    npc_vehicles = []
    for sp in npc_spawn_points:
        npc = world.try_spawn_actor(npc_vehicle_bp, sp)
        if npc:
            npc.set_autopilot(True)
            npc_vehicles.append(npc)

    front_rgb_q, rear_rgb_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
    front_depth_q, rear_depth_q = queue.Queue(maxsize=1), queue.Queue(maxsize=1)

    def spawn_camera(bp_name, loc, rot, q, attach_to):
        try:
            bp = blueprint_library.find(bp_name)
            bp.set_attribute("image_size_x", str(args.camera_width))
            bp.set_attribute("image_size_y", str(args.camera_height))
            bp.set_attribute("fov", str(90.0))
            trans = carla.Transform(carla.Location(**loc), carla.Rotation(**rot))
            cam = world.spawn_actor(bp, trans, attach_to=attach_to)
            if bp_name == 'sensor.camera.rgb':
                cam.listen(lambda data: camera_callback(data, q))
            elif bp_name == 'sensor.camera.depth':
                cam.listen(lambda data: depth_callback(data, q))
            return cam
        except Exception as e:
            print(f"Error spawning camera {bp_name}: {e}")
            raise

    # Adjust camera positions for better lane visibility
    front_rgb = spawn_camera('sensor.camera.rgb', dict(x=2.5, z=1.2), dict(pitch=-15.0), front_rgb_q, vehicle)
    rear_rgb = spawn_camera('sensor.camera.rgb', dict(x=-2.5, z=1.2), dict(yaw=180, pitch=-15.0), rear_rgb_q, vehicle)
    front_depth = spawn_camera('sensor.camera.depth', dict(x=2.5, z=1.2), dict(pitch=-15.0), front_depth_q, vehicle)
    rear_depth = spawn_camera('sensor.camera.depth', dict(x=-2.5, z=1.2), dict(yaw=180, pitch=-15.0), rear_depth_q, vehicle)

    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        20.0,
        (args.camera_width * 2, args.camera_height)
    )
    if not out.isOpened():
        print("❌ Failed to open video writer. Ensure OpenCV is built with FFmpeg support.")
        raise RuntimeError("Video writer initialization failed.")

    yolop = YOLOPModel(weights_path=args.weights, device=args.device, img_size=args.img_size)
    # Warm up YOLOP model
    dummy_image = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    for _ in range(5):
        yolop.infer(dummy_image)
    print("✅ YOLOP model warmed up.")


    print("🚗 Running... Press 'q' to stop.")
    cv2.namedWindow("YOLOP Views", cv2.WINDOW_NORMAL)

    front_fps_list, rear_fps_list = [], []

    try:
        while True:
            world.tick()  # Advance the simulator frame

            # Clear queues to avoid stale data
            while not front_rgb_q.empty():
                try:
                    front_rgb_q.get_nowait()
                except queue.Empty:
                    break
            while not rear_rgb_q.empty():
                try:
                    rear_rgb_q.get_nowait()
                except queue.Empty:
                    break

            # Get front RGB frame
            try:
                f_rgb = front_rgb_q.get(timeout=2.0)
                if not isinstance(f_rgb, np.ndarray) or f_rgb.shape != (args.camera_height, args.camera_width, 3):
                    print(f"⚠️ Invalid front RGB frame: {type(f_rgb)} {f_rgb.shape if isinstance(f_rgb, np.ndarray) else 'N/A'}")
                    continue
            except queue.Empty:
                print("⚠️ Front RGB timeout")
                continue

            # Get rear RGB frame
            try:
                r_rgb = rear_rgb_q.get(timeout=2.0)
                if not isinstance(r_rgb, np.ndarray) or r_rgb.shape != (args.camera_height, args.camera_width, 3):
                    print(f"⚠️ Invalid rear RGB frame: {type(r_rgb)} {r_rgb.shape if isinstance(r_rgb, np.ndarray) else 'N/A'}")
                    continue
            except queue.Empty:
                print("⚠️ Rear RGB timeout")
                continue

            # Optional: get depth frames (non-blocking)
            try:
                f_depth = front_depth_q.get(timeout=0.5)
            except queue.Empty:
                f_depth = None

            try:
                r_depth = rear_depth_q.get(timeout=0.5)
            except queue.Empty:
                r_depth = None

            # Convert RGB to BGR for YOLOP inference
            f_rgb_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
            r_rgb_bgr = cv2.cvtColor(r_rgb, cv2.COLOR_RGB2BGR)

            # Run YOLOP inference
            try:
                t1 = time.time()
                f_disp, f_dets, lane_front, _ = yolop.infer(f_rgb_bgr)
                t2 = time.time()
                r_disp, r_dets, lane_rear, _ = yolop.infer(r_rgb_bgr)
                t3 = time.time()
            except Exception as e:
                print(f"⚠️ YOLOP inference error: {e}")
                continue

            fps_front = 1 / (t2 - t1 + 1e-6)
            fps_rear = 1 / (t3 - t2 + 1e-6)
            front_fps_list.append(fps_front)
            rear_fps_list.append(fps_rear)

            # Lane + depth visualization
            if lane_front:
                f_disp = lane_front.show_roads(f_disp)
                if f_depth:
                    depth_arr = depth_to_array(f_depth)
                    f_disp = annotate_vehicle_zones(f_disp, f_dets, depth_arr, lane_front)

            if lane_rear:
                r_disp = lane_rear.show_roads(r_disp)
                if r_depth:
                    depth_arr = depth_to_array(r_depth)
                    r_disp = annotate_vehicle_zones(r_disp, r_dets, depth_arr, lane_rear)

            # Resize and combine views
            if f_disp is None or r_disp is None:
                print("⚠️ Skipping frame due to empty output.")
                continue

            f_disp = cv2.resize(f_disp, (args.camera_width, args.camera_height))
            r_disp = cv2.resize(r_disp, (args.camera_width, args.camera_height))
            comb = np.hstack([f_disp, r_disp])

            if comb.shape[2] != 3:
                print("⚠️ Invalid frame shape, skipping.")
                continue

            cv2.putText(comb, f'Front FPS: {fps_front:.2f}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(comb, f'Rear FPS: {fps_rear:.2f}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if comb.shape[0] != args.camera_height or comb.shape[1] != args.camera_width * 2 or comb.shape[2] != 3:
                print(f"⚠️ Skipping invalid frame size: {comb.shape}")
                continue

            out.write(comb)
            cv2.imshow("YOLOP Views", comb)
            if cv2.waitKey(1) in [ord('q'), 27]:
                break

            time.sleep(0.01)

    finally:
        print("🛑 Stopping simulation and cleaning up.")
        try:
            out.release()
            for sensor in [front_rgb, rear_rgb, front_depth, rear_depth]:
                if sensor and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
            if walker_controller and walker_controller.is_alive:
                walker_controller.stop()
                walker_controller.destroy()
            if pedestrian and pedestrian.is_alive:
                pedestrian.destroy()
            for npc in npc_vehicles:
                if npc and npc.is_alive:
                    npc.destroy()
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")

        if front_fps_list and rear_fps_list:
            print(f"📊 Average FPS - Front: {sum(front_fps_list)/len(front_fps_list):.2f}, Rear: {sum(rear_fps_list)/len(rear_fps_list):.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--camera_width', type=int, default=1280)
    parser.add_argument('--camera_height', type=int, default=720)
    parser.add_argument('--weights', type=str, default='yolopv2.pt')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--conf-thres', type=float, default=0.5)
    args = parser.parse_args()

    client = get_client(args.host, args.port)
    run_simulation(client, args)