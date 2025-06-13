import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np

from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, drivingArea_and_lanes_mask, detected_lanes, show_detected_lanes,
    AverageMeter, LoadImages
)
from utils.Lane import Lane

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/input/example.mp4', help='source')  # Updated to your video
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')  # Kept at 384 per your success
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

def detect(opt):
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    stride = 32
    device = select_device(opt.device)
    model = torch.jit.load(weights[0], map_location=device)
    half = device.type != 'cpu'
    model = model.to(device)
    print("✅ Model loaded on:", next(model.parameters()).device)

    if half:
        model.half()
    model.eval()

    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    # Create OpenCV window
    cv2.namedWindow('YOLOP Output', cv2.WINDOW_NORMAL)

    t0 = time.time()
    frame_count = 0
    total_time = 0.0

    try:
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = time_synchronized()
            [pred, anchor_grid], seg, ll = model(img)
            t2 = time_synchronized()

            pred = split_for_trace_model(pred, anchor_grid)
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

            lane = Lane(ll)
            lanes = lane.detect_lanes()
            lane.detect_road()

            frame_time = t2 - t1
            total_time += frame_time
            frame_count += 1
            current_fps = 1 / (frame_time + 1e-6)

            for i, det in enumerate(pred):
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)
                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
                s += '%gx%g ' % img.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img:
                            plot_one_box(xyxy, im0, line_thickness=3)

                im0 = lane.show_detected_lanes(im0)
                im0 = lane.show_roads(im0)

                # Display the frame
                cv2.imshow('YOLOP Output', im0)
                key = cv2.waitKey(30) & 0xFF  # Increased to 30ms for better responsiveness
                if key == ord('q'):
                    print("🛑 Manually stopped processing.")
                    break

                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f"Image saved: {save_path}")
                    else:
                        if vid_path != save_path:
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()
                            fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
                            w, h = im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

            print(f'{s}Done. ({frame_time:.4f}s/frame) | ⚡ FPS: {current_fps:.2f}')

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up
        cv2.destroyAllWindows()
        if save_img and vid_writer:
            vid_writer.release()
            print(f"Video saved: {vid_path}")

    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f'\n🚀 Average FPS over {frame_count} frames: {avg_fps:.2f}')

    print(f'Done. ({time.time() - t0:.3f}s total)')

if __name__ == '_main_':
    opt = make_parser().parse_args()
    print(opt)
    with torch.no_grad():
        detect(opt)