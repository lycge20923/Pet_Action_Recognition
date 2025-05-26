
import argparse
import os
import cv2
import numpy as np
import time
import torch
from PIL import Image
import torch.nn.functional as F
import json

from .SEA_RAFT.core.raft import RAFT
from .SEA_RAFT.core.utils.flow_viz import flow_to_image
from .SEA_RAFT.core.utils.utils import load_ckpt

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_args(args):
    json_path = args.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(args).items()):
        args_dict[key] = value
    return args

def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

class OpticalFlowModel:
    def __init__(self, model_url_name:str, cfg:str, device:str="cuda", path:str=None, **kwargs):
        args = argparse.Namespace(cfg = cfg, path = path, url = model_url_name, device = device)
        self.args = parse_args(args)
        self.model = RAFT.from_pretrained(model_url_name, args=self.args).to(device)
        self.model.eval()
        self.device = device
    
    def forward_flow(self, image1, image2):
        output = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        flow_final = output['flow'][-1]
        info_final = output['info'][-1]
        return flow_final, info_final
    
    def calc_flow(self, image1, image2):
        img1 = F.interpolate(image1, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        H, W = img1.shape[2:]
        flow, info = self.forward_flow(img1, img2)
        flow_down = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        info_down = F.interpolate(info, scale_factor=0.5 ** self.args.scale, mode='area')
        return flow_down, info_down
    
    def get_heatmap(self, info):
        raw_b = info[:, 2:]
        log_b = torch.zeros_like(raw_b)
        weight = info[:, :2].softmax(dim=1)              
        log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
        log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
        heatmap = (log_b * weight).sum(dim=1, keepdim=True)
        return heatmap
    
    def vis_heatmap(self, image, heatmap):
        # theta = 0.01
        # print(heatmap.max(), heatmap.min(), heatmap.mean())
        heatmap = heatmap[:, :, 0]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        # heatmap = heatmap > 0.01
        heatmap = (heatmap * 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = image * 0.3 + colored_heatmap * 0.7
        # Create a color bar
        height, width = image.shape[:2]
        color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
        # Add the color bar to the image
        overlay = overlay.astype(np.uint8)
        combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
        return cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    
    def predict(self, input_path:str, output_path:str=None, correspondence:bool=False):
        # read video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: cannot open video {input_path}")
            return
        if output_path:
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            h = h * 2 + 50 if correspondence else h
            
            # declare writing video
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Read first frame
        ret, frame1 = cap.read()
        if not ret:
            print("Error: cannot read first frame")
            return
        
        total_model_time = 0.0
        idx = 0
        optical_flows, heatmaps = [], []
        with torch.no_grad():
            while True:
                ret, frame2 = cap.read()
                if not ret:
                    break
                frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                frame1_rgb = torch.tensor(frame1_rgb, dtype=torch.float32).permute(2, 0, 1)
                frame2_rgb = torch.tensor(frame2_rgb, dtype=torch.float32).permute(2, 0, 1)
                frame1_rgb = frame1_rgb[None].to(self.device)
                frame2_rgb = frame2_rgb[None].to(self.device)
                
                # inference
                t0 = time.time()
                flow, info= self.calc_flow(frame1_rgb, frame2_rgb)
                total_model_time += (time.time() - t0)
                heatmap = self.get_heatmap(info)
                
                # add to annotation
                flow_store = flow.detach().cpu().numpy().tolist()
                heatmap_store = heatmap.detach().cpu().numpy().tolist()
                optical_flows.append(flow_store)
                heatmaps.append(heatmap_store)
                if output_path:
                    flow_vis = flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True)
                    if correspondence:
                        heatmap_vis = self.vis_heatmap(frame1_rgb[0].permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())
                        vis = np.concatenate((heatmap_vis, flow_vis), axis=0)
                        # print(vis.shape)
                        # print(flow_vis.shape)
                        # print(heatmap_vis.shape)
                        # print(w, h)
                    else:
                        vis = flow_vis
                        # print(vis.shape)
                        # print(w, h)
                    writer.write(vis)

                # Next
                frame1 = frame2
                idx += 1

        cap.release()
        if output_path:
            writer.release()
        print("Done.")
        
        # complete
        zero_flow = np.zeros_like(np.array(optical_flows[0]))
        optical_flows.append(zero_flow.tolist())
        zero_heatmap = np.zeros_like(np.array(heatmaps[0]))
        heatmaps.append(zero_heatmap)
        
        # calculate fps
        exec_fps = idx / total_model_time if total_model_time > 0 else float('inf')
        print(f"Done. Model inference: {idx} frames in {total_model_time:.2f}s → {exec_fps:.2f} FPS")
        return {"optical_flows":optical_flows, "heatmaps":heatmaps, "stat":{"exec_fps": exec_fps}}

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]   
    opt = OpticalFlowModel(device="cuda", model_url_name="MemorySlices/Tartan-C-T-TSKH-spring540x960-M", cfg="features/SEA_RAFT/config/eval/spring-M.json")
    result = opt.predict(input_path=input_path, output_path=output_path, correspondence=True)
    optical_flows = result["optical_flows"]
    print(np.array(optical_flows).shape)
    