
import argparse
import os
import cv2
import numpy as np
import time
import torch
from PIL import Image

from .RAFT.core.raft import RAFT
from .RAFT.core.utils import flow_viz
from .RAFT.core.utils.utils import InputPadder

class OpticalFlowModel:
    def __init__(self, model_dir:str="models/of",
                 model_name:str="things", 
                 device:str="cuda",
                 small:bool=False,
                 mixed_precision:bool=False,
                 alternate_corr:bool=False
                 ):
        self.model_dir = model_dir
        self.model_file_name = f"raft-{model_name}.pth"
        self.device = device
        self._check_and_load_model(small, mixed_precision, alternate_corr)
        
    def _check_and_load_model(self, small, mixed_precision, alternate_corr):
        # load if not exist
        model_path = os.path.join(self.model_dir, self.model_file_name)
        if not os.path.exists(model_path):
            LOAD_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
            ZIP_NAME = "models.zip"
            os.system(f"wget {LOAD_URL}")
            os.system(f"unzip -j {ZIP_NAME} {os.path.join('models', self.model_file_name)} -d {self.model_dir}")
            os.remove(ZIP_NAME)
        model_cfg = argparse.Namespace(
            model = model_path,
            small = small,
            mixed_precision = mixed_precision,
            alternate_corr = alternate_corr
        )
        self.model = torch.nn.DataParallel(RAFT(model_cfg))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.module.to(self.device)
        self.model.eval()
        
    def frame_to_tensor(self, frame):
        """
        Convert BGR frame (numpy) to torch tensor [1,3,H,W] in RGB order, float.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To torch tensor
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return tensor[None].to(self.device)
    
    def viz(self, img=None, flo=None, correspondence=None):
        """
        Generate and save a visualization of image and flow for frame index idx.
        img: [1,3,H,W], flo: [1,3,H,W]
        """
        flo_np = flo[0].permute(1,2,0).cpu().numpy()
        flow_img = flow_viz.flow_to_image(flo_np)
        if correspondence:
            img_np = img[0].permute(1,2,0).cpu().numpy()
            img_disp = img_np.clip(0,255).astype(np.uint8)
            h, w = img_disp.shape[:2]
            flow_img = cv2.resize(flow_img, (w, h), interpolation=cv2.INTER_LINEAR)
            vis = np.concatenate([img_disp, flow_img], axis=0)
        else:
            vis = flow_img
            
        # Convert RGB to BGR for OpenCV
        out = vis[:, :, ::-1]
        return out
    
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
            h = h * 2 if correspondence else h
            
            # declare writing video
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        
        # Read first frame
        ret, frame1 = cap.read()
        if not ret:
            print("Error: cannot read first frame")
            return
        img1 = self.frame_to_tensor(frame1)
        
        total_model_time = 0.0
        idx = 0
        optical_flows = []
        with torch.no_grad():
            while True:
                ret, frame2 = cap.read()
                if not ret:
                    break
                img2 = self.frame_to_tensor(frame2)

                # Pad input to multiple of 8
                padder = InputPadder(img1.shape)
                i1, i2 = padder.pad(img1, img2)

                # Compute flow
                t0 = time.time()
                _, flow_up = self.model(i1, i2, iters=20, test_mode=True)
                t1 = time.time()
                total_model_time += (t1 - t0)
                
                # Visualize and save
                if output_path:
                    out_frame = self.viz(img1, flow_up, correspondence)
                    writer.write(out_frame)
                    
                # adjust to let it writable to json
                flow_up = flow_up.cpu().numpy().tolist()
                optical_flows.append(flow_up)

                # Next
                img1 = img2
                idx += 1

        cap.release()
        if output_path:
            writer.release()
        print("Done.")
        
        # complete
        zero_flow = np.zeros_like(np.array(optical_flows[0]))
        optical_flows.append(zero_flow.tolist())
        
        # calculate fps
        exec_fps = idx / total_model_time if total_model_time > 0 else float('inf')
        print(f"Done. Model inference: {idx} frames in {total_model_time:.2f}s → {exec_fps:.2f} FPS")
        return {"optical_flows":optical_flows, "stat":{"exec_fps": exec_fps}}
        
if __name__ == "__main__":
    import sys
    import numpy as np
    import torch 
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    opt = OpticalFlowModel()
    outs = opt.predict(input_path=input_path, output_path=output_path, correspondence=False) 
    # nup = []
    # for x in outs["optical_flows"]:
    #     out_ = opt.viz(flo=torch.from_numpy(np.array(x)))
    #     nup.append(out_)
    
    # np.save("output/test.npy", np.array(nup))