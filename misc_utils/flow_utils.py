'''
Usage:

from misc_utils.flow_utils import RAFTFlow, load_image_as_tensor, warp_image, MyRandomPerspective, generate_sample
image = load_image_as_tensor('hamburger_pic.jpeg', image_size)
flow_estimator = RAFTFlow()
res = generate_sample(
    image, 
    flow_estimator, 
    distortion_scale=distortion_scale,
)
f1 = res['input'][None]
f2 = res['target'][None]
flow = res['flow'][None]
f1_warp = warp_image(f1, flow)
show_image(f1_warp[0])
show_image(f2[0])
'''
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import numpy as np

def warp_image(image, flow, mode='bilinear'):
    """ Warp an image using optical flow.
    Args:
        image (torch.Tensor): Input image tensor with shape (N, C, H, W).
        flow (torch.Tensor): Optical flow tensor with shape (N, 2, H, W).
    Returns:
        warped_image (torch.Tensor): Warped image tensor with shape (N, C, H, W).
    """
    # check shape
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    if len(flow.shape) == 3:
        flow = flow.unsqueeze(0)
    if image.device != flow.device:
        flow = flow.to(image.device)
    assert image.shape[0] == flow.shape[0], f'Batch size of image and flow must be the same. Got {image.shape[0]} and {flow.shape[0]}.'
    assert image.shape[2:] == flow.shape[2:], f'Height and width of image and flow must be the same. Got {image.shape[2:]} and {flow.shape[2:]}.'
    # Generate a grid of sampling points
    grid = torch.tensor(
        np.array(np.meshgrid(range(image.shape[3]), range(image.shape[2]), indexing='xy')), 
        dtype=torch.float32, device=image.device
    )[None]
    grid = grid.permute(0, 2, 3, 1).repeat(image.shape[0], 1, 1, 1)  # (N, H, W, 2)
    grid += flow.permute(0, 2, 3, 1)  # add optical flow to grid

    # Normalize grid to [-1, 1]
    grid[:, :, :, 0] = 2 * (grid[:, :, :, 0] / (image.shape[3] - 1) - 0.5)
    grid[:, :, :, 1] = 2 * (grid[:, :, :, 1] / (image.shape[2] - 1) - 0.5)

    # Sample input image using the grid
    warped_image = F.grid_sample(image, grid, mode=mode, align_corners=True)

    return warped_image

def resize_flow(flow, size):
    """
    Resize optical flow tensor to a new size.

    Args:
        flow (torch.Tensor): Optical flow tensor with shape (B, 2, H, W).
        size (tuple[int, int]): Target size as a tuple (H, W).

    Returns:
        flow_resized (torch.Tensor): Resized optical flow tensor with shape (B, 2, H, W).
    """
    # Unpack the target size
    H, W = size

    # Compute the scaling factors
    h, w = flow.shape[2:]
    scale_x = W / w
    scale_y = H / h

    # Scale the optical flow by the resizing factors
    flow_scaled = flow.clone()
    flow_scaled[:, 0] *= scale_x
    flow_scaled[:, 1] *= scale_y

    # Resize the optical flow to the new size (H, W)
    flow_resized = F.interpolate(flow_scaled, size=(H, W), mode='bilinear', align_corners=False)

    return flow_resized

def check_consistency(flow1: torch.Tensor, flow2: torch.Tensor) -> torch.Tensor:
    """
    Check the consistency of two optical flows.
    flow1: (B, 2, H, W)
    flow2: (B, 2, H, W)
    if want the output to be forward flow, then flow1 is the forward flow and flow2 is the backward flow
    return: (H, W)
    """
    device = flow1.device
    height, width = flow1.shape[2:]

    kernel_x = torch.tensor([[0.5, 0, -0.5]]).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.tensor([[0.5], [0], [-0.5]]).unsqueeze(0).unsqueeze(0).to(device)
    grad_x = torch.nn.functional.conv2d(flow1[:, :1], kernel_x, padding=(0, 1))
    grad_y = torch.nn.functional.conv2d(flow1[:, 1:], kernel_y, padding=(1, 0))

    motion_edge = (grad_x * grad_x + grad_y * grad_y).sum(dim=1).squeeze(0)

    ax, ay = torch.meshgrid(torch.arange(width, device=device), torch.arange(height, device=device), indexing='xy')
    bx, by = ax + flow1[:, 0], ay + flow1[:, 1]

    x1, y1 = torch.floor(bx).long(), torch.floor(by).long()
    x2, y2 = x1 + 1, y1 + 1
    x1 = torch.clamp(x1, 0, width - 1)
    x2 = torch.clamp(x2, 0, width - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    y2 = torch.clamp(y2, 0, height - 1)

    alpha_x, alpha_y = bx - x1.float(), by - y1.float()

    a = (1.0 - alpha_x) * flow2[:, 0, y1, x1] + alpha_x * flow2[:, 0, y1, x2]
    b = (1.0 - alpha_x) * flow2[:, 0, y2, x1] + alpha_x * flow2[:, 0, y2, x2]
    u = (1.0 - alpha_y) * a + alpha_y * b

    a = (1.0 - alpha_x) * flow2[:, 1, y1, x1] + alpha_x * flow2[:, 1, y1, x2]
    b = (1.0 - alpha_x) * flow2[:, 1, y2, x1] + alpha_x * flow2[:, 1, y2, x2]
    v = (1.0 - alpha_y) * a + alpha_y * b

    cx, cy = bx + u, by + v
    u2, v2 = flow1[:, 0], flow1[:, 1]

    reliable = ((((cx - ax) ** 2 + (cy - ay) ** 2) < (0.01 * (u2 ** 2 + v2 ** 2 + u ** 2 + v ** 2) + 0.5)) & (motion_edge <= 0.01 * (u2 ** 2 + v2 ** 2) + 0.002)).float()

    return reliable # (B, 1, H, W)


class RAFTFlow(torch.nn.Module):
    '''
    # Instantiate the RAFTFlow class
    raft_flow = RAFTFlow(device='cuda')

    # Load a pair of image frames as PyTorch tensors
    img1 = torch.tensor(np.random.rand(3, 720, 1280), dtype=torch.float32)
    img2 = torch.tensor(np.random.rand(3, 720, 1280), dtype=torch.float32)

    # Compute optical flow between the two frames
    (optional) image_size = (256, 256) or None
    flow = raft_flow.compute_flow(img1, img2, image_size) # flow will be computed at the original image size if image_size is None
    # this flow can be used to warp the second image to the first image

    # Warp the second image using the flow
    warped_img = warp_image(img2, flow)
    '''
    def __init__(self, *args):
        """
        Args:
            device (str): Device to run the model on ("cpu" or "cuda").
        """
        super().__init__(*args)
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights, progress=False)
        self.model_transform = weights.transforms()

    def forward(self, img1, img2, img_size=None):
        """
        Compute optical flow between two frames using RAFT model.

        Args:
            img1 (torch.Tensor): First frame tensor with shape (B, C, H, W).
            img2 (torch.Tensor): Second frame tensor with shape (B, C, H, W).
            img_size (tuple): Size of the input images to be processed.

        Returns:
            flow (torch.Tensor): Optical flow tensor with shape (B, 2, H, W).
        """
        original_size = img1.shape[2:]
        # Preprocess the input frames
        if img_size is not None:
            img1 = TF.resize(img1, size=img_size, antialias=False)
            img2 = TF.resize(img2, size=img_size, antialias=False)

        img1, img2 = self.model_transform(img1, img2)

        # Compute the optical flow using the RAFT model
        with torch.no_grad():
            list_of_flows = self.model(img1, img2)
        flow = list_of_flows[-1]

        if img_size is not None:
            flow = resize_flow(flow, original_size)

        return flow
