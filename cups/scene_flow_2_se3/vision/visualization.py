import math
import os

import cv2
import matplotlib
import numpy as np
import open3d as o3d
import torch

import cups.scene_flow_2_se3.geometric.pinhole as o4geo_pinhole
import cups.scene_flow_2_se3.mask.rearrange as o4mask_rearr
import cups.scene_flow_2_se3.rearrange as ops_rearr
from cups.scene_flow_2_se3.vision.resize import resize


def create_vwriter(name, width, height, fps=10):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    vwriter = cv2.VideoWriter(
        os.path.join("videos", name + ".mp4"),
        fourcc,
        fps,
        (width, height),
    )
    return vwriter


def rgb_2_grayscale(x):
    # x: Bx3xHxW

    # rgb_weights: 3x1x1
    # https://en.wikipedia.org/wiki/Luma_%28video%29

    dtype = x.dtype
    device = x.device

    rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=dtype, device=device)

    rgb_weights = rgb_weights.view(-1, 1, 1)

    x = x * rgb_weights

    x = torch.sum(x, dim=1, keepdim=True)

    return x


# input: flow: torch.tensor 2xHxW
# output: flow_rgb: numpy.ndarray 3xHxW
def flow2rgb_old(flow, max_value=100):
    flow_map_np = flow.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float("nan")
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_map = rgb_map.clip(0, 1)
    rgb_map = torch.from_numpy(rgb_map)
    rgb_map = rgb_map.to(flow.device)
    return rgb_map


def flow2startendpoints(flow):
    _, H, W = flow.shape

    endpoints = o4geo_pinhole.oflow_2_pxl2d(flow.unsqueeze(0))[0]

    startpoints = o4geo_pinhole.oflow_2_pxl2d(0.0 * flow.unsqueeze(0))[0]

    return startpoints, endpoints


def depth2rgb(depth, rel=False):
    if len(depth.shape) == 2:
        H, W = depth.shape
    elif len(depth.shape) == 3:
        _, H, W = depth.shape
        depth = depth[0]

    device = depth.device
    dtype = depth.dtype

    np_depth = depth.detach().cpu().numpy()
    # cv2.COLORMAP_PLASMA, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO
    # depth range: 1/80 - 1/ 1e-3
    if not rel:
        np_depth = 1.0 / np_depth * 7
        np_depth = np.clip(np_depth, 0, 1)
    else:
        np_depth = np_depth - np_depth.min()
        np_depth = np_depth / (np.median(np_depth[np_depth != np.inf]) * 2)  # .max()
        np_depth = np.clip(np_depth, 0, 1)

    np_depth_rgb = cv2.applyColorMap((np_depth * 255.0).astype(np.uint8), cv2.COLORMAP_MAGMA) / 255.0

    depth_rgb = torch.from_numpy(np_depth_rgb).permute(2, 0, 1)
    depth_rgb = torch.flip(depth_rgb, dims=(0,))
    depth_rgb = depth_rgb.to(device)

    return depth_rgb


def get_colors(K, device=None, last_white=False, last_white_grey=False):

    if last_white_grey:
        K = K - 2
        color_grey = 0.7 * torch.ones(size=(1, 3))
        color_white = torch.ones(size=(1, 3))
    elif last_white:
        K = K - 1
        color_white = torch.ones(size=(1, 3))

    torch_colors = (
        torch.from_numpy(
            cv2.applyColorMap(
                tensor_to_cv_img((torch.arange(K).repeat(1, 1, 1).type(torch.float32) + 1.0) / (K + 1)),
                cv2.COLORMAP_JET,
            )
        )
        / 255.0
    )
    torch_colors = torch_colors[0]

    if last_white_grey:
        torch_colors = torch.cat((torch_colors, color_grey, color_white), dim=0)
    elif last_white:
        torch_colors = torch.cat((torch_colors, color_white), dim=0)

    if device is not None:
        torch_colors = torch_colors.to(device)
    # K x 3
    return torch_colors


def draw_pixels(img, pxls, colors=None, radius_in=3, radius_out=5):
    # pxls: K x 2
    K, _ = pxls.shape

    device = img.device
    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    if colors is None:
        colors = get_colors(K) * 0.0
    colors = (colors.detach().cpu().numpy() * 255).astype(np.uint8)

    # img = cv2.circle(img.copy(), (240, 240), 5, (255, 255, 255), -1)
    for k in range(K):
        cv2.circle(
            img,
            (int(pxls[k, 0].item()), int(pxls[k, 1].item())),
            radius_out,
            (
                colors[K - 1 - k, 0].item(),
                colors[K - 1 - k, 1].item(),
                colors[K - 1 - k, 2].item(),
            ),
            radius_in,
        )
        cv2.circle(
            img,
            (int(pxls[k, 0].item()), int(pxls[k, 1].item())),
            radius_in,
            (255, 255, 255),
            -1,
        )

    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device)
    # img = cv_img_to_tensor(img, device=device)
    return img


def mask2rgb(torch_mask, binary_mask=False, last_white=False, last_white_grey=False, img=None, colors=None):
    K, H, W = torch_mask.shape
    device = torch_mask.device
    torch_mask = torch_mask.type(torch.float32)

    if binary_mask:
        min_prob = 0.5  # 1.0 / K
        torch_mask = (torch_mask > min_prob) * 1.0

    if colors is None:
        torch_colors = get_colors(K, device=device, last_white=last_white, last_white_grey=last_white_grey)
    else:
        torch_colors = colors[:K]

    # Kx3x1x1 * Kx1xHxW
    torch_mask_rgb = torch.sum(
        torch_colors.unsqueeze(-1).unsqueeze(-1) * torch_mask.unsqueeze(1),
        dim=0,
    )
    # 3xHxW
    torch_mask_rgb = torch_mask_rgb / (
        torch.clamp(torch.max(torch_mask_rgb, dim=0, keepdim=True)[0], 1.0, np.inf) + 1e-7
    )

    if K > 1:
        # 3 x H x W
        for i in range(K):
            width = int(W / 20)
            height = int(H / K)
            torch_mask_rgb[:, i * height : (i + 1) * height - int(height / 10), :width] = (
                torch_colors[i].unsqueeze(-1).unsqueeze(-1)
            )

    torch_mask_rgb = torch_mask_rgb.to(device)

    if img is not None:
        torch_mask_rgb = 0.8 * torch_mask_rgb + 0.2 * img

    return torch_mask_rgb


def disp2rgb(disp, draw_arrows=False):

    _, H, W = disp.shape
    dtype = disp.dtype
    device = disp.device

    disp_rgb = depth2rgb(1.0 / disp, rel=True)
    if draw_arrows:
        disp_as_oflow = torch.zeros(size=(2, H, W), dtype=dtype, device=device)
        disp_as_oflow[:1] = disp
        start, end = flow2startendpoints(disp_as_oflow.clone())
        disp_rgb = draw_grid_arrows_in_rgb(disp_rgb, start, end)

    return disp_rgb
    # flow = torch.zeros(size=(2, H, W), dtype=dtype, device=device)
    # flow[0:1] = disp
    # return flow2rgb(flow, draw_arrows=draw_arrows)


"""
def disp2rgb(disp, vmax=80):
    disp_np = (disp).detach().cpu().numpy()

    if vmax is None:
        vmax = np.percentile(disp_np, 95)

    disp_np = disp_np / vmax

    disp_np = torch.from_numpy(disp_np).permute(1, 2, 0)

    disp_rgb = matplotlib.cm.get_cmap("magma")(disp_np)[:, :, 0, :3]

    disp_rgb = torch.from_numpy(disp_rgb).permute(2, 0, 1)

    disp_rgb = disp_rgb.to(disp.device)

    return disp_rgb
"""


def flow2rgb(flow_torch, draw_arrows=False, srcs_flow=None, size_wheel=70):
    # 2 x H x W or B x 2 x H x W
    dims_in = flow_torch.dim()
    if dims_in == 4:
        B = flow_torch.shape[0]
    else:
        B = 1

    flow_rgb = []
    for b in range(B):
        if dims_in == 4:
            b_flow_torch = flow_torch[b]
            if srcs_flow is not None:
                b_srcs_flow = srcs_flow[b]
            else:
                b_srcs_flow = None
        else:
            b_flow_torch = flow_torch
            b_srcs_flow = srcs_flow

        _, H, W = b_flow_torch.shape
        b_flow_torch_1 = b_flow_torch.clone()

        flow = b_flow_torch_1.detach().cpu().numpy()

        offset_x = W - 3 * int(size_wheel / 2)
        offset_y = H - 3 * int(size_wheel / 2)
        for y in range(size_wheel):
            for x in range(size_wheel):
                radius = ((y - size_wheel / 2.0) ** 2 + (x - size_wheel / 2.0) ** 2) ** 0.5
                if radius <= size_wheel / 2.0:
                    if offset_y + y > 0 and offset_y + y < H and offset_x + x < W and offset_x + x > 0:
                        flow[0, offset_y + y, offset_x + x] = x - size_wheel / 2.0
                        flow[1, offset_y + y, offset_x + x] = y - size_wheel / 2.0

        # 2 x H x W
        flow[0] = -flow[0]
        flow[1] = -flow[1]

        scaling = 50.0 / (H**2 + W**2) ** 0.5
        motion_angle = np.arctan2(flow[0], flow[1])
        motion_magnitude = (flow[0] ** 2 + flow[1] ** 2) ** 0.5
        flow_hsv = np.stack(
            [
                ((motion_angle / np.math.pi) + 1.0) / 2.0,
                np.clip(motion_magnitude * scaling, 0.0, 1.0),
                np.ones_like(motion_magnitude),
            ],
            axis=-1,
        )

        b_flow_rgb = matplotlib.colors.hsv_to_rgb(flow_hsv)
        """b_srcs_flow = b_srcs_flow.flatten(1).permute(1, 0) num_srcs = b_srcs_flow.shape[0] b_srcs_flow =
        b_srcs_flow.detach().cpu().numpy()

        for i in range(num_srcs):     flow_rgb = cv2.circle(flow_rgb, (b_srcs_flow[i, 0], b_srcs_flow[i, 1]), radius=0,
        color=(0, 0, 255), thickness=-1)
        """

        b_flow_rgb = torch.from_numpy(b_flow_rgb).permute(2, 0, 1)

        b_flow_rgb = b_flow_rgb.to(b_flow_torch.device)

        if b_srcs_flow is not None:
            start, end = flow2startendpoints(b_srcs_flow.clone())
            b_flow_rgb = draw_grid_arrows_in_rgb(b_flow_rgb, start, end)

        if draw_arrows:
            start, end = flow2startendpoints(b_flow_torch.clone())
            b_flow_rgb = draw_grid_arrows_in_rgb(b_flow_rgb, start, end)

        flow_rgb.append(b_flow_rgb)

    if dims_in == 4:
        flow_rgb = torch.stack(flow_rgb)
    else:
        flow_rgb = flow_rgb[0]
    return flow_rgb


def draw_arrows_in_rgb(img, start, end, colors=None, thickness=4):
    # start,end :  N x 2
    # colors: K x 3
    _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    N = start.shape[0]

    if colors is None:
        colors = get_colors(N) * 0.0
    colors = (colors.detach().cpu().numpy() * 255).astype(np.uint8)

    start[:, 0] = torch.clamp(start[:, 0], -W // 2, W + W // 2)
    start[:, 1] = torch.clamp(start[:, 1], -H // 2, H + H // 2)

    end[:, 0] = torch.clamp(end[:, 0], -W // 2, W + W // 2)
    end[:, 1] = torch.clamp(end[:, 1], -H // 2, H + H // 2)

    start = start.detach().cpu().numpy()
    end = end.detach().cpu().numpy()

    for i in range(N):
        if np.isnan(start[i]).any() or np.isnan(end[i]).any():
            continue
        cv2.arrowedLine(
            img,
            pt1=tuple(start[i].astype(int)),
            pt2=tuple(end[i].astype(int)),
            color=tuple([int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])]),  # (0, 255, 0),
            thickness=thickness,
            tipLength=0.5,
        )

    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device)
    return img


def draw_grid_arrows_in_rgb(img, start, end):
    # start , end : 2, H, W
    _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    start[0, :] = torch.clamp(start[0], -W // 2, W + W // 2)
    start[1, :] = torch.clamp(start[1], -H // 2, H + H // 2)

    end[0, :] = torch.clamp(end[0, :], -W // 2, W + W // 2)
    end[1, :] = torch.clamp(end[1, :], -H // 2, H + H // 2)

    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    threshold = 3.0

    start = start.clone().permute(1, 2, 0)
    end = end.clone().permute(1, 2, 0)
    # start end, HxWx2

    start = start.detach().cpu().numpy()
    end = end.detach().cpu().numpy()

    norm = np.linalg.norm(end - start, axis=2)
    norm[norm < threshold] = 0

    # Draw all the nonzero values
    nz = np.nonzero(norm)

    H, W = norm.shape
    # skip_amount = (len(nz[0]) // 100) + 1

    for h in range(0, H, H // 15 + 1):
        for w in range(0, W, W // 15 + 1):
            if norm[h, w] >= threshold:
                cv2.arrowedLine(
                    img,
                    pt1=tuple(start[h, w].astype(int)),
                    pt2=tuple(end[h, w].astype(int)),
                    color=(0, 255, 0),
                    thickness=2,
                    tipLength=1,
                )

    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device)

    return img


import matplotlib.pyplot as plt


def get_image_from_plot(fig, ax):
    ax.axis("off")
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image_from_plot


def visualize_flow(flow, draw_arrows=False, resize=True, duration=0):
    # x_in: 2xHxW

    if resize:
        _, H, W = flow.shape
        max_H = 720
        max_W = 1280
        scale_factor = min(max_H / H, max_W / W)
        flow = torch.nn.functional.interpolate(flow.unsqueeze(0), scale_factor=(scale_factor, scale_factor))[0]

    rgb = flow2rgb(flow, draw_arrows=draw_arrows)

    img = tensor_to_cv_img(rgb)

    cv2.imshow("flow", img)
    cv2.waitKey(duration)


def visualize_hist(x, split_dim=None, max=1e10):
    if split_dim == None:
        x = x.flatten().cpu().detach().numpy()
        x = x[x <= max]
    else:
        x = list(torch.split(x, 1, dim=split_dim))
        for i in range(len(x)):
            x[i] = x[i].flatten().cpu().detach().numpy()
            x[i] = x[i][x[i] <= max]

    plt.clf()
    plt.hist(x)
    plt.show(block=False)
    plt.pause(0.001)


def draw_text_in_rgb(img, text="title0"):
    # 3xHxW
    _, H, W = img.shape
    device = img.device
    dtype = img.dtype

    img = tensor_to_cv_img(img.clone())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (10, 50)  # left, top
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text, topLeftCornerOfText, font, fontScale, fontColor, lineType)

    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.to(device)

    return img


def visualize_data_comparison(data_1, outlier_1, data_2, outlier_2, data_gt, data_in):
    flow_vis = torch.cat(
        (
            flow2rgb(data_1["oflow"][0], draw_arrows=True),
            flow2rgb(data_2["oflow"][0], draw_arrows=True),
        ),
        dim=2,
    )
    flow_vis = draw_text_in_rgb(flow_vis, "flow A. vs B.")

    if "objs_labels" in data_in.keys():
        gt_objs_masks_rgb = mask2rgb(
            o4mask_rearr.label2unique2onehot(data_in["objs_labels"])[0],
            img=data_in["rgb_l_01"][0, :3],
        )
    else:
        gt_objs_masks_rgb = data_in["rgb_l_01"][0, :3]
    mask_vis = torch.cat(
        (
            gt_objs_masks_rgb,
            mask2rgb(data_2["objs_masks"], img=data_in["rgb_l_01"][0, :3]),
        ),
        dim=2,
    )
    mask_vis = draw_text_in_rgb(mask_vis, "segmentation A. vs B.")

    # V* ~nn_disp1_outlier_pxl
    outlier_vis = torch.cat(
        (
            mask2rgb((outlier_1["sflow"])[0], img=data_in["rgb_l_01"][0, :3]),
            mask2rgb((outlier_2["sflow"])[0], img=data_in["rgb_l_01"][0, :3]),
        ),
        dim=2,
    )
    outlier_vis = draw_text_in_rgb(outlier_vis, "outlier-sflow A. vs. B.")

    flow_dev_vis = torch.cat(
        (
            depth2rgb(data_2["pt3d_f0_1"][0, 2:], rel=True),
            flow2rgb(
                data_2["oflow"][0] * data_gt["oflow_valid"][0] - data_gt["oflow"][0],
                draw_arrows=True,
            ),
        ),
        dim=2,
    )
    flow_dev_vis = draw_text_in_rgb(flow_dev_vis, "flow (B-GT)")

    disp_dev_vis = torch.cat(
        (
            disp2rgb((data_2["disp_0"][0] - data_gt["disp_0"][0]) * data_gt["disp_valid_0"][0], draw_arrows=False),
            disp2rgb(
                (data_2["disp_f0_1"][0] - data_gt["disp_f0_1"][0]) * data_gt["disp_valid_f0_1"][0],
                draw_arrows=False,
            ),
        ),
        dim=2,
    )
    disp_dev_vis = draw_text_in_rgb(disp_dev_vis, "disp (B-GT)")

    # data_pred_se3['disp_f0_1'], gt_disps2, gt_masks_disps2_valid
    # mask_vis = torch.cat((o4vis.mask2rgb(o4mask_rearr.label2unique2onehot(gt_labels_objs)[0]), o4vis.mask2rgb(o4mask_rearr.label2unique2onehot(labels_objs)[0], last_white_grey=True)), dim=2)
    visualize_img(
        torch.cat((flow_vis, mask_vis, outlier_vis, flow_dev_vis, disp_dev_vis), dim=1), width=None, height=720
    )


def visualize_data(data, height=None, width=None):

    rgbs = []

    num_rgbs = 0
    for key, val in data.items():
        print(key)

        if key.startswith("rgb"):
            B = val.shape[0]
            C = val.shape[1]
            for b in range(B):
                for c in range(C // 3):
                    rgbs.append(val[b, c * 3 : (c + 1) * 3])
        elif "valid" in key or "inbound" in key or "oflow_occ" in key or "disp_occ" in key:
            B = val.shape[0]
            for b in range(B):
                rgbs.append(mask2rgb(val[b]))
        elif key.startswith("pt3d"):
            B = val.shape[0]
            for b in range(B):
                # visualize_pts3d(val[b], change_viewport=True, return_img=False)
                rgbs.append(visualize_pts3d(val[b], change_viewport=True, return_img=True))
        elif key.startswith("depth"):
            B = val.shape[0]
            for b in range(B):
                rgbs.append(depth2rgb(val[b], rel=True))
        elif key.startswith("disp"):
            B = val.shape[0]
            for b in range(B):
                rgbs.append(disp2rgb(val[b]))
        elif key.startswith("oflow"):
            B = val.shape[0]
            for b in range(B):
                rgbs.append(flow2rgb(val[b]))

        elif key.startswith("objs_labels"):
            rgbs.append(mask2rgb(o4mask_rearr.label2unique2onehot(val)[0]))

        for i in range(num_rgbs, len(rgbs)):
            rgbs[i] = draw_text_in_rgb(rgbs[i], key)
        num_rgbs = len(rgbs)
    K = len(rgbs)
    H = 0
    W = 0
    for k in range(K):
        _, H_k, W_k = rgbs[k].shape
        if H_k > H:
            H = H_k
        if W_k > W:
            W = W_k
    tensor_rgbs = torch.zeros(size=(K, 3, H, W), dtype=rgbs[0].dtype, device=rgbs[0].device)
    for k in range(K):
        _, H_k, W_k = rgbs[k].shape
        tensor_rgbs[k, :, :H_k, :W_k] = rgbs[k]
    visualize_imgs(tensor_rgbs, height=height, width=width)


def visualize_imgs(
    rgb,
    duration=0,
    vwriter=None,
    fpath=None,
    height=None,
    width=None,
    mask_overlay=None,
):
    # rgb: K x 3 x H x W

    rgb = (rgb * 1.0).clamp(0, 1)
    if mask_overlay is not None:
        rgb = (rgb + mask_overlay) / 2.0

    rgb = torch.nn.functional.pad(rgb, (1, 1, 1, 1), "constant", 1.0)
    margin = 2
    torch.nn.functional.pad(rgb, (1, 1), "constant", 0)

    K, C, H, W = rgb.shape

    prop_w = 4
    prop_h = 3
    grid_width = math.ceil(math.sqrt((K * prop_w**2) / prop_h**2))
    grid_height = math.ceil((grid_width * prop_h) / prop_w)

    grid_height = math.ceil(K / grid_width)
    grid_total = grid_height * grid_width

    img_placeholder = torch.zeros_like(rgb[:1]).repeat(grid_total - K, 1, 1, 1)

    rgb = torch.cat((rgb, img_placeholder), dim=0)

    rgb = rgb.reshape(grid_height, grid_width, C, H, W).permute(2, 0, 3, 1, 4)

    rgb = rgb.reshape(C, grid_height * H, grid_width * W)

    visualize_img(rgb, duration, vwriter, fpath, height, width)


def visualize_img(rgb, duration=0, vwriter=None, fpath=None, height=None, width=None):
    # img: 3xHxW
    rgb = rgb.clone()

    if width is not None:
        orig_width = rgb.size(2)
        scale_factor = width / orig_width

    elif height is not None:
        orig_height = rgb.size(1)
        scale_factor = height / orig_height

    if width or height is not None:
        rgb = resize(
            rgb[None,],
            scale_factor=scale_factor,
        )[0]

    img = tensor_to_cv_img(rgb)

    if vwriter is not None:
        vwriter.write(img)

    if fpath is not None:
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        cv2.imwrite(fpath, img)
    else:
        cv2.imshow("img", img)
        cv2.waitKey(duration)


def visualize_imgpair(imgpair):
    # imgpair: 6xHxW
    img1 = imgpair[:3]
    img2 = imgpair[3:]
    img = torch.cat((img1, img2), dim=2)

    visualize_img(img)


def tensor_to_cv_img(x_in):
    # x_in : CxHxW float32
    # x_out : HxWxC uint8
    x_in = x_in * 1.0
    x_in = torch.clamp(x_in, min=0.0, max=1.0)
    x_out = (x_in.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    x_out = x_out[:, :, ::-1]
    return x_out


def cv_img_to_tensor(x_in, device=None):
    # x_in : HxWxC uint 8
    x_out = x_in[:, :, ::-1] / 255.0
    x_out = torch.from_numpy(x_out).permute(2, 0, 1)

    if device is not None:
        x_out = x_out.to(device)

    return x_out


def render_pts3d(pts):
    pass
    # pcd = o3d.io.read_point_cloud("pcds/entire_hall_3d_cropped_binary.pcd")
    # downpcd.paint_uniform_color(np.zeros([3, 1]))
    # o3d.visualization.draw_geometries([downpcd])


"""
def visualize_pts3d(pts, img=None, change_viewport=True):
    dims = pts.dim()

    if dims == 4:
        _, _, H, W = pts.shape
        pts = pts.permute(1, 0, 2, 3).flatten(1)
        pts = pts.permute(1, 0)
        if img is not None:
            img = img.permute(1, 0, 2, 3).flatten(1).permute(1, 0)

    elif dims == 3:
        _, H, W = pts.shape
        # 3 x H x W
        pts = pts.flatten(1)
        pts = pts.permute(1, 0)
        if img is not None:
            img = img.flatten(1).permute(1, 0)
    elif dims == 2:
        W = 1280
        H = 720
        img = None
        # expects to be Nx3 then
        pass
    else:
        print("error: input for scene flow visualization must be 2D,3D or 4D.")
        return 0

    N, _ = pts.shape
    N = int(N / 2)
    pts = pts.detach().cpu().numpy()
    pairs = torch.arange(N).repeat(2, 1)
    pairs[1:2, :] += N
    pairs = pairs.permute(1, 0)
    pairs = pairs.detach().cpu().numpy()
    # N x 2
    pts = np.expand_dims(pts, axis=2)
    pairs = np.expand_dims(pairs, axis=2)


    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if img == None:
        pass
        #colors = [[0, 0.5, 0] for i in range(len(pairs))]
        #pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Nx3
        img = img.detach().cpu().numpy() * 0.5
        img = np.expand_dims(img, axis=2)
        colors = img
        pcd.colors = o3d.utility.Vector3dVector(colors)


    visualize_geometries3d([pcd], change_viewport=change_viewport)
"""
from PIL import Image


def save_gif(imgs, fpath):
    if not os.path.exists(os.path.dirname(fpath)):
        os.makedirs(os.path.dirname(fpath))
    for img_id in range(len(imgs)):
        imgs[img_id] = Image.fromarray(np.uint8(imgs[img_id].permute(1, 2, 0).detach().cpu().numpy() * 255))
    imgs[0].save(fpath, format="GIF", append_images=imgs, save_all=True, duration=100, loop=0)


def visualize_geometries3d(
    geometries, change_viewport=True, W=640, H=480, fpath=None, return_img=False, visualize_rot_x=False
):
    up = np.expand_dims(np.array([0.0, -1.0, -0.0], dtype=float), axis=1)
    # looking direction
    lookat = np.expand_dims(np.array([0.0, 0.0, 0.1], dtype=float), axis=1)

    # pos (x, y, z)
    front = np.expand_dims(np.array([1, 0, -20], dtype=float), axis=1)

    zoom = 0.1
    vis = o3d.visualization.Visualizer()
    visible = True
    # if fpath is not None or return_img:
    #    visible = False
    vis.create_window(visible=visible, width=W, height=H)

    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.poll_events()
    vis.update_renderer()

    view_control = vis.get_view_control()
    if change_viewport:
        view_control.set_lookat(lookat)
        view_control.set_up(up)
        view_control.set_front(front)
        view_control.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    if fpath is not None or return_img:
        if visualize_rot_x:
            imgs = []
            for i in range(100):
                img = vis.capture_screen_float_buffer(do_render=True)
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
                imgs.append(img)
                view_control.rotate(10.0, 0.0)
                vis.update_renderer()
            if fpath is not None:
                save_gif(imgs, fpath)
            else:
                vis.destroy_window()
                return imgs
        else:
            img = vis.capture_screen_float_buffer(do_render=True)
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            if fpath is not None:
                visualize_img(img, fpath=fpath, duration=0)
            elif return_img:
                vis.destroy_window()
                return img
    else:
        vis.run()
    vis.destroy_window()


def visualize_pca(pts, delay=0):
    # pts: N x D
    from sklearn.decomposition import PCA

    pts = pts.detach().cpu().numpy()

    pca = PCA(n_components=2)
    pts2d = pca.fit_transform(pts)

    plt.scatter(pts2d[:, 0], pts2d[:, 1])
    plt.show(block=False)
    plt.waitforbuttonpress(1)
    plt.clf()


def visualize_sflow(pts1, pts2, img=None):
    dims = pts1.dim()

    if dims == 4:
        _, _, H, W = pts1.shape
        pts = torch.cat(
            (pts1.permute(1, 0, 2, 3).flatten(1), pts2.permute(1, 0, 2, 3).flatten(1)),
            dim=1,
        )
        pts = pts.permute(1, 0)
        if img is not None:
            img = img.permute(1, 0, 2, 3).flatten(1).permute(1, 0)

    elif dims == 3:
        _, H, W = pts1.shape
        # 3 x H x W
        pts = torch.cat(
            (pts1.flatten(1), pts2.flatten(1)),
            dim=1,
        )
        pts = pts.permute(1, 0)
        if img is not None:
            img = img.flatten(1).permute(1, 0)
    elif dims == 2:
        H = 256
        W = 832
        pts = torch.cat(
            (pts1, pts2),
            dim=0,
        )
        # expects to be Nx3 then
        pass
    else:
        print("error: input for scene flow visualization must be 2D,3D or 4D.")
        return 0
    # pts: N x 3
    N, _ = pts.shape
    N = int(N / 2)
    pts = pts.detach().cpu().numpy()
    pairs = torch.arange(N).repeat(2, 1)
    pairs[1:2, :] += N
    pairs = pairs.permute(1, 0)
    pairs = pairs.detach().cpu().numpy()
    # N x 2
    pts = np.expand_dims(pts, axis=2)
    pairs = np.expand_dims(pairs, axis=2)
    if img == None:
        colors = [[0, 0.5, 0] for i in range(len(pairs))]
    else:
        # Nx3
        img = img.detach().cpu().numpy() * 0.5
        img = np.expand_dims(img, axis=2)
        colors = img

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(pairs),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    up = np.expand_dims(np.array([0.0, -1.0, 0.0], dtype=np.float), axis=1)
    # looking direction
    lookat = np.expand_dims(np.array([0.0, 0.0, 10.0], dtype=np.float), axis=1)
    # pos
    front = np.expand_dims(np.array([0.0, -0.2, -1.1], dtype=np.float), axis=1)
    zoom = 0.0000000001
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=W, height=H)
    vis.add_geometry(line_set)
    # o3d.visualization.draw_geometries(
    #    [line_set], lookat=lookat, up=up, front=front, zoom=0.01
    # )
    vis.poll_events()
    vis.update_renderer()

    view_control = vis.get_view_control()
    view_control.set_lookat(lookat)
    view_control.set_up(up)
    view_control.set_front(front)
    view_control.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    # vis.run()
    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()
    return img


def visualize_pts3d(pts3d, change_viewport=False, fpath=None, return_img=False, visualize_rot_x=False, img=None):
    # K-times  3 x H x W / 3 x N
    # 1-time  3 x H x  W
    # pts3d_backup = []
    # for el in pts3d:
    #    pts3d_backup.append(el.clone())

    input_dev = pts3d[0].device

    if not isinstance(pts3d, list):
        pts3d = [pts3d]

    for id in range(len(pts3d)):
        pts3d[id] = pts3d[id].detach().cpu()

    K = len(pts3d)
    if len(pts3d[0].shape) == 3:
        _, H, W = pts3d[0].shape
    else:
        H = 480
        W = 640

    geometries = []
    colors = get_colors(K)
    for k in range(K):
        k_pts3d = pts3d[k].flatten(1)
        # 3 x N
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(k_pts3d.detach().cpu().numpy().T)
        if img is not None:
            img = (img.flatten(1) + 0.5) / 2.0
            colors = img.detach().cpu().numpy().T
            geometry.colors = o3d.utility.Vector3dVector(colors)
        N = k_pts3d.shape[1]

        if K > 1:
            geometry.colors = o3d.utility.Vector3dVector(colors[k].repeat(N, 1).detach().cpu().numpy())

        geometries.append(geometry)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    geometries.append(mesh_frame)
    # red, green, blue = x, y, z

    if return_img:
        img = visualize_geometries3d(
            geometries,
            H=H,
            W=W,
            change_viewport=change_viewport,
            fpath=fpath,
            return_img=return_img,
            visualize_rot_x=visualize_rot_x,
        )
        return img.to(input_dev)
    else:
        visualize_geometries3d(
            geometries,
            H=H,
            W=W,
            change_viewport=change_viewport,
            fpath=fpath,
            return_img=return_img,
            visualize_rot_x=visualize_rot_x,
        )


def visualize_boxes3d(boxes3d):
    B, K, _ = boxes3d["center"].shape
    # boxes3d['axes'] = boxes3d['axes'].permute(0, 1, 3, 2)
    geometries = []
    colors = get_colors(K)
    for b in range(B):
        for k in range(K):
            # geometry = o3d.geometry.OrientedBoundingBox.create_from_points(
            #    o3d.utility.Vector3dVector(boxes3d['pts'][b][k].detach().cpu().numpy().T)
            # )
            geometry = o3d.geometry.OrientedBoundingBox(
                center=boxes3d["center"][b, k, :, None].detach().cpu().numpy(),  # 3x1
                R=boxes3d["axes"][b, k, :, :].detach().cpu().numpy(),  # 3x3
                extent=boxes3d["measures"][b, k, :, None].detach().cpu().numpy(),  # 3x1
            )
            geometries.append(geometry)

            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(boxes3d["pts"][b][k].detach().cpu().numpy().T)
            N = boxes3d["pts"][b][k].shape[1]
            geometry.colors = o3d.utility.Vector3dVector(colors[k].repeat(N, 1).detach().cpu().numpy())

            geometries.append(geometry)

    visualize_geometries3d(geometries)


def choose_from_neighborhood(x, patch_size):
    x = x.permute(1, 0, 2, 3)
    x = ops_rearr.neighbors_to_channels(x * 1.0, patch_size)
    avg = torch.mean(x, dim=1, keepdim=True)
    id = torch.argmax(avg, dim=0, keepdim=True)
    return o4mask_rearr.label2onehot(id)


class PhotoTransform:
    def __init__(self, device=None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def __call__(self, imgpair):
        imgpair = self.random_swap_channels(imgpair)
        imgpair = self.random_adjust_hue(imgpair)

        return imgpair

    def random_swap_channels(self, imgpair):
        # imgpair: 6 x H x W
        _, H, W = imgpair.shape

        channel_indices = torch.arange(3)
        # 3
        offset = torch.randint(low=0, high=3, size=(1,))
        reverse = torch.randint(low=0, high=2, size=(1,))
        # 1
        channel_indices = ((3**reverse) - 1) + ((channel_indices + offset) % 3) * ((-1) ** reverse)
        # 3
        channel_indices = torch.cat((channel_indices, channel_indices + 3))
        imgpair = imgpair[channel_indices]
        # 6 x H x W

        return imgpair

    def random_adjust_hue(self, imgpair):

        device = imgpair.device

        imgpair[:3] = kornia.color.rgb_to_hsv(imgpair[:3])
        imgpair[3:] = kornia.color.rgb_to_hsv(imgpair[3:])

        hue = torch.rand(size=(1,), device=device) - 0.5

        imgpair[2] = imgpair[2] + hue
        imgpair[5] = imgpair[5] + hue

        imgpair[:3] = kornia.color.hsv_to_rgb(imgpair[:3])
        imgpair[3:] = kornia.color.hsv_to_rgb(imgpair[3:])

        imgpair = torch.clamp(input=imgpair, min=0.0, max=1.0)

        return imgpair
