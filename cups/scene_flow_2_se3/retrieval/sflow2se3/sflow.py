import cups.scene_flow_2_se3.visual._2d as o4visual2d


class SFlow:
    """Describes scene flow and additional information in form BxCxHxW or CxHxW."""

    def __init__(self, data_sflow: dict = None, args: dict = None):
        """Creates SFlow object from dictionary.

        data_sflow dict: dictionary which contains information for scene flow
        """

        if data_sflow is not None:
            self.rgb = data_sflow["rgb_l_01"][0, :3]
            self.pt3d_0 = data_sflow["pt3d_0"][0]
            self.pt3d_1 = data_sflow["pt3d_f0_1"][0]
            self.oflow = data_sflow["oflow"][0]
            self.sflow = self.pt3d_1 - self.pt3d_0

            if self.pt3d_0.dim() == 3:
                self.dim = 3
                self.H = self.pt3d_0.size(1)
                self.W = self.pt3d_0.size(2)
            elif self.pt3d_0.dim() == 4:
                self.dim = 4
                self.H = self.pt3d_0.size(2)
                self.W = self.pt3d_0.size(3)
            else:
                print("error : unknown number of dim for sflow : ", self.pt3d_0.dim())

            self.depth_reliable_0 = data_sflow["pt3d_valid_0"][0]
            self.depth_reliable_1 = data_sflow["pt3d_valid_f0_1"][0]
            # self.depth_reliable_0[:, : int(self.H * (1.0 - args.sflow2se3_pt3d_valid_req_pxl_lower_than_h)), :] = False
            # self.depth_reliable_1[:, : int(self.H * (1.0 - args.sflow2se3_pt3d_valid_req_pxl_lower_than_h)), :] = False
            # self.depth_reliable_0[..., int(self.W * (1.0 - args.sflow2se3_pt3d_valid_req_pxl_lower_than_w)):] = False
            # self.depth_reliable_1[..., int(self.W * (1.0 - args.sflow2se3_pt3d_valid_req_pxl_lower_than_w)):] = False
            # self.depth_reliable_0[..., :int(self.W * (args.sflow2se3_pt3d_valid_req_pxl_lower_than_w))] = False
            # self.depth_reliable_1[..., :int(self.W * (args.sflow2se3_pt3d_valid_req_pxl_lower_than_w))] = False
            self.depth_reliable_01 = self.depth_reliable_0 * self.depth_reliable_1

            self.resize_mode = "nearest_v2"  # None
            self.cam_ext = None  # data_sflow['eval_camera_extrinsics']
            self.cam_int = data_sflow["projection_matrix"][0]
            self.cam_baseline = data_sflow["baseline"][0]
            self.cam_H = self.H
            self.cam_W = self.W

        if args is not None:
            self.std_disp_temp = args.sflow2se3_model_se3_likelihood_disp_abs_std
            self.std_oflow_x = args.sflow2se3_model_se3_likelihood_oflow_abs_std[0]
            self.std_oflow_y = args.sflow2se3_model_se3_likelihood_oflow_abs_std[1]
            self.std_nn_x = args.sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std
            self.std_nn_y = args.sflow2se3_model_euclidean_nn_uv_dev_rel_to_width_std
            self.std_nn_z = args.sflow2se3_model_euclidean_nn_rel_depth_dev_std
            self.inlier_hard_thresh = args.sflow2se3_model_inlier_hard_threshold

    def resizeToNewObject(
        self, H_out: int = None, W_out: int = None, scale_factor: float = None, mode: str = "bilinear"
    ):
        """Creates new object by resizing the current information.

        Parameters
        ----------
        self SFlow: current SFlow object
        H_out int: height of target resolution
        W_out int: width of target resolution
        scale_factor float: relative target resolution if (H_out, W_out) are not specified
        mode str: mode for how to resample, e.g. 'bilinear', 'nearest' or 'nearest_v2'
        """
        sflow_new = SFlow(data_sflow=None)
        sflow_new.rgb = self.rgb
        sflow_new.pt3d_0 = o4visual2d.resize(
            self.pt3d_0, H_out=H_out, W_out=W_out, scale_factor=scale_factor, mode=mode
        )
        sflow_new.pt3d_1 = o4visual2d.resize(
            self.pt3d_1, H_out=H_out, W_out=W_out, scale_factor=scale_factor, mode=mode
        )
        sflow_new.oflow = o4visual2d.resize(self.oflow, H_out=H_out, W_out=W_out, scale_factor=scale_factor, mode=mode)
        sflow_new.sflow = sflow_new.pt3d_1 - sflow_new.pt3d_0
        sflow_new.depth_reliable_0 = o4visual2d.resize(
            self.depth_reliable_0, H_out=H_out, W_out=W_out, scale_factor=scale_factor, mode=mode
        )
        sflow_new.depth_reliable_1 = o4visual2d.resize(
            self.depth_reliable_1, H_out=H_out, W_out=W_out, scale_factor=scale_factor, mode=mode
        )
        sflow_new.depth_reliable_01 = sflow_new.depth_reliable_0 * sflow_new.depth_reliable_1
        if sflow_new.pt3d_0.dim() == 3:
            sflow_new.dim = 3
            sflow_new.H = sflow_new.pt3d_0.size(1)
            sflow_new.W = sflow_new.pt3d_0.size(2)
        elif sflow_new.pt3d_0.dim() == 4:
            sflow_new.dim = 4
            sflow_new.H = sflow_new.pt3d_0.size(2)
            sflow_new.W = sflow_new.pt3d_0.size(3)
        else:
            print("error : unknown number of dim for sflow : ", sflow_new.pt3d_0.dim())

        sflow_new.cam_ext = self.cam_ext
        sflow_new.cam_int = self.cam_int
        sflow_new.cam_H = self.cam_H
        sflow_new.cam_W = self.cam_W
        sflow_new.cam_baseline = self.cam_baseline
        sflow_new.resize_mode = mode

        sflow_new.std_disp_temp = self.std_disp_temp
        sflow_new.std_oflow_x = self.std_oflow_x
        sflow_new.std_oflow_y = self.std_oflow_y
        sflow_new.inlier_hard_thresh = self.inlier_hard_thresh
        sflow_new.std_nn_x = self.std_nn_x
        sflow_new.std_nn_y = self.std_nn_y
        sflow_new.std_nn_z = self.std_nn_z
        return sflow_new
