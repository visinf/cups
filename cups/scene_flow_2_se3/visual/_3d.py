import open3d as o3d

import cups.scene_flow_2_se3.visual._2d as o4visual2d


def visualize_se3s(se3s):
    # N x 4 x 4
    geometries = []

    # mesh = o3d.geometry.create_mesh_arrow(cylinder_radius=1.0, cone_radius=1.5, cylinder_height=5.0, cone_height=4.0,
    #                            resolution=20, cylinder_split=4, cone_split=1)

    se3s = se3s.detach().cpu().numpy()
    for i in range(len(se3s)):
        T = se3s[i]
        # T: 4x4
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05).transform(T)
        # x: red y: green z: blue
        geometries.append(mesh)

    o4visual2d.visualize_geometries3d(geometries, change_viewport=False)
