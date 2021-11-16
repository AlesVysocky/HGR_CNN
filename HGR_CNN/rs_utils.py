import pyrealsense2 as rs

class RSUtils:
    def __init__(self,intrinsics,extrot,extrans):
        self.intrinsics = intrinsics
        self.extrinsics = rs.extrinsics()
        self.extrinsics.rotation = extrot
        self.extrinsics.translation = extrans
    
    def transform_px_to_pt(self,px,depth):
        return rs.rs2_deproject_pixel_to_point(self.intrinsics,px,depth)

    def transform_pt_to_base(self,pt):
        return rs.rs2_transform_point_to_point(self.extrinsics,pt)