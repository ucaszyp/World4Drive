import torch

@torch.no_grad()
def get_locations(features, stride, pad_h, pad_w):
        """
        Position embedding for image pixels.
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations
    
def get_locations_reso(h, w, device, stride, reso):
        """
        Position embedding for image pixels.
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """
        
        # 计算x方向的位移
        shifts_x = torch.arange(
            0, stride*w, step=stride // reso,
            dtype=torch.float32, device=device
        )
        # 计算y方向的位移
        shifts_y = torch.arange(
            0, h * stride, step=stride // reso,
            dtype=torch.float32, device=device
        ) 
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_y  = shift_y + stride // reso // 2
        shift_x  = shift_x + stride // reso // 2
        coord = torch.stack([shift_x, shift_y, torch.ones_like(shift_x)], dim=0).reshape(3, -1)
        # shift_x = shift_x.reshape(-1)
        # shift_y = shift_y.reshape(-1)
        # locations = torch.stack((shift_x, shift_y), dim=1)
        
        # locations = locations.reshape(h, w, 2)
        
        return coord