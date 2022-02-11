import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedFilter():
    def box_filter(self, x, r):
        channel =  x.shape[1] # Batch, Channel, H, W
        kernel_size = (2*r+1)
        weight = 1.0/(kernel_size**2)
        box_kernel = weight*torch.ones((channel, 1, kernel_size, kernel_size), dtype=torch.float32, device=x.device)
        output = F.conv2d(x, weight=box_kernel, stride=1, padding=r, groups=channel) #tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')

        return output


    def guided_filter(self, x, y, r, eps=1e-2):
        # Batch, Channel, H, W
        _, _, H, W = x.shape

        N = self.box_filter(torch.ones((1, 1, H, W), dtype=x.dtype, device=x.device), r)

        mean_x = self.box_filter(x, r) / N
        mean_y = self.box_filter(y, r) / N
        cov_xy = self.box_filter(x * y, r) / N - mean_x * mean_y
        var_x  = self.box_filter(x * x, r) / N - mean_x * mean_x

        A = cov_xy / (var_x + eps)
        b = mean_y - A * mean_x

        mean_A = self.box_filter(A, r) / N
        mean_b = self.box_filter(b, r) / N

        output = mean_A * x + mean_b
        return output

    def process(self, x: torch.Tensor, y: torch.Tensor, r, eps=1e-2):
        return self.guided_filter(x, y, r, eps)



if __name__ == "__main__":
    guided_filter = GuidedFilter()
    input = torch.randn(5,3,256,256)
    result = guided_filter.process(input, input, r=5)
    print(result.shape)