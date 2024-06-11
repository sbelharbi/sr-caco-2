# https://www.jianshu.com/p/8c9e9e57d48e

@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel

def GaussianBlur(batch_img, ksize, sigma=None):
    kernel = getGaussianKernel(ksize, sigma) # 生成权重
    B, C, H, W = batch_img.shape # C：图像通道数，group convolution 要用到
    # 生成 group convolution 的卷积核
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2 # 保持卷积前后图像尺寸不变
    # mode=relfect 更适合计算边缘像素的权重
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None,
                           stride=1, padding=0, groups=C)
    return weighted_pix


def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    # 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    # 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    # 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2),
                                                      keepdim=True)

    # 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(
        weights_color)

    # 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
    # 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
    # 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix
