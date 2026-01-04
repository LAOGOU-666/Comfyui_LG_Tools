import torch
import numpy as np
from PIL import Image, ImageOps
import cv2

class MaskBoxCropNodeV2:
    """
    根据mask裁剪图像并调整大小
    """
    
    CATEGORY = "AFL/Mask"
    DESCRIPTION = "根据mask裁剪图像区域并调整大小"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "resize_mode": (["NaN", "lanczos", "nearest-exact", "bilinear", "bicubic"], {"default": "lanczos"}),
            },
            "optional": {
                "Box_grow_factor": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 5.0, "step": 0.05, "tooltip": "裁剪区域的扩展倍数，1.0表示不扩展，大于1.0表示按比例扩大"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "目标图像的百万像素数，以1024*1024为1百万像素基准"}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1, "tooltip": "目标分辨率必须被此数字整除"}),
                "ratio": (["auto", "1:1", "4:3", "3:4", "16:9", "9:16"], {"default": "auto", "tooltip": "裁剪比例模式，auto为自动检测最接近比例"}),
                "startup_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "当mask的box面积与输入图像的面积占比达到此阈值时，跳过ratio和box_grow_factor判断"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "CROPBOX", "MASK")
    RETURN_NAMES = ("cropped_image", "crop_box", "cropped_mask")
    FUNCTION = "crop_and_resize"

    def _tensor_to_pil(self, tensor):
        """将ComfyUI的tensor转换为PIL图像"""
        # 确保输入tensor是正确的数据类型和形状
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        # 处理不同的tensor形状
        if len(tensor.shape) == 4:
            # 标准的4D tensor (batch, height, width, channels)
            img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(tensor.shape) == 3:
            # 3D tensor (height, width, channels)
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
            
        return Image.fromarray(img_np)
    
    def _tensor_to_pil_mask(self, mask):
        """将ComfyUI的mask tensor转换为PIL图像"""
        # 确保输入mask是正确的数据类型
        if mask.dtype != torch.float32:
            mask = mask.float()
            
        # 处理不同的mask形状
        if len(mask.shape) == 4:
            # 标准的4D mask tensor (batch, height, width, channels)
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 3:
            # 3D mask tensor (batch, height, width) 或 (height, width, channels)
            if mask.shape[0] == 1:  # (1, height, width)
                mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            else:  # (height, width, channels) 或其他情况
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 2:
            # 2D mask tensor (height, width)
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
        # 确保mask_np是2D数组
        if len(mask_np.shape) > 2:
            # 如果是3D数组，取第一个通道
            mask_np = mask_np[:, :, 0] if mask_np.shape[2] > 1 else mask_np[:, :, 0]
            
        return Image.fromarray(mask_np, mode='L')

    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为ComfyUI的tensor"""
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None,]
    
    def _pil_to_mask(self, pil_mask):
        """将PIL mask转换为ComfyUI的mask tensor"""
        mask_np = np.array(pil_mask).astype(np.float32) / 255.0
        return torch.from_numpy(mask_np)[None,]
    
    def _find_best_aspect_ratio(self, width, height):
        """根据输入宽高找到最接近的预设宽高比"""
        # 预设的宽高比列表 (width, height)
        aspect_ratios = [(1, 1), (4, 3), (3, 4), (16, 9), (9, 16)]
        
        # 计算输入尺寸的宽高比
        input_ratio = width / height
        
        # 找出最接近的宽高比
        best_ratio = aspect_ratios[0]
        min_diff = float('inf')
        
        for ratio in aspect_ratios:
            ratio_value = ratio[0] / ratio[1]
            diff = abs(input_ratio - ratio_value)
            if diff < min_diff:
                min_diff = diff
                best_ratio = ratio
        
        return best_ratio
    
    def _calculate_target_dimensions(self, megapixels, aspect_ratio, divisible_by=1):
        """根据百万像素数、宽高比和可整除要求计算目标尺寸"""
        # 1024*1024 = 约1百万像素
        total_pixels = megapixels * 1024 * 1024
        
        # 根据宽高比计算目标宽度和高度
        width_ratio, height_ratio = aspect_ratio
        aspect_ratio_value = width_ratio / height_ratio
        
        target_height = int((total_pixels / aspect_ratio_value) ** 0.5)
        target_width = int(target_height * aspect_ratio_value)
        
        # 确保尺寸能被指定数字整除
        if divisible_by > 1:
            # 计算最接近但不小于原尺寸的可整数值
            target_width = ((target_width + divisible_by - 1) // divisible_by) * divisible_by
            target_height = ((target_height + divisible_by - 1) // divisible_by) * divisible_by
        elif divisible_by == 1:
            # 保持原有逻辑，确保尺寸为偶数
            target_width = target_width + 1 if target_width % 2 != 0 else target_width
            target_height = target_height + 1 if target_height % 2 != 0 else target_height
        
        return (target_width, target_height)
    
    def crop_and_resize(self, image, mask, resize_mode, Box_grow_factor=1.0, megapixels=1.0, divisible_by=1, ratio="auto", startup_threshold=0.4):
        # 将输入转换为PIL图像
        pil_image = self._tensor_to_pil(image)
        pil_mask = self._tensor_to_pil_mask(mask)
        
        # 确保mask是二值图像
        pil_mask = pil_mask.convert('L')
        
        # 获取mask的边界框
        bbox = pil_mask.getbbox()
        if bbox is None:
            # 如果没有找到mask，返回整个图像
            bbox = (0, 0, pil_image.width, pil_image.height)
        
        # 紧密贴合mask边缘计算裁剪区域
        x1, y1, x2, y2 = bbox
        
        # 计算边界框的宽度和高度
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 计算mask的box面积与输入图像面积的比例
        image_area = pil_image.width * pil_image.height
        bbox_area = bbox_width * bbox_height
        area_ratio = bbox_area / image_area
        
        # 判断是否需要跳过ratio和box_grow_factor判断
        skip_ratio_and_grow = area_ratio >= startup_threshold
        
        if skip_ratio_and_grow:
            # 当mask的box面积占比超过阈值时，不启动ratio判断，也不启用box_grow_factor判断，返回整个原始图像
            crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, pil_image.width, pil_image.height
            # 使用原始图像的宽高比
            best_aspect_ratio = (pil_image.width, pil_image.height)
        else:
            # 找到宽高比
            if ratio != "auto":
                # 使用指定的比例
                width_ratio, height_ratio = map(int, ratio.split(":"))
                best_aspect_ratio = (width_ratio, height_ratio)
            else:
                # 找到最接近的预设宽高比
                best_aspect_ratio = self._find_best_aspect_ratio(bbox_width, bbox_height)
            width_ratio, height_ratio = best_aspect_ratio
            
            # 根据宽高比和Box_grow_factor计算目标尺寸
            if width_ratio >= height_ratio:  # 横向或正方形
                base_size = max(bbox_width, bbox_height * (width_ratio / height_ratio))
            else:  # 纵向
                base_size = max(bbox_height, bbox_width * (height_ratio / width_ratio))
            
            target_size = int(base_size * Box_grow_factor)
            
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 计算裁剪区域边界
            if width_ratio >= height_ratio:  # 横向或正方形
                half_width = target_size // 2
                half_height = int(half_width * (height_ratio / width_ratio))
            else:  # 纵向
                half_height = target_size // 2
                half_width = int(half_height * (width_ratio / height_ratio))
            
            crop_x1 = center_x - half_width
            crop_y1 = center_y - half_height
            crop_x2 = center_x + half_width
            crop_y2 = center_y + half_height
        
        # 处理边界超出图像的情况
        pad_left = max(0, -crop_x1)
        pad_top = max(0, -crop_y1)
        pad_right = max(0, crop_x2 - pil_image.width)
        pad_bottom = max(0, crop_y2 - pil_image.height)
        
        # 如果需要padding，则对图像和mask都进行padding
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            pil_image = ImageOps.expand(pil_image, (pad_left, pad_top, pad_right, pad_bottom), fill=(255, 255, 255))
            pil_mask = ImageOps.expand(pil_mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            
            # 调整坐标
            crop_x1 += pad_left
            crop_y1 += pad_top
            crop_x2 += pad_left
            crop_y2 += pad_top
        
        # 裁剪区域
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        cropped_image = pil_image.crop(crop_box)
        cropped_mask = pil_mask.crop(crop_box)
        
        # 根据resize_mode决定是否进行缩放
        if resize_mode == "NaN":
            # 不执行缩放，但应用divisible_by要求
            resized_image = cropped_image
            resized_mask = cropped_mask
            
            # 如果divisible_by大于1，确保尺寸可被整除
            if divisible_by > 1:
                width, height = cropped_image.size
                
                # 计算新的尺寸，确保可被divisible_by整除
                new_width = ((width + divisible_by - 1) // divisible_by) * divisible_by
                new_height = ((height + divisible_by - 1) // divisible_by) * divisible_by
                
                # 如果尺寸发生变化，进行缩放
                if new_width != width or new_height != height:
                    # 使用lanczos算法进行质量较好的缩放
                    resized_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
                    resized_mask = cropped_mask.resize((new_width, new_height), Image.LANCZOS)
        else:
            # 计算目标尺寸
            target_dimensions = self._calculate_target_dimensions(megapixels, best_aspect_ratio, divisible_by)
            
            # 获取重采样过滤器
            resample_filter = {
                "lanczos": Image.LANCZOS,
                "nearest-exact": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC
            }.get(resize_mode, Image.LANCZOS)
            
            # 进行缩放
            resized_image = cropped_image.resize(target_dimensions, resample_filter)
            resized_mask = cropped_mask.resize(target_dimensions, resample_filter)
        
        # 转换回tensor
        output_image = self._pil_to_tensor(resized_image)
        output_mask = self._pil_to_mask(resized_mask)
        
        # 返回crop_box信息以便还原使用
        crop_info = {
            "original_coords": crop_box,
            "padded_size": (pil_image.width, pil_image.height),
            "original_image_size": (image.shape[2], image.shape[1]),  # width, height
            "pad_info": (pad_left, pad_top, pad_right, pad_bottom)
        }
        
        return (output_image, crop_info, output_mask)


class ImageRestoreNodeV2:
    """
    将处理后的图像粘贴回原来的图像中
    """
    
    CATEGORY = "AFL/Mask"
    DESCRIPTION = "将处理后的图像粘贴回原图"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "crop_box": ("CROPBOX",),
                "blur_amount": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
                "mask_protect": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "tooltip": "控制输入mask边缘模糊的值，保护mask区域不被blur_amount的腐蚀所影响"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "restore_image"
    
    def _tensor_to_pil(self, tensor):
        """将ComfyUI的tensor转换为PIL图像"""
        # 确保输入tensor是正确的数据类型和形状
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
            
        # 处理不同的tensor形状
        if len(tensor.shape) == 4:
            # 标准的4D tensor (batch, height, width, channels)
            img_np = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(tensor.shape) == 3:
            # 3D tensor (height, width, channels)
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
            
        return Image.fromarray(img_np)
    
    def _tensor_to_pil_mask(self, mask):
        """将ComfyUI的mask tensor转换为PIL图像"""
        # 确保输入mask是正确的数据类型
        if mask.dtype != torch.float32:
            mask = mask.float()
            
        # 处理不同的mask形状
        if len(mask.shape) == 4:
            # 标准的4D mask tensor (batch, height, width, channels)
            mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 3:
            # 3D mask tensor (batch, height, width) 或 (height, width, channels)
            if mask.shape[0] == 1:  # (1, height, width)
                mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            else:  # (height, width, channels) 或其他情况
                mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        elif len(mask.shape) == 2:
            # 2D mask tensor (height, width)
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            
        # 确保mask_np是2D数组
        if len(mask_np.shape) > 2:
            # 如果是3D数组，取第一个通道
            mask_np = mask_np[:, :, 0] if mask_np.shape[2] > 1 else mask_np[:, :, 0]
            
        return Image.fromarray(mask_np, mode='L')

    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为ComfyUI的tensor"""
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None,]
    
    def restore_image(self, original_image, processed_image, crop_box, blur_amount, mask=None, mask_protect=0):
        # 将输入转换为PIL图像
        original_pil = self._tensor_to_pil(original_image)
        processed_pil = self._tensor_to_pil(processed_image)
        
        # 获取裁剪信息
        original_coords = crop_box["original_coords"]
        padded_size = crop_box["padded_size"]
        original_image_size = crop_box["original_image_size"]
        pad_info = crop_box["pad_info"]
        
        pad_left, pad_top, pad_right, pad_bottom = pad_info
        
        # 调整processed_image大小以匹配裁剪区域
        crop_width = original_coords[2] - original_coords[0]
        crop_height = original_coords[3] - original_coords[1]
        resized_processed = processed_pil.resize((crop_width, crop_height), Image.LANCZOS)
        
        # 创建一个与填充后图像相同大小的图像副本
        restored_image = original_pil.copy()
        if padded_size != (original_image_size[0], original_image_size[1]):
            # 如果之前进行了padding，我们需要创建一个填充后的图像
            restored_image = Image.new("RGB", padded_size, (255, 255, 255))
            # 粘贴原始图像的有效区域
            orig_region = (
                pad_left, 
                pad_top, 
                pad_left + original_image_size[0], 
                pad_top + original_image_size[1]
            )
            restored_image.paste(original_pil, orig_region)
        
        # 保存原始填充后图像用于边缘模糊
        padded_original = restored_image.copy()
        
        # 将处理后的图像粘贴回去
        restored_image.paste(resized_processed, original_coords[:2])
        
        # 应用边缘模糊效果
        if blur_amount > 0:
            restored_image = self._apply_edge_blur(
                restored_image, 
                padded_original, 
                original_coords, 
                blur_amount,
                pad_info,
                original_image_size,
                mask,  # 传递mask参数
                mask_protect  # 传递mask_protect参数
            )
        
        # 移除padding回到原始尺寸
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            restored_image = restored_image.crop((
                pad_left, 
                pad_top, 
                pad_left + original_image_size[0], 
                pad_top + original_image_size[1]
            ))
        
        # 转换回tensor
        output_image = self._pil_to_tensor(restored_image)
        return (output_image,)
    
    def _apply_edge_blur(self, restored_image, original_image, crop_coords, blur_amount, pad_info, original_size, input_mask=None, mask_protect=0):
        """应用边缘模糊效果，支持mask保护
        blur_amount: 边缘纯白色的像素数量（从裁剪框边缘向内延伸的宽度，完全透明区域，显示原始图像）
        """
        # 将PIL图像转换为numpy数组进行处理
        restored_np = np.array(restored_image)
        original_np = np.array(original_image)
        
        x1, y1, x2, y2 = crop_coords
        pad_left, pad_top, pad_right, pad_bottom = pad_info
        orig_width, orig_height = original_size
        
        # 计算在填充后图像中的实际坐标
        actual_x1 = x1
        actual_y1 = y1
        actual_x2 = x2
        actual_y2 = y2
        
        # 创建基础mask，初始化为全0（完全使用处理后图像）
        base_mask = np.zeros(restored_np.shape[:2], dtype=np.uint8)
        
        # 定义边缘纯白色区域（完全透明，显示原始图像）
        # 边缘区域：从裁剪框边缘向内延伸blur_amount像素的区域
        if blur_amount > 0:
            # 计算内部区域边界（从边缘向内延伸blur_amount像素后的边界）
            inner_x1 = actual_x1 + blur_amount
            inner_y1 = actual_y1 + blur_amount
            inner_x2 = actual_x2 - blur_amount
            inner_y2 = actual_y2 - blur_amount
            
            # 确保内部区域边界有效（不超出裁剪框范围）
            inner_x1 = max(actual_x1, min(inner_x1, actual_x2))
            inner_y1 = max(actual_y1, min(inner_y1, actual_y2))
            inner_x2 = max(actual_x1, min(inner_x2, actual_x2))
            inner_y2 = max(actual_y1, min(inner_y2, actual_y2))
            
            # 定义过渡区域边界（从边缘向内延伸2*blur_amount像素，用于模糊过渡）
            transition_x1 = actual_x1 + blur_amount
            transition_y1 = actual_y1 + blur_amount
            transition_x2 = actual_x2 - blur_amount
            transition_y2 = actual_y2 - blur_amount
            
            # 创建过渡区域的mask（用于后续模糊）
            transition_mask = np.zeros(restored_np.shape[:2], dtype=np.uint8)
            
            # 定义纯白色边缘区域（完全显示原始图像）
            # 1. 左边边缘：从左边界到inner_x1
            if actual_x1 < inner_x1:
                base_mask[:, actual_x1:inner_x1] = 255
            
            # 2. 右边边缘：从inner_x2到右边界
            if inner_x2 < actual_x2:
                base_mask[:, inner_x2:actual_x2] = 255
            
            # 3. 顶部边缘：从顶部边界到inner_y1
            if actual_y1 < inner_y1:
                base_mask[actual_y1:inner_y1, actual_x1:actual_x2] = 255
            
            # 4. 底部边缘：从inner_y2到actual_y2
            if inner_y2 < actual_y2:
                base_mask[inner_y2:actual_y2, actual_x1:actual_x2] = 255
            
            # 定义过渡区域（用于创建平滑过渡效果）
            # 这个区域将在后面应用高斯模糊
            # 1. 左侧过渡区：从inner_x1到transition_x1 + blur_amount
            left_transition_end = min(transition_x1 + blur_amount, actual_x2)
            if inner_x1 < left_transition_end:
                transition_mask[:, inner_x1:left_transition_end] = 255
            
            # 2. 右侧过渡区：从transition_x2 - blur_amount到inner_x2
            right_transition_start = max(transition_x2 - blur_amount, actual_x1)
            if right_transition_start < inner_x2:
                transition_mask[:, right_transition_start:inner_x2] = 255
            
            # 3. 顶部过渡区：从inner_y1到transition_y1 + blur_amount
            top_transition_end = min(transition_y1 + blur_amount, actual_y2)
            if inner_y1 < top_transition_end:
                transition_mask[inner_y1:top_transition_end, actual_x1:actual_x2] = 255
            
            # 4. 底部过渡区：从transition_y2 - blur_amount到inner_y2
            bottom_transition_start = max(transition_y2 - blur_amount, actual_y1)
            if bottom_transition_start < inner_y2:
                transition_mask[bottom_transition_start:inner_y2, actual_x1:actual_x2] = 255
            
            # 清除内部区域（中心部分）为0，确保内部区域完全使用处理后图像
            if inner_x1 < inner_x2 and inner_y1 < inner_y2:
                base_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
                transition_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0
        
        # 应用高斯模糊来创建从纯白到纯黑的平滑过渡
        # 只对过渡区域应用模糊，保留纯白色边缘区域
        if blur_amount > 0:
            # 模糊半径为blur_amount，确保过渡平滑
            # 使用奇数大小的内核以获得更好的效果
            kernel_size = 2 * blur_amount + 1
            
            # 只对过渡区域应用模糊
            blurred_transition = cv2.GaussianBlur(transition_mask, (kernel_size, kernel_size), 0)
            
            # 创建最终mask：纯白色边缘区域保持不变，过渡区域使用模糊后的结果
            # 先复制基础mask（包含纯白色边缘区域）
            final_mask = base_mask.copy()
            
            # 在过渡区域应用模糊效果，但确保不覆盖纯白色边缘区域
            # 只在base_mask为0且transition_mask不为0的区域应用模糊
            transition_area = (base_mask == 0) & (transition_mask > 0)
            final_mask[transition_area] = blurred_transition[transition_area]
        else:
            final_mask = base_mask
        
        # 归一化到0-1范围
        final_mask = final_mask.astype(np.float32) / 255.0
        
        # 如果有输入mask并且mask_protect大于0，应用mask保护
        if input_mask is not None and mask_protect > 0:
            # 将输入mask转换为PIL图像
            pil_input_mask = self._tensor_to_pil_mask(input_mask)
            
            # 调整mask大小以匹配裁剪区域
            mask_width = actual_x2 - actual_x1
            mask_height = actual_y2 - actual_y1
            resized_mask = pil_input_mask.resize((mask_width, mask_height), Image.LANCZOS)
            
            # 创建与填充后图像相同大小的mask
            full_mask = np.ones(restored_np.shape[:2], dtype=np.uint8) * 255
            full_mask[actual_y1:actual_y2, actual_x1:actual_x2] = np.array(resized_mask)
            
            # 应用mask_protect模糊，使mask边缘平滑
            if mask_protect > 0:
                full_mask = cv2.GaussianBlur(full_mask, (2 * mask_protect + 1, 2 * mask_protect + 1), 0)
            
            # 归一化到0-1范围
            full_mask = full_mask.astype(np.float32) / 255.0
            
            # 保护mask区域：在mask区域内使用较小的mask值（更倾向于显示处理后图像）
            # 使用full_mask作为权重，越接近mask中心，权重越大
            final_mask = final_mask * (1 - full_mask) + 0 * full_mask
        
        # 扩展mask维度以匹配图像
        final_mask = np.stack([final_mask] * 3, axis=-1)
        
        # 应用混合：
        # - mask值为0：完全使用处理后的图像 (restored_np)
        # - mask值为1：完全使用原始图像 (original_np)
        # - 中间值：混合使用
        restored_np = (restored_np * (1 - final_mask) + original_np * final_mask).astype(np.uint8)
        
        # 转换回PIL图像
        return Image.fromarray(restored_np)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AFL2:MaskBoxCropNodeV2": MaskBoxCropNodeV2,
    "AFL2:ImageRestoreNodeV2": ImageRestoreNodeV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL2:MaskBoxCropNodeV2": "AFL Target box cropV2",
    "AFL2:ImageRestoreNodeV2": "AFL Target restoreV2"
}