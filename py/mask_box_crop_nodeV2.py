import torch
import numpy as np
from PIL import Image, ImageOps
import cv2
import re

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
                "fill_color": ("STRING", {"default": "#FFFFFF", "tooltip": "边界超出时的填充颜色，支持hex格式(#FFFFFF/#FFF)或颜色名称(red/blue/green等)"}),
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
            img_np = np.clip(tensor[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        elif len(tensor.shape) == 3:
            # 3D tensor (height, width, channels)
            img_np = np.clip(tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            img_np = np.clip(tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            
        return Image.fromarray(img_np)
    
    def _tensor_to_pil_mask(self, mask):
        """将ComfyUI的mask tensor转换为PIL图像"""
        # 确保输入mask是正确的数据类型
        if mask.dtype != torch.float32:
            mask = mask.float()
            
        # 处理不同的mask形状
        if len(mask.shape) == 4:
            # 标准的4D mask tensor (batch, height, width, channels)
            mask_np = np.clip(mask[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        elif len(mask.shape) == 3:
            # 3D mask tensor (batch, height, width) 或 (height, width, channels)
            if mask.shape[0] == 1:  # (1, height, width)
                mask_np = np.clip(mask[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            else:  # (height, width, channels) 或其他情况
                mask_np = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        elif len(mask.shape) == 2:
            # 2D mask tensor (height, width)
            mask_np = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            mask_np = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            
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
    
    def _hex_to_rgb(self, hex_color):
        """将hex颜色值或颜色名称转换为RGB元组"""
        # 基础颜色名称映射
        color_names = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 128, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'orange': (255, 165, 0),
            'pink': (255, 192, 203),
            'purple': (128, 0, 128),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
        }
        
        # 清理输入并转小写
        hex_color = hex_color.strip().lower()
        
        # 检查是否是颜色名称
        if hex_color in color_names:
            return color_names[hex_color]
        
        # 移除#前缀
        hex_color = hex_color.lstrip('#')
        
        # 支持3位和6位hex格式
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        
        # 验证hex格式
        if not re.match('^[0-9a-f]{6}$', hex_color):
            # 无效格式，返回白色
            return (255, 255, 255)
        
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    
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
    
    def _process_single_image(self, pil_image, pil_mask, resize_mode, Box_grow_factor, megapixels, divisible_by, ratio, startup_threshold, original_width, original_height, fill_color=(255, 255, 255)):
        """处理单张图片的裁剪和缩放"""
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
            pil_image = ImageOps.expand(pil_image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill_color)
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
        
        # 返回crop_box信息以便还原使用
        crop_info = {
            "original_coords": crop_box,
            "padded_size": (pil_image.width, pil_image.height),
            "original_image_size": (original_width, original_height),  # width, height
            "pad_info": (pad_left, pad_top, pad_right, pad_bottom),
            "fill_color": fill_color
        }
        
        return resized_image, resized_mask, crop_info

    def crop_and_resize(self, image, mask, resize_mode, Box_grow_factor=1.0, megapixels=1.0, divisible_by=1, ratio="auto", startup_threshold=0.4, fill_color="#FFFFFF"):
        image_batch_size = image.shape[0]
        mask_batch_size = mask.shape[0] if len(mask.shape) == 3 else 1
        # 取image和mask中较大的batch_size
        batch_size = max(image_batch_size, mask_batch_size)
        
        original_width = image.shape[2]
        original_height = image.shape[1]
        
        # 转换hex颜色为RGB元组
        fill_color_rgb = self._hex_to_rgb(fill_color)
        
        output_images = []
        output_masks = []
        crop_infos = []
        
        for i in range(batch_size):
            # 获取当前批次的图像，如果image只有1张则复用
            single_image = image[i] if i < image_batch_size else image[0]
            # 获取当前批次的mask，如果mask只有1张则复用
            if len(mask.shape) == 3:
                single_mask = mask[i] if i < mask_batch_size else mask[0]
            else:
                single_mask = mask
            
            # 转换为PIL图像
            img_np = np.clip(single_image.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)
            
            mask_np = np.clip(single_mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            if len(mask_np.shape) > 2:
                mask_np = mask_np[:, :, 0]
            pil_mask = Image.fromarray(mask_np, mode='L')
            
            # 处理单张图片
            resized_image, resized_mask, crop_info = self._process_single_image(
                pil_image, pil_mask, resize_mode, Box_grow_factor, 
                megapixels, divisible_by, ratio, startup_threshold,
                original_width, original_height, fill_color_rgb
            )
            
            # 转换回tensor
            img_tensor = np.array(resized_image).astype(np.float32) / 255.0
            mask_tensor = np.array(resized_mask).astype(np.float32) / 255.0
            
            output_images.append(img_tensor)
            output_masks.append(mask_tensor)
            crop_infos.append(crop_info)
        
        # 堆叠为批次tensor
        output_image = torch.from_numpy(np.stack(output_images, axis=0))
        output_mask = torch.from_numpy(np.stack(output_masks, axis=0))
        
        # 返回crop_box信息列表以便还原使用
        crop_info_batch = {
            "batch_size": batch_size,
            "crop_infos": crop_infos
        }
        
        return (output_image, crop_info_batch, output_mask)


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
                "blur_amount": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "tooltip": "边缘羽化值，对mask边缘或bbox边缘应用高斯模糊"}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1, "tooltip": "遮罩扩展值，正值扩展，负值收缩"}),
            },
            "optional": {
                "mask": ("MASK",),
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
            img_np = np.clip(tensor[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        elif len(tensor.shape) == 3:
            # 3D tensor (height, width, channels)
            img_np = np.clip(tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            img_np = np.clip(tensor.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            
        return Image.fromarray(img_np)
    
    def _tensor_to_pil_mask(self, mask):
        """将ComfyUI的mask tensor转换为PIL图像"""
        # 确保输入mask是正确的数据类型
        if mask.dtype != torch.float32:
            mask = mask.float()
            
        # 处理不同的mask形状
        if len(mask.shape) == 4:
            # 标准的4D mask tensor (batch, height, width, channels)
            mask_np = np.clip(mask[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        elif len(mask.shape) == 3:
            # 3D mask tensor (batch, height, width) 或 (height, width, channels)
            if mask.shape[0] == 1:  # (1, height, width)
                mask_np = np.clip(mask[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
            else:  # (height, width, channels) 或其他情况
                mask_np = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        elif len(mask.shape) == 2:
            # 2D mask tensor (height, width)
            mask_np = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        else:
            # 其他情况，尝试处理
            mask_np = np.clip(mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            
        # 确保mask_np是2D数组
        if len(mask_np.shape) > 2:
            # 如果是3D数组，取第一个通道
            mask_np = mask_np[:, :, 0] if mask_np.shape[2] > 1 else mask_np[:, :, 0]
            
        return Image.fromarray(mask_np, mode='L')

    def _pil_to_tensor(self, pil_image):
        """将PIL图像转换为ComfyUI的tensor"""
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None,]
    
    def _restore_single_image(self, original_pil, processed_pil, crop_info, blur_amount, mask_expand, single_mask=None):
        """处理单张图片的还原"""
        # 获取裁剪信息
        original_coords = crop_info["original_coords"]
        padded_size = crop_info["padded_size"]
        original_image_size = crop_info["original_image_size"]
        pad_info = crop_info["pad_info"]
        fill_color = crop_info.get("fill_color", (255, 255, 255))  # 兼容旧格式
        
        pad_left, pad_top, pad_right, pad_bottom = pad_info
        
        # 调整processed_image大小以匹配裁剪区域
        crop_width = original_coords[2] - original_coords[0]
        crop_height = original_coords[3] - original_coords[1]
        resized_processed = processed_pil.resize((crop_width, crop_height), Image.LANCZOS)
        
        # 创建一个与填充后图像相同大小的图像副本
        restored_image = original_pil.copy()
        if padded_size != (original_image_size[0], original_image_size[1]):
            # 如果之前进行了padding，我们需要创建一个填充后的图像
            restored_image = Image.new("RGB", padded_size, fill_color)
            # 粘贴原始图像的有效区域
            orig_region = (
                pad_left, 
                pad_top, 
                pad_left + original_image_size[0], 
                pad_top + original_image_size[1]
            )
            restored_image.paste(original_pil, orig_region)
        
        # 保存原始填充后图像用于混合
        padded_original = restored_image.copy()
        
        # 根据是否有mask决定粘贴方式
        if single_mask is not None:
            # 有mask：只在mask区域粘贴处理后的图像
            restored_image = self._apply_mask_blend(
                restored_image,
                resized_processed,
                padded_original,
                original_coords,
                single_mask,
                blur_amount,
                mask_expand
            )
        else:
            # 无mask：粘贴整个bbox区域，使用边缘模糊
            restored_image.paste(resized_processed, original_coords[:2])
            if blur_amount > 0 or mask_expand != 0:
                restored_image = self._apply_bbox_edge_blur(
                    restored_image,
                    padded_original,
                    original_coords,
                    blur_amount,
                    mask_expand
                )
        
        # 移除padding回到原始尺寸
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            restored_image = restored_image.crop((
                pad_left, 
                pad_top, 
                pad_left + original_image_size[0], 
                pad_top + original_image_size[1]
            ))
        
        return restored_image

    def restore_image(self, original_image, processed_image, crop_box, blur_amount, mask_expand, mask=None):
        batch_size = original_image.shape[0]
        
        # 检查是否为批次格式的crop_box
        if "batch_size" in crop_box:
            crop_infos = crop_box["crop_infos"]
        else:
            # 兼容旧格式（单张图片）
            crop_infos = [crop_box] * batch_size
        
        output_images = []
        
        for i in range(batch_size):
            # 获取当前批次的图像
            single_original = original_image[i]
            single_processed = processed_image[i]
            crop_info = crop_infos[i] if i < len(crop_infos) else crop_infos[0]
            
            # 转换为PIL图像
            orig_np = np.clip(single_original.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            original_pil = Image.fromarray(orig_np)
            
            proc_np = np.clip(single_processed.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            processed_pil = Image.fromarray(proc_np)
            
            # 获取当前批次的mask（如果有）
            single_mask = None
            if mask is not None:
                if len(mask.shape) == 3:
                    single_mask = mask[i] if i < mask.shape[0] else mask[0]
                else:
                    single_mask = mask
            
            # 处理单张图片
            restored_image = self._restore_single_image(
                original_pil, processed_pil, crop_info, 
                blur_amount, mask_expand, single_mask
            )
            
            # 转换回numpy
            img_tensor = np.array(restored_image).astype(np.float32) / 255.0
            output_images.append(img_tensor)
        
        # 堆叠为批次tensor
        output_image = torch.from_numpy(np.stack(output_images, axis=0))
        
        return (output_image,)
    
    def _apply_mask_blend(self, restored_image, resized_processed, original_image, crop_coords, input_mask, blur_amount, mask_expand):
        """使用mask混合图像，只在mask区域粘贴处理后的图像
        blur_amount: 对mask边缘应用高斯模糊羽化
        mask_expand: 遮罩扩展值，正值扩展(dilate)，负值收缩(erode)
        input_mask: 单个mask tensor (height, width) 或 (1, height, width)
        """
        restored_np = np.array(restored_image)
        processed_np = np.array(resized_processed)
        original_np = np.array(original_image)
        
        x1, y1, x2, y2 = crop_coords
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 将输入mask转换为numpy数组
        if torch.is_tensor(input_mask):
            mask_np = np.clip(input_mask.cpu().numpy() * 255, 0, 255).astype(np.uint8)
            if len(mask_np.shape) > 2:
                mask_np = mask_np[0] if mask_np.shape[0] == 1 else mask_np[:, :, 0]
        else:
            mask_np = input_mask
        
        # 转换为PIL并调整大小
        pil_mask = Image.fromarray(mask_np, mode='L')
        resized_mask = pil_mask.resize((crop_width, crop_height), Image.LANCZOS)
        mask_np = np.array(resized_mask)
        
        # 对mask进行扩展或收缩
        if mask_expand != 0:
            abs_expand = abs(mask_expand)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs_expand * 2 + 1, abs_expand * 2 + 1))
            if mask_expand > 0:
                # 正值：扩展(膨胀)
                mask_np = cv2.dilate(mask_np, kernel, iterations=1)
            else:
                # 负值：收缩(腐蚀)
                mask_np = cv2.erode(mask_np, kernel, iterations=1)
        
        # 对mask边缘应用高斯模糊羽化
        if blur_amount > 0:
            kernel_size = blur_amount * 2 + 1
            mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
        
        # 归一化到0-1范围
        mask_float = mask_np.astype(np.float32) / 255.0
        
        # 扩展mask维度以匹配图像通道
        mask_3ch = np.stack([mask_float] * 3, axis=-1)
        
        # 提取原图在crop区域的部分
        original_crop = original_np[y1:y2, x1:x2]
        
        # 在crop区域内混合：mask白色区域使用processed，黑色区域使用original
        blended_crop = (processed_np * mask_3ch + original_crop * (1 - mask_3ch)).astype(np.uint8)
        
        # 将混合结果放回原图
        restored_np[y1:y2, x1:x2] = blended_crop
        
        return Image.fromarray(restored_np)
    
    def _apply_bbox_edge_blur(self, restored_image, original_image, crop_coords, blur_amount, mask_expand):
        """对bbox边缘应用高斯模糊过渡
        blur_amount: 高斯模糊的半径
        mask_expand: 遮罩扩展值，正值扩展(dilate)，负值收缩(erode)
        """
        restored_np = np.array(restored_image)
        original_np = np.array(original_image)
        
        x1, y1, x2, y2 = crop_coords
        img_h, img_w = restored_np.shape[:2]
        
        # 创建bbox形状的mask，内部为白色
        bbox_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        bbox_mask[y1:y2, x1:x2] = 255
        
        # 对mask进行扩展或收缩
        if mask_expand != 0:
            abs_expand = abs(mask_expand)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs_expand * 2 + 1, abs_expand * 2 + 1))
            if mask_expand > 0:
                # 正值：扩展(膨胀)
                bbox_mask = cv2.dilate(bbox_mask, kernel, iterations=1)
            else:
                # 负值：收缩(腐蚀)
                bbox_mask = cv2.erode(bbox_mask, kernel, iterations=1)
        
        # 对mask应用高斯模糊，创建边缘羽化
        if blur_amount > 0:
            kernel_size = blur_amount * 2 + 1
            bbox_mask = cv2.GaussianBlur(bbox_mask, (kernel_size, kernel_size), 0)
        
        # 归一化到0-1范围
        mask_float = bbox_mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_float] * 3, axis=-1)
        
        # 混合：mask白色区域使用restored，黑色区域使用original
        result_np = (restored_np * mask_3ch + original_np * (1 - mask_3ch)).astype(np.uint8)
        
        return Image.fromarray(result_np)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AFL2:MaskBoxCropNodeV2": MaskBoxCropNodeV2,
    "AFL2:ImageRestoreNodeV2": ImageRestoreNodeV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL2:MaskBoxCropNodeV2": "AFL Target box cropV2",
    "AFL2:ImageRestoreNodeV2": "AFL Target restoreV2"
}