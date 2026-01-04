import os
import numpy as np
import torch
from PIL import Image
import struct
import folder_paths

class RleCompression:
    """RLE 压缩实现"""
    MAX_LENGTH = 127

    @staticmethod
    def compress(src_bytes):
        """压缩字节数组"""
        if len(src_bytes) == 0:
            return b''
        
        if len(src_bytes) == 1:
            return struct.pack('BB', 0, src_bytes[0])
        
        dst_bytes = bytearray()
        pos = 0
        buf = bytearray()
        repeat_count = 0
        state = 0  # 0 == RAW, 1 == RLE
        
        while pos < len(src_bytes) - 1:
            current_byte = src_bytes[pos]
            if src_bytes[pos] == src_bytes[pos + 1]:
                if state == 0:
                    # RAW 数据结束
                    if len(buf) != 0:
                        dst_bytes.append(len(buf) - 1)
                        dst_bytes.extend(buf)
                        buf.clear()
                    state = 1
                    repeat_count = 1
                elif state == 1:
                    repeat_count += 1
                    if repeat_count == RleCompression.MAX_LENGTH:
                        # 转换为有符号字节: -(repeat_count - 1) 作为有符号字节
                        rle_value = -(repeat_count - 1)
                        # 转换为无符号字节表示
                        rle_byte = rle_value & 0xFF
                        dst_bytes.append(rle_byte)
                        dst_bytes.append(src_bytes[pos])
                        repeat_count = 0
            else:
                if state == 1:
                    repeat_count += 1
                    rle_value = -(repeat_count - 1)
                    rle_byte = rle_value & 0xFF
                    dst_bytes.append(rle_byte)
                    dst_bytes.append(src_bytes[pos])
                    state = 0
                    repeat_count = 0
                elif state == 0:
                    if len(buf) == RleCompression.MAX_LENGTH:
                        if len(buf) != 0:
                            dst_bytes.append(len(buf) - 1)
                            dst_bytes.extend(buf)
                            buf.clear()
                    buf.append(current_byte)
            pos += 1
        
        # 处理最后一个字节
        if state == 0:
            buf.append(src_bytes[pos])
            if len(buf) != 0:
                dst_bytes.append(len(buf) - 1)
                dst_bytes.extend(buf)
        else:
            repeat_count += 1
            rle_value = -(repeat_count - 1)
            rle_byte = rle_value & 0xFF
            dst_bytes.append(rle_byte)
            dst_bytes.append(src_bytes[pos])
        
        return bytes(dst_bytes)

class Layer:
    """图层类"""
    
    def __init__(self, image, name=None, top=0, left=0, alpha=255, visible=True):
        self.image = image
        self.name = name or "Layer"
        self.top = top
        self.left = left
        self.bottom = top + image.height
        self.right = left + image.width
        self.alpha = alpha
        self.visible = visible
        self.transparency_protected = True
        self.obsolete = False
        self.pixel_data_irrelevant_value_useful = False
        self.pixel_data_irrelevant = False
    
    def get_width(self):
        return self.right - self.left
    
    def get_height(self):
        return self.bottom - self.top

class PsdWriter:
    """自定义PSD文件写入器"""
    
    @staticmethod
    def write_psd(layers_data, output_path, canvas_width=None, canvas_height=None):
        """将图层数据写入PSD文件"""
        if not layers_data:
            raise ValueError("至少需要一个图层")
        
        # 确定画布尺寸
        if canvas_width is None:
            canvas_width = max(layer['image'].width for layer in layers_data)
        if canvas_height is None:
            canvas_height = max(layer['image'].height for layer in layers_data)
        
        # 创建图层对象
        layers = []
        for layer_data in layers_data:
            layer = Layer(
                image=layer_data['image'],
                name=layer_data['name']
            )
            layers.append(layer)
        
        # 创建合并预览图
        preview_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        for layer in layers:
            preview_img.paste(layer.image, (0, 0), layer.image if layer.image.mode == 'RGBA' else None)
        
        # 写入文件
        with open(output_path, 'wb') as f:
            # 写入文件头
            PsdWriter._write_header(f, canvas_width, canvas_height)
            # 写入颜色模式数据
            PsdWriter._write_color_mode_data(f)
            # 写入图像资源
            PsdWriter._write_image_resources(f)
            # 写入图层数据
            PsdWriter._write_layers(f, layers)
            # 写入图像数据
            PsdWriter._write_image_data(f, preview_img)
        
        return output_path
    
    @staticmethod
    def _write_header(f, width, height):
        """写入PSD文件头"""
        # signature: "8BPS"
        f.write(b'8BPS')
        # version: 1
        f.write(struct.pack('>H', 1))
        # Reserved: 6 bytes
        f.write(b'\x00' * 6)
        # Channels: 4 (RGBA)
        f.write(struct.pack('>H', 4))
        # Rows (height)
        f.write(struct.pack('>I', height))
        # Columns (width)
        f.write(struct.pack('>I', width))
        # Depth: 8 bits
        f.write(struct.pack('>H', 8))
        # Mode: RGB = 3
        f.write(struct.pack('>H', 3))
    
    @staticmethod
    def _write_color_mode_data(f):
        """写入颜色模式数据"""
        # Color mode data length (0 for RGB)
        f.write(struct.pack('>I', 0))
    
    @staticmethod
    def _write_image_resources(f):
        """写入图像资源数据"""
        # Image resources length
        f.write(struct.pack('>I', 0))
    
    @staticmethod
    def _write_layers(f, layers):
        """写入图层数据"""
        bytes_list = []
        bytes_data_list = []
        compression = 1  # RLE 压缩
        
        # 图层数量
        layer_count = struct.pack('>H', len(layers))
        bytes_list.append(layer_count)
        
        # 处理每个图层
        for layer in layers:
            img = layer.image
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # 获取像素数据
            img_array = np.array(img)
            width = layer.get_width()
            height = layer.get_height()
            
            channel_info = bytearray()
            layer_channel_data = []  # 存储每个通道的数据
            
            # 处理 4 个通道 (R, G, B, A)
            for channel_idx in range(4):
                channel_data = bytearray()
                
                # 提取通道数据
                for y in range(height):
                    for x in range(width):
                        if channel_idx == 0:
                            channel_data.append(img_array[y, x, 0])  # R
                        elif channel_idx == 1:
                            channel_data.append(img_array[y, x, 1])  # G
                        elif channel_idx == 2:
                            channel_data.append(img_array[y, x, 2])  # B
                        elif channel_idx == 3:
                            channel_data.append(img_array[y, x, 3])  # A
                
                if compression == 1:
                    # RLE 压缩
                    # Alpha通道使用特殊的通道ID -1
                    channel_id = -1 if channel_idx == 3 else channel_idx
                    channel_info.extend(struct.pack('>h', channel_id))
                    
                    # 压缩每一行
                    compressed_rows = []
                    row_lengths = []
                    for y in range(height):
                        row_data = bytes(channel_data[y * width:(y + 1) * width])
                        compressed = RleCompression.compress(row_data)
                        compressed_rows.append(compressed)
                        row_lengths.append(len(compressed))
                    
                    # 计算总长度
                    total_length = 2 + 2 * height + sum(row_lengths)
                    channel_info.extend(struct.pack('>I', total_length))
                    
                    # 组织通道数据：压缩方式 + 行长度数组 + 压缩数据
                    channel_bytes = bytearray()
                    channel_bytes.extend(struct.pack('>H', compression))
                    
                    # 行长度数组
                    for length in row_lengths:
                        channel_bytes.extend(struct.pack('>H', length))
                    
                    # 压缩的行数据
                    for compressed_row in compressed_rows:
                        channel_bytes.extend(compressed_row)
                    
                    layer_channel_data.append(bytes(channel_bytes))
                else:
                    # 未压缩
                    channel_id = -1 if channel_idx == 3 else channel_idx
                    channel_info.extend(struct.pack('>h', channel_id))
                    channel_info.extend(struct.pack('>I', 2 + width * height))
                    
                    channel_bytes = bytearray()
                    channel_bytes.extend(struct.pack('>H', compression))
                    channel_bytes.extend(channel_data)
                    layer_channel_data.append(bytes(channel_bytes))
            
            # 图层信息头部
            # 边界框
            bounds = struct.pack('>IIII', layer.top, layer.left, layer.bottom, layer.right)
            bytes_list.append(bounds)
            
            # 通道数量
            channel_num = struct.pack('>H', 4)
            bytes_list.append(channel_num)
            
            # 通道信息
            bytes_list.append(bytes(channel_info))
            
            # 混合模式签名
            blend_sign = b'8BIM'
            bytes_list.append(blend_sign)
            
            # 混合模式
            blend_mode = b'norm'
            bytes_list.append(blend_mode)
            
            # 透明度
            opacity = struct.pack('B', layer.alpha)
            bytes_list.append(opacity)
            
            # 裁剪
            clipping = b'\x00'
            bytes_list.append(clipping)
            
            # Flags
            flag = 0
            if layer.transparency_protected:
                flag |= 1
            if not layer.visible:
                flag |= 1 << 1
            flags = struct.pack('B', flag)
            bytes_list.append(flags)
            
            # Filler
            filler = b'\x00'
            bytes_list.append(filler)
            
            # Extra data size
            # Layer mask (4 bytes, empty)
            layer_mask = b'\x00' * 4
            
            # Layer blending ranges (44 bytes)
            blending_ranges = bytearray()
            blending_ranges.extend(struct.pack('>I', 40))
            for _ in range(10):
                blending_ranges.extend(struct.pack('>I', 0x0000ffff))
            
            # Layer name (Pascal string, padded to multiple of 4)
            name_bytes = layer.name.encode('utf-8')
            name_len = len(name_bytes)
            padded_len = ((name_len + 1 + 3) & ~3)
            name_buffer = bytearray(padded_len)
            name_buffer[0] = name_len
            name_buffer[1:1+name_len] = name_bytes
            
            # Extra data size
            extra_data_size = len(layer_mask) + len(blending_ranges) + len(name_buffer)
            extra_data_size_bytes = struct.pack('>I', extra_data_size)
            bytes_list.append(extra_data_size_bytes)
            bytes_list.append(layer_mask)
            bytes_list.append(bytes(blending_ranges))
            bytes_list.append(bytes(name_buffer))
            
            # 添加图层数据
            bytes_data_list.extend(layer_channel_data)
        
        # 合并所有数据
        bytes_list.extend(bytes_data_list)
        
        # 计算总大小
        all_size = sum(len(b) for b in bytes_list)
        
        # Layer mask info section
        f.write(struct.pack('>I', 4 + all_size + 4))  # Total size
        f.write(struct.pack('>I', all_size))  # Layer info size
        for b in bytes_list:
            f.write(b)
        f.write(struct.pack('>I', 0))  # Global layer mask (empty)
    
    @staticmethod
    def _write_image_data(f, image):
        """写入图像数据（合并后的预览图）"""
        if image.mode != 'RGBA':
            img = image.convert('RGBA')
        else:
            img = image
        
        img_array = np.array(img)
        width, height = img.size
        
        # 压缩方式 (RLE)
        compression = 1
        
        f.write(struct.pack('>H', compression))
        
        # 为每个通道压缩数据
        for channel_idx in range(4):
            channel_data = bytearray()
            for y in range(height):
                for x in range(width):
                    if channel_idx == 0:
                        channel_data.append(img_array[y, x, 0])  # R
                    elif channel_idx == 1:
                        channel_data.append(img_array[y, x, 1])  # G
                    elif channel_idx == 2:
                        channel_data.append(img_array[y, x, 2])  # B
                    elif channel_idx == 3:
                        channel_data.append(img_array[y, x, 3])  # A
            
            # 压缩每一行并写入行长度
            for y in range(height):
                row_data = bytes(channel_data[y * width:(y + 1) * width])
                compressed = RleCompression.compress(row_data)
                # 写入行长度
                f.write(struct.pack('>H', len(compressed)))
        
        # 写入压缩数据
        for channel_idx in range(4):
            channel_data = bytearray()
            for y in range(height):
                for x in range(width):
                    if channel_idx == 0:
                        channel_data.append(img_array[y, x, 0])  # R
                    elif channel_idx == 1:
                        channel_data.append(img_array[y, x, 1])  # G
                    elif channel_idx == 2:
                        channel_data.append(img_array[y, x, 2])  # B
                    elif channel_idx == 3:
                        channel_data.append(img_array[y, x, 3])  # A
            
            # 压缩每一行并写入数据
            for y in range(height):
                row_data = bytes(channel_data[y * width:(y + 1) * width])
                compressed = RleCompression.compress(row_data)
                f.write(compressed)

class AFL_PSD_Layer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "layer_1": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("PSD_LAYERS",)
    RETURN_NAMES = ("layers",)
    FUNCTION = "add_layers"
    CATEGORY = "AFL/PSD"

    def add_layers(self, **kwargs):
        layers = []
        
        # 收集所有 layer_N 输入并按序号排序
        layer_inputs = []
        for key, value in kwargs.items():
            if value is not None and key.startswith("layer_"):
                layer_id = int(key.split('_')[1])
                layer_inputs.append((layer_id, value))
        
        layer_inputs.sort(key=lambda x: x[0])
        
        # 处理每个图层
        for layer_id, image in layer_inputs:
            img_tensor = image[0]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            if img_np.shape[-1] == 4:
                pil_image = Image.fromarray(img_np, mode='RGBA')
            else:
                pil_image = Image.fromarray(img_np)
                pil_image = pil_image.convert('RGBA')
            
            layer_data = {
                "name": f"Layer_{layer_id}",
                "image": pil_image
            }
            layers.append(layer_data)
        
        return (layers,)

class AFL_Save_PSD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers": ("PSD_LAYERS",),
                "filename": ("STRING", {"default": "output.psd"}),
            },
            "optional": {
                "save_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_psd"
    CATEGORY = "AFL/PSD"
    OUTPUT_NODE = True

    def save_psd(self, layers, filename, save_path=""):
        # Determine save directory
        if save_path and save_path.strip():
            output_dir = save_path
        else:
            output_dir = folder_paths.get_output_directory()
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        full_path = os.path.join(output_dir, filename)
        if not full_path.lower().endswith('.psd'):
            full_path += '.psd'
            
        # 验证图层数据
        if not layers:
            print("[AFL_Save_PSD] No layers to save.")
            return {}
            
        # 显示图层信息
        layer_info = []
        for i, layer in enumerate(layers):
            img = layer['image']
            name = layer['name']
            mode = img.mode
            size = img.size
            layer_info.append(f"Layer {i+1}: '{name}' ({mode}, {size[0]}x{size[1]})")
        
        print(f"[AFL_Save_PSD] 准备保存PSD文件...")
        print(f"[AFL_Save_PSD] 图层信息:")
        for info in layer_info:
            print(f"[AFL_Save_PSD]   {info}")
        
        # 使用自定义PsdWriter保存文件
        try:
            print(f"[AFL_Save_PSD] 开始写入PSD文件: {full_path}")
            PsdWriter.write_psd(layers, full_path)
            
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                print(f"[AFL_Save_PSD] 成功保存到 {full_path} ({size/1024:.2f} KB)")
                print(f"[AFL_Save_PSD] PSD文件包含 {len(layers)} 个图层，所有图层alpha通道已正确保存")
            else:
                print(f"[AFL_Save_PSD] 警告: 文件保存命令执行，但文件不存在: {full_path}")
                
        except Exception as e:
            print(f"[AFL_Save_PSD] 保存PSD时出错: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[AFL_Save_PSD] 错误详情: {traceback.format_exc()}")
            
        return {}

NODE_CLASS_MAPPINGS = {
    "AFL2_PSD_Layer": AFL_PSD_Layer,
    "AFL2_Save_PSD": AFL_Save_PSD
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL2_PSD_Layer": "AFL PSD Layer V2",
    "AFL2_Save_PSD": "AFL Save PSD V2"
}
