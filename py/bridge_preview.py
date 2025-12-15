from .md import *
import torch
import json
import os
import random
import numpy as np
import hashlib
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.utils
from threading import Event
import threading
from server import PromptServer
from aiohttp import web
def get_bridge_storage():
    if not hasattr(PromptServer.instance, '_bridge_node_data'):
        PromptServer.instance._bridge_node_data = {}
    return PromptServer.instance._bridge_node_data
def get_bridge_cache():
    """获取桥接节点的缓存存储"""
    if not hasattr(PromptServer.instance, '_bridge_node_cache'):
        PromptServer.instance._bridge_node_cache = {}
    return PromptServer.instance._bridge_node_cache
class BridgePreviewNode(PreviewImage):
    """桥接预览节点，等待前端遮罩操作完成后输出图片"""
    def __init__(self):
        super().__init__()
        self.prefix_append = "_bridge_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "file_info": ("STRING", {"default": "", "readonly": True, "multiline": False}),
                "skip": ("BOOLEAN", {"default": False, "label_on": "Skip", "label_off": "Open dialog"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID"
            },
        }
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("处理后图像", "遮罩")
    FUNCTION = "process_image"
    OUTPUT_NODE = True
    CATEGORY = "🎈LAOGOU/Image"
    def calculate_image_hash(self, images):
        """计算图片的哈希值用于检测是否改变"""
        try:
            np_images = images.cpu().numpy()
            image_bytes = np_images.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
        except:
            return None
    def save_output_images(self, images, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        """保存输出图片（带遮罩的RGBA图片）用于前端显示"""
        try:
            if images.shape[0] != mask.shape[0]:
                return None
            batch_size = images.shape[0]
            saved_images = []
            for i in range(batch_size):
                single_image = images[i:i+1]
                single_mask = mask[i]
                np_image = (single_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                np_mask = (single_mask.cpu().numpy() * 255).astype(np.uint8)
                h, w, c = np_image.shape
                rgba_image = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_image[:, :, :3] = np_image
                rgba_image[:, :, 3] = np_mask
                pil_image = Image.fromarray(rgba_image, 'RGBA')
                counter = 1
                while True:
                    filename = f"{filename_prefix}_{counter:05d}_.png"
                    full_path = os.path.join(folder_paths.get_output_directory(), filename)
                    if not os.path.exists(full_path):
                        break
                    counter += 1
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))
                pil_image.save(full_path, pnginfo=metadata, compress_level=4)
                saved_images.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "output"
                })
            return {"ui": {"images": saved_images}}
        except Exception as e:
            return None
    def process_image(self, images, file_info="", skip=False, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, unique_id=None):
        """处理图像，等待前端遮罩操作完成"""
        try:
            node_id = str(unique_id)
            current_hash = self.calculate_image_hash(images)
            cache = get_bridge_cache()
            
            # 如果启用跳过预览，直接返回缓存结果或原图
            if skip:
                if node_id in cache and current_hash:
                    cached_data = cache[node_id]
                    if (cached_data.get("input_hash") == current_hash and 
                        cached_data.get("final_result")):
                        # 返回缓存的最终结果，保持与弹窗模式一致的遮罩反转
                        cached_images, cached_mask = cached_data["final_result"]
                        return (cached_images, 1 - cached_mask)
                
                # 没有缓存或输入改变，返回原图和全白遮罩
                batch_size, height, width, channels = images.shape
                default_mask = torch.ones((batch_size, height, width), dtype=torch.float32)
                return (images, default_mask)
            
            # 正常预览遮罩编辑流程
            event = Event()
            preview_urls = []
            should_send_to_frontend = True
            
            if node_id in cache and current_hash:
                cached_data = cache[node_id]
                if cached_data.get("input_hash") == current_hash and cached_data.get("output_urls"):
                    preview_urls = cached_data["output_urls"]
                    should_send_to_frontend = True
                else:
                    preview_result = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
                    preview_urls = preview_result["ui"]["images"] if preview_result and "ui" in preview_result else []
            else:
                preview_result = self.save_images(images, filename_prefix, prompt, extra_pnginfo)
                preview_urls = preview_result["ui"]["images"] if preview_result and "ui" in preview_result else []
            get_bridge_storage()[node_id] = {
                "event": event,
                "result": None,
                "original_images": images,
                "processing_complete": False,
                "input_hash": current_hash
            }
            
            try:
                PromptServer.instance.send_sync("bridge_preview_update", {
                    "node_id": node_id,
                    "urls": preview_urls
                })
                
                if not event.wait(timeout=30):
                    if node_id in get_bridge_storage():
                        del get_bridge_storage()[node_id]
                    batch_size, height, width, channels = images.shape
                    default_mask = torch.ones((batch_size, height, width), dtype=torch.float32)
                    return (images, default_mask)
                
                result_data = None
                if node_id in get_bridge_storage():
                    result_data = get_bridge_storage()[node_id]["result"]
                    del get_bridge_storage()[node_id]
                
                batch_size, height, width, channels = images.shape
                default_mask = torch.ones((batch_size, height, width), dtype=torch.float32)
                
                if result_data is not None and isinstance(result_data, tuple) and len(result_data) == 2:
                    final_images, final_mask = result_data
                    if current_hash:
                        # 缓存时使用原始遮罩
                        output_result = self.save_output_images(final_images, final_mask, filename_prefix + "_output", prompt, extra_pnginfo)
                        output_urls = output_result["ui"]["images"] if output_result and "ui" in output_result else []
                        cache[node_id] = {
                            "input_hash": current_hash,
                            "output_urls": output_urls,
                            "final_result": (final_images, final_mask)  # 缓存原始遮罩
                        }
                    # 返回时反转遮罩，但不影响缓存
                    return (final_images, 1 - final_mask)
                else:
                    return (images, default_mask)
                    
            except Exception as e:
                if node_id in get_bridge_storage():
                    del get_bridge_storage()[node_id]
                batch_size, height, width, channels = images.shape
                default_mask = torch.ones((batch_size, height, width), dtype=torch.float32)
                return (images, default_mask)
                
        except Exception as e:
            batch_size, height, width, channels = images.shape
            default_mask = torch.ones((batch_size, height, width), dtype=torch.float32)
            return (images, default_mask)
@PromptServer.instance.routes.post("/bridge_preview/confirm")
async def confirm_bridge_preview(request):
    """处理前端确认遮罩操作完成"""
    try:
        data = await request.json()
        node_id = str(data.get("node_id"))
        file_info = data.get("file_info")
        if node_id not in get_bridge_storage():
            return web.json_response({"success": False, "error": "节点未找到或已超时"})
        try:
            node_info = get_bridge_storage()[node_id]
            if file_info:
                processed_image, mask_image = load_processed_image(file_info)
                if processed_image is not None and mask_image is not None:
                    node_info["result"] = (processed_image, mask_image)
            node_info["processing_complete"] = True
            node_info["event"].set()
            return web.json_response({"success": True})
        except Exception as e:
            if node_id in get_bridge_storage() and "event" in get_bridge_storage()[node_id]:
                get_bridge_storage()[node_id]["event"].set()
            return web.json_response({"success": False, "error": str(e)})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})
@PromptServer.instance.routes.post("/bridge_preview/cancel")
async def cancel_bridge_preview(request):
    """取消桥接预览操作"""
    try:
        data = await request.json()
        node_id = str(data.get("node_id"))
        if node_id in get_bridge_storage():
            get_bridge_storage()[node_id]["event"].set()
            return web.json_response({"success": True, "message": f"节点 {node_id} 已取消"})
        else:
            return web.json_response({"success": False, "error": f"节点 {node_id} 未找到或已超时"})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)})
def load_processed_image(file_info):
    """从文件信息加载处理后的图片，返回图像和遮罩"""
    try:
        filename = None
        subfolder = ""
        file_type = "output"
        
        if isinstance(file_info, dict):
            filename = file_info.get("filename")
            subfolder = file_info.get("subfolder", "")
            file_type = file_info.get("type", "output")
        elif isinstance(file_info, str):
            # 尝试解析新版本字符串格式: "path/filename.ext [type]"
            file_info_str = str(file_info).strip()
            
            # 查找最后的 [type] 部分
            if '[' in file_info_str and file_info_str.endswith(']'):
                # 分离路径和类型
                last_bracket = file_info_str.rfind('[')
                path_part = file_info_str[:last_bracket].strip()
                type_part = file_info_str[last_bracket+1:-1].strip()
                
                # 解析路径部分
                if '/' in path_part:
                    # 包含子文件夹
                    path_parts = path_part.split('/')
                    filename = path_parts[-1]  # 最后一部分是文件名
                    subfolder = '/'.join(path_parts[:-1])  # 前面的部分是子文件夹
                else:
                    # 只有文件名
                    filename = path_part
                    subfolder = ""
                
                # 解析类型
                file_type = type_part.lower() if type_part else "output"
            else:
                # 没有类型信息，直接作为路径处理
                if '/' in file_info_str:
                    path_parts = file_info_str.split('/')
                    filename = path_parts[-1]
                    subfolder = '/'.join(path_parts[:-1])
                else:
                    filename = file_info_str
                    subfolder = ""
                file_type = "output"  # 默认类型
        else:
            return None, None
            
        if not filename:
            return None, None
        if file_type == "input":
            base_dir = folder_paths.get_input_directory()
        elif file_type == "output":
            base_dir = folder_paths.get_output_directory()
        elif file_type == "temp":
            base_dir = folder_paths.get_temp_directory()
        else:
            base_dir = folder_paths.get_output_directory()
        if subfolder:
            file_path = os.path.join(base_dir, subfolder, filename)
        else:
            file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            image = Image.open(file_path)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            np_image = np.array(image)
            rgb_image = np_image[:, :, :3]
            alpha_channel = np_image[:, :, 3]
            tensor_image = torch.from_numpy(rgb_image / 255.0).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(alpha_channel / 255.0).float().unsqueeze(0)
            return tensor_image, mask_tensor
        else:
            return None, None
    except Exception as e:
        pass
    return None, None
NODE_CLASS_MAPPINGS = {
    "BridgePreviewNode": BridgePreviewNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BridgePreviewNode": "🎈LG_PreviewBridge_V2"
}