import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { MultiButtonWidget } from "./multi_button_widget.js";

async function loadLatestImage(node, folder_type) {
	// 获取指定目录中的最新图片
	const res = await api.fetchApi(`/lg/get/latest_image?type=${folder_type}`, { cache: "no-store" });
	if (res.status == 200) {
		const item = await res.json();
		if (item && item.filename) {
			const imageWidget = node.widgets.find(w => w.name === 'image');
			if (!imageWidget) return false;
			
			// 保存文件信息的JSON字符串
			const fileInfo = JSON.stringify({
				filename: item.filename,
				subfolder: item.subfolder || '',
				type: item.type || 'temp'
			});
			
			// 保存到节点属性和 widget
			node._latestFileInfo = fileInfo;
			imageWidget.value = fileInfo;
			
			// 加载并显示图像
			const image = new Image();
			image.src = api.apiURL(`/view?filename=${item.filename}&type=${item.type}&subfolder=${item.subfolder || ''}`);
			node._imgs = [image];
			return true;
		}
	}
	return false;
}

app.registerExtension({
    name: "Comfy.LG.CachePreview",
    
    nodeCreated(node, app) {
        if (node.comfyClass !== "CachePreviewBridge") return;
        
        let imageWidget = node.widgets.find(w => w.name === 'image');
        if (!imageWidget) return;
        
        // 存储当前文件信息
        node._latestFileInfo = null;
        
        // 重写序列化方法，确保执行时使用最新值
        imageWidget.serializeValue = function(nodeId, widgetIndex) {
            if (node._latestFileInfo) {
                return node._latestFileInfo;
            }
            return this.value || "";
        };
        
        node._imgs = [new Image()];
        node.imageIndex = 0;
        
        // 使用多按钮组件创建刷新按钮
        const refreshWidget = node.addCustomWidget(MultiButtonWidget(app, "Refresh From", {
            labelWidth: 80,
            buttonSpacing: 4
        }, [
            {
                text: "Temp",
                callback: () => {
                    loadLatestImage(node, "temp").then(success => {
                        if (success) {
                            app.graph.setDirtyCanvas(true);
                        }
                    });
                }
            },
            {
                text: "Output",
                callback: () => {
                    loadLatestImage(node, "output").then(success => {
                        if (success) {
                            app.graph.setDirtyCanvas(true);
                        }
                    });
                }
            }
        ]));
        refreshWidget.serialize = false;
        
        // 重写 imgs 属性，处理来自 MaskEditor 的粘贴
        Object.defineProperty(node, 'imgs', {
            set(v) {
                if (!v || v.length === 0) return;
                
                const stackTrace = new Error().stack;
                
                // 来自 MaskEditor 的粘贴
                if (stackTrace.includes('pasteFromClipspace')) {
                    if (v[0] && v[0].src) {
                        const urlParts = v[0].src.split("?");
                        if (urlParts.length > 1) {
                            const sp = new URLSearchParams(urlParts[1]);
                            const filename = sp.get('filename');
                            const type = sp.get('type') || 'input';
                            const subfolder = sp.get('subfolder') || '';
                            
                            if (filename) {
                                // 保存文件信息的JSON字符串
                                const fileInfo = JSON.stringify({
                                    filename: filename,
                                    subfolder: subfolder,
                                    type: type
                                });
                                
                                // 保存到节点属性，序列化时会使用
                                node._latestFileInfo = fileInfo;
                                imageWidget.value = fileInfo;
                                
                                // 直接使用传入的图像
                                node._imgs = v;
                                app.graph.setDirtyCanvas(true);
                                
                                return;
                            }
                        }
                    }
                }
                
                // 其他情况直接设置
                node._imgs = v;
            },
            get() {
                return node._imgs;
            }
        });
    }
});
