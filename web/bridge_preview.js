import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
const waitingNodes = new Map();
const processedNodes = new Set();
function updateNodePreview(nodeId, fileInfo) {
    const node = app.graph.getNodeById(parseInt(nodeId));
    if (!node || !fileInfo) return;
    const imageUrl = fileInfo.subfolder
        ? `/view?filename=${encodeURIComponent(fileInfo.filename)}&subfolder=${encodeURIComponent(fileInfo.subfolder)}&type=${fileInfo.type || 'input'}`
        : `/view?filename=${encodeURIComponent(fileInfo.filename)}&type=${fileInfo.type || 'input'}`;
    node.images = [{
        filename: fileInfo.filename,
        subfolder: fileInfo.subfolder || "",
        type: fileInfo.type || "input",
        url: imageUrl
    }];
    if (node.onDrawBackground) {
        const img = new Image();
        img.onload = () => {
            node.imgs = [img];
            app.graph.setDirtyCanvas(true);
        };
        img.src = imageUrl;
    }
    updateStatusWidget(node, `${nodeId}_${fileInfo.filename}`);
}
function updateStatusWidget(node, statusText) {
    const statusWidget = node.widgets?.find(w => w.name === "file_info");
    if (statusWidget) {
        statusWidget.value = statusText;
        app.graph.setDirtyCanvas(true);
        app.graph.change();
    }
}
api.addEventListener("bridge_preview_update", (event) => {
    const { node_id, urls } = event.detail;
    const node = app.graph._nodes_by_id[node_id];
    if (!node || !urls?.length) return;
    waitingNodes.set(node_id, { urls, timestamp: Date.now(), node });
    const imageData = urls.map((url, index) => ({
        index,
        filename: url.filename,
        subfolder: url.subfolder,
        type: url.type
    }));
    node.imageData = imageData;
    node.imgs = [];
    imageData.forEach((imgData, i) => {
        const img = new Image();
        img.onload = () => {
            app.graph.setDirtyCanvas(true);
            if (i === imageData.length - 1) {
                setupClipspace(node_id, urls);
            }
        };
        img.src = `/view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${imgData.subfolder || ''}&${app.getPreviewFormatParam()}`;
        node.imgs.push(img);
    });
    node.setSizeForImage?.();
    node.update?.();
});
function setupClipspace(nodeId, urls) {
    const ComfyApp = app.constructor;
    if (!ComfyApp.clipspace) ComfyApp.clipspace = {};
    if (!app.clipspace) app.clipspace = {};
    const images = urls.map(url => ({
        filename: url.filename,
        subfolder: url.subfolder || "",
        type: url.type || "output"
    }));
    const imgs = urls.map(url => ({
        src: `${window.location.origin}/view?filename=${encodeURIComponent(url.filename)}&type=${url.type}&subfolder=${encodeURIComponent(url.subfolder || '')}`,
        filename: url.filename,
        subfolder: url.subfolder || "",
        type: url.type || "output"
    }));
    [ComfyApp.clipspace, app.clipspace].forEach(clipspace => {
        clipspace.images = images;
        clipspace.imgs = imgs;
        clipspace.selectedIndex = 0;
    });
    setTimeout(() => {
        const node = app.graph.getNodeById(parseInt(nodeId));
        if (!node) return;
        
        app.canvas.selectNode(node);
        
        let success = false;
        
        try {
            if (app.extensionManager?.command?.execute) {
                app.extensionManager.command.execute('Comfy.MaskEditor.OpenMaskEditor');
                success = true;
            }
        } catch (error) {
            // Silently fail and try next method
        }
        
        if (!success) {
            try {
                const ComfyApp = app.constructor;
                const openMaskEditor = ComfyApp.open_maskeditor || app.open_maskeditor;
                if (openMaskEditor && typeof openMaskEditor === 'function') {
                    openMaskEditor();
                    success = true;
                }
            } catch (error) {
                // Silently fail
            }
        }
        
        bindCancelButton();
    }, 100);
}
function bindCancelButton() {
    const checkInterval = setInterval(() => {
        const maskEditor = findMaskEditor();
        if (!maskEditor) return;
        
        const cancelButtons = Array.from(maskEditor.querySelectorAll('button')).filter(btn => {
            const text = btn.textContent.trim().toLowerCase();
            return text === 'cancel' || text === '取消' || text.includes('cancel') || text.includes('取消');
        });
        
        if (cancelButtons.length > 0) {
            cancelButtons.forEach(button => {
                if (!button.hasAttribute('data-bridge-bound')) {
                    button.setAttribute('data-bridge-bound', 'true');
                    button.addEventListener('click', () => {
                        setTimeout(handleMaskEditorCancel, 50);
                    }, { capture: true });
                }
            });
            clearInterval(checkInterval);
        }
    }, 300);
    setTimeout(() => clearInterval(checkInterval), 10000);
}
function findMaskEditor() {
    const newMaskEditor = document.querySelector('.mask-editor-dialog');
    if (newMaskEditor) {
        return newMaskEditor;
    }
    
    const modals = document.querySelectorAll('div.comfy-modal, .comfy-modal, [class*="modal"]');
    for (const modal of modals) {
        if (modal.querySelector('canvas') && modal.style.display !== 'none') {
            return modal;
        }
    }
    
    const elements = document.querySelectorAll('*');
    for (const element of elements) {
        const buttons = element.querySelectorAll('button');
        if (buttons.length >= 2 && element.querySelector('canvas')) {
            const buttonTexts = Array.from(buttons).map(btn => btn.textContent.trim().toLowerCase());
            if (buttonTexts.some(text => text.includes('cancel') || text.includes('取消')) &&
                buttonTexts.some(text => text.includes('save') || text.includes('保存'))) {
                return element;
            }
        }
    }
    
    return null;
}
function handleMaskEditorCancel() {
    waitingNodes.forEach((nodeInfo, nodeId) => {
        sendCancelSignal(nodeId);
    });
    
    waitingNodes.clear();
    processedNodes.clear();
}
async function sendCancelSignal(nodeId) {
    try {
        await api.fetchApi("/bridge_preview/cancel", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ node_id: String(nodeId) })
        });
    } catch (error) {
        console.error(`[BridgePreview] Failed to send cancel signal:`, error);
    }
}
const originalFetch = api.fetchApi;
api.fetchApi = async function(url, options) {
    const result = await originalFetch.call(this, url, options);
    if (url === "/upload/mask" && result.ok) {
        await handleMaskUpload(result);
    }
    return result;
};
async function handleMaskUpload(result) {
    try {
        const responseData = await result.clone().json();
        const fileInfo = responseData?.name ? {
            filename: responseData.name,
            subfolder: responseData.subfolder || "clipspace",
            type: responseData.type || "input"
        } : null;
        if (!fileInfo) return;
        let latestNodeId = null;
        let latestTimestamp = 0;
        for (const [nodeId, nodeInfo] of waitingNodes) {
            if (nodeInfo.timestamp > latestTimestamp) {
                latestTimestamp = nodeInfo.timestamp;
                latestNodeId = nodeId;
            }
        }
        if (!latestNodeId) return;
        try {
            updateNodePreview(latestNodeId, fileInfo);
            const confirmResponse = await originalFetch.call(api, "/bridge_preview/confirm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    node_id: String(latestNodeId),
                    file_info: fileInfo
                })
            });
            if (confirmResponse.ok) {
                processedNodes.add(String(latestNodeId));
                setTimeout(() => {
                    waitingNodes.delete(latestNodeId);
                }, 1000);
            }
        } catch (error) {
            console.error(`[BridgePreview] Failed to process node ${latestNodeId}:`, error);
        }
    } catch (error) {
        console.error(`[BridgePreview] Failed to handle mask result:`, error);
    }
}
setInterval(() => {
    const now = Date.now();
    for (const [nodeId, nodeInfo] of waitingNodes) {
        if (now - nodeInfo.timestamp > 30000) {
            waitingNodes.delete(nodeId);
            processedNodes.delete(String(nodeId));
        }
    }
}, 5000);
console.log("[BridgePreview] Bridge preview module loaded");