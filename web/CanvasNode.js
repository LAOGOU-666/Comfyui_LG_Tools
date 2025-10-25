import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fabric } from "./lib/fabric-slim.min.js";  // 使用精简版fabric.js以提高性能
import { queueManager } from "./queue_shortcut.js";

// 定义常量
const CANVAS_SIZE = {
    WIDTH: 420,
    HEIGHT: 200,
    BOTTOM_MARGIN: 110,  // 节点下沿扩展值
    RIGHT_MARGIN: 20,  // 节点右沿扩展值
    CONTROL_PANEL_HEIGHT: 30  // 控制面板高度
};

// 模块级变量
const instances = [];
let eventHandlersInitialized = false;
let globalUpdateHandler, globalGetStateHandler;

// 模块级函数 - 清理指定节点ID的实例
function cleanup(nodeId) {
    const index = instances.findIndex(inst => inst.node?.id === nodeId);
    if (index !== -1) {
        const instance = instances[index];
        
        // 从数组中移除
        instances.splice(index, 1);
        
        // 清理核心资源
        if (instance.canvas) {
            instance.canvas.off();
            instance.canvas.dispose();
        }
        if (instance.canvasContainer?.parentNode) {
            instance.canvasContainer.parentNode.removeChild(instance.canvasContainer);
        }
        return true;
    }
    return false;
}

// 模块级函数 - 设置全局事件处理
function setupGlobalEventHandling() {
    // 如果已经设置了全局处理，则不重复设置
    if (eventHandlersInitialized) return;
    
    // 设置全局事件处理函数
    globalUpdateHandler = async (event) => {
        const data = event.detail;
        if (!data || !data.node_id) return;
        
        const nodeId = data.node_id;
        const instance = instances.find(inst => inst.node?.id.toString() === nodeId);
        
        if (!instance) return;

        if (data.canvas_data) {
            await instance.updateCanvas(data.canvas_data, data.transform_data);
        } else {
            await instance.sendCanvasState();
        }
    };
    
    globalGetStateHandler = async (event) => {
        const data = event.detail;
        if (!data || !data.node_id) return;
        
        const nodeId = data.node_id;
        const instance = instances.find(inst => inst.node?.id.toString() === nodeId);
        
        if (instance) {
            await instance.sendCanvasState();
        }
    };
    
    // 添加全局事件监听器
    api.addEventListener("fast_canvas_update", globalUpdateHandler);
    api.addEventListener("fast_canvas_get_state", globalGetStateHandler);
    
    eventHandlersInitialized = true;
}
class FastCanvas {
    constructor(node, initialSize = null) {
        this.node = node;
        this.lastCanvasState = null; 
        
        // 使用传入的初始尺寸或默认尺寸
        this.originalSize = initialSize || {
            width: CANVAS_SIZE.WIDTH,
            height: CANVAS_SIZE.HEIGHT
        };
        
        // 添加到实例数组
        instances.push(this);
        
        // 确保全局事件处理已设置
        setupGlobalEventHandling();
        
        // 其他初始化代码...
        this.currentSize = { ...this.originalSize };
        this.maxDisplaySize = 768;
        // 创建缩放值显示元素
        this.scaleText = document.createElement('div');
        this.scaleText.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.5);
            color: white;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
        `;
        this.initCanvas();
        this.setupDragAndDrop();
        this.setupPaste();
    }

    cleanup() {
        try {

            if (this.node?.id) {
                cleanup(this.node.id);
            } else {
                const index = instances.indexOf(this);
                if (index !== -1) {
                    instances.splice(index, 1);
                }
            }
        } catch (e) {
            console.error('[清理错误]', e);
        }
    }

    initCanvas() {
        this.canvasContainer = document.createElement('div');
        this.canvasContainer.className = 'fast-canvas-container';
        
        // 创建 canvas 元素
        const canvasElement = document.createElement('canvas');
        
        // 使用原始尺寸创建画布
        this.canvas = new fabric.Canvas(canvasElement, {
            width: this.originalSize.width,
            height: this.originalSize.height,
            preserveObjectStacking: true,
            selection: true
        });
        // 设置选中框样式
        this.canvas.selectionColor = 'rgba(0, 123, 255, 0.3)';  // 选中区域填充色
        this.canvas.selectionBorderColor = '#007bff';  // 选中框边框颜色
        this.canvas.selectionLineWidth = 2;  // 选中框边框宽度
        
        // 设置控制点样式
        fabric.Object.prototype.set({
            transparentCorners: false,  // 控制点不透明
            cornerColor: '#007bff',  // 控制点颜色
            cornerSize: 20,  // 控制点大小
            cornerStyle: 'circle',  // 控制点形状为圆形
            cornerStrokeColor: '#ffffff',  // 控制点边框颜色
            cornerStrokeWidth: 2,  // 控制点边框宽度
            padding: 5,  // 选中框内边距
            borderColor: '#007bff',  // 边框颜色
            borderScaleFactor: 2,  // 边框宽度
            hasRotatingPoint: true,  // 显示旋转控制点
            rotatingPointOffset: 30  // 旋转控制点偏移距离
        });

        // 创建默认背景图像
        const defaultBackground = new fabric.Rect({
            width: this.originalSize.width,
            height: this.originalSize.height,
            left: 0,
            top: 0,
            fill: '#000000',
            selectable: false,
            evented: false,
            excludeFromExport: false 
        });
        
        // 设置为背景图像
        this.canvas.setBackgroundImage(defaultBackground, () => {
            this.canvas.renderAll();
        });
        
        this.canvasContainer.style.cssText = `
            position: relative;
            width: ${this.originalSize.width}px;
            height: ${this.originalSize.height + CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px;
            background: transparent; // 改为透明背景
        `;
        
        // 创建画布包装容器
        const canvasWrapper = document.createElement('div');
        canvasWrapper.style.cssText = `
            width: 100%;
            height: ${this.originalSize.height}px;
        `;
        canvasWrapper.appendChild(this.canvas.wrapperEl);
        this.canvasContainer.appendChild(canvasWrapper);
        this.canvasContainer.appendChild(this.scaleText);
        
        // 创建控制面板
        this.controlPanel = new ControlPanel(this.canvas, this.node);
        this.canvasContainer.appendChild(this.controlPanel.getContainer());
        
        // 初始化状态
        this.layers = new Map();
        this.background = defaultBackground;  // 保存背景引用
        this.isDragging = false;
        this.isLocked = false;
        this.initContextMenu();
        // 设置事件监听
        this.setupEventListeners();
        // 设置节点的初始尺寸
        this.node.size = [
            this.originalSize.width + CANVAS_SIZE.RIGHT_MARGIN,
            this.originalSize.height + CANVAS_SIZE.CONTROL_PANEL_HEIGHT + CANVAS_SIZE.BOTTOM_MARGIN
        ];
    }

    initContextMenu() {
        // 创建菜单元素
        const menu = document.createElement('div');
        menu.className = 'fast-canvas-context-menu';
        menu.style.cssText = `
            position: fixed;
            display: none;
            background: #2a2a2a;
            border: 1px solid #3f3f3f;
            border-radius: 6px;
            padding: 3px;
            z-index: 1000;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: background-color 0.2s ease;
            min-width: 120px;
        `;

        const menuItems = [
            { text: '设为背景', action: () => this.setLayerAsBackground() },
            {
                text: '变换',
                submenu: [
                    { text: '水平翻转', action: () => this.flipObject('horizontal') },
                    { text: '垂直翻转', action: () => this.flipObject('vertical') },
                    { text: '居中', action: () => this.centerObject('both') },
                    { text: '水平居中', action: () => this.centerObject('horizontal') },
                    { text: '垂直居中', action: () => this.centerObject('vertical') }
                ]
            },
            {
                text: '调整图层',
                submenu: [
                    { text: '上移一层', action: () => {
                        const obj = this.canvas.getActiveObject();
                        if (obj) {
                            obj.bringForward();
                            this.canvas.renderAll();

                        }
                    }},
                    { text: '下移一层', action: () => {
                        const obj = this.canvas.getActiveObject();
                        if (obj) {
                            obj.sendBackwards();
                            this.canvas.renderAll();

                        }
                    }},
                    { text: '置于顶层', action: () => {
                        const obj = this.canvas.getActiveObject();
                        if (obj) {
                            obj.bringToFront();
                            this.canvas.renderAll();

                        }
                    }},
                    { text: '置于底层', action: () => {
                        const obj = this.canvas.getActiveObject();
                        if (obj) {
                            obj.sendToBack();
                            this.canvas.renderAll();

                        }
                    }}
                ]
            },
            {
                text: '混合模式',
                submenu: [
                    { text: '正常 (source-over)', action: () => this.setObjectCompositeMode('source-over') },
                    { text: '正片叠底 (multiply)', action: () => this.setObjectCompositeMode('multiply') },
                    { text: '滤色 (screen)', action: () => this.setObjectCompositeMode('screen') },
                    { text: '叠加 (overlay)', action: () => this.setObjectCompositeMode('overlay') },
                    { text: '变亮 (lighten)', action: () => this.setObjectCompositeMode('lighten') },
                    { text: '变暗 (darken)', action: () => this.setObjectCompositeMode('darken') },
                    { text: '强光 (hard-light)', action: () => this.setObjectCompositeMode('hard-light') },
                    { text: '柔光 (soft-light)', action: () => this.setObjectCompositeMode('soft-light') },
                    { text: '差值 (difference)', action: () => this.setObjectCompositeMode('difference') },
                    { text: '排除 (exclusion)', action: () => this.setObjectCompositeMode('exclusion') },
                ]
            },
            { text: '调整透明度', action: () => this.showOpacityDialog() },
            { text: '恢复原尺寸', action: () => this.resetObjectSize() }
        ];
        
        // 创建菜单项
        menuItems.forEach(item => {
            const menuItem = document.createElement('div');
            menuItem.textContent = item.text;
            menuItem.style.cssText = `
                padding: 6px 15px;
                margin: 1px 3px;
                cursor: pointer;
                white-space: nowrap;
                transition: background-color 0.15s ease;
                font-size: 13px;
                border-radius: 4px;
                position: relative;
            `;
        
            if (item.submenu) {
                // 添加子菜单指示器
                menuItem.style.paddingRight = '25px';
                const arrow = document.createElement('span');
                arrow.textContent = '▶';
                arrow.style.cssText = `
                    position: absolute;
                    right: 8px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 10px;
                `;
                menuItem.appendChild(arrow);
        
                // 创建子菜单
                const submenu = document.createElement('div');
                submenu.style.cssText = `
                    position: absolute;
                    left: 100%;
                    top: 0;
                    display: none;
                    background: #2a2a2a;
                    border: 1px solid #3f3f3f;
                    border-radius: 6px;
                    padding: 3px;
                    z-index: 1001;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                `;
        
                // 创建子菜单项
                item.submenu.forEach(subItem => {
                    const subMenuItem = document.createElement('div');
                    subMenuItem.textContent = subItem.text;
                    subMenuItem.style.cssText = `
                        padding: 6px 15px;
                        margin: 1px 3px;
                        cursor: pointer;
                        white-space: nowrap;
                        transition: background-color 0.15s ease;
                        font-size: 13px;
                        border-radius: 4px;
                    `;
                    subMenuItem.onmouseover = () => subMenuItem.style.backgroundColor = '#3f3f3f';
                    subMenuItem.onmouseout = () => subMenuItem.style.backgroundColor = '';
                    subMenuItem.onclick = (e) => {
                        e.stopPropagation();
                        subItem.action();
                        menu.style.display = 'none';
                    };
                    submenu.appendChild(subMenuItem);
                });
        
                // 显示/隐藏子菜单
                menuItem.onmouseover = () => {
                    submenu.style.display = 'block';
                    menuItem.style.backgroundColor = '#3f3f3f';
                };
                menuItem.onmouseout = (e) => {
                    if (!e.relatedTarget || !submenu.contains(e.relatedTarget)) {
                        submenu.style.display = 'none';
                        menuItem.style.backgroundColor = '';
                    }
                };
                submenu.onmouseover = () => {
                    submenu.style.display = 'block';
                    menuItem.style.backgroundColor = '#3f3f3f';
                };
                submenu.onmouseout = (e) => {
                    if (!menuItem.contains(e.relatedTarget)) {
                        submenu.style.display = 'none';
                        menuItem.style.backgroundColor = '';
                    }
                };
        
                menuItem.appendChild(submenu);
            } else {
                // 普通菜单项的鼠标悬停效果
                menuItem.onmouseover = () => menuItem.style.backgroundColor = '#3f3f3f';
                menuItem.onmouseout = () => menuItem.style.backgroundColor = '';
                menuItem.onclick = (e) => {
                    e.stopPropagation();
                    item.action();
                    menu.style.display = 'none';
                };
            }
        
            menu.appendChild(menuItem);
        });
        // 在画布容器上监听右键事件
        this.canvasContainer.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const activeObject = this.canvas.getActiveObject();
            if (activeObject) {
                menu.style.left = `${e.pageX}px`;
                menu.style.top = `${e.pageY}px`;
                menu.style.display = 'block';
                
                // 添加一次性点击事件监听器
                const closeMenu = (e) => {
                    if (!menu.contains(e.target)) {
                        menu.style.display = 'none';
                        document.removeEventListener('click', closeMenu);
                        document.removeEventListener('contextmenu', closeMenu);
                    }
                };
                
                // 延迟添加事件监听器，避免立即触发
                setTimeout(() => {
                    document.addEventListener('click', closeMenu);
                    document.addEventListener('contextmenu', closeMenu);
                }, 0);
            }
            
            return false;
        }, true);

        // 阻止菜单内右键点击
        menu.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            e.stopPropagation();
            return false;
        });

        document.body.appendChild(menu);
    }

    // 菜单操作方法
    flipObject(direction) {
        const activeObject = this.canvas.getActiveObject();
        if (activeObject) {
            if (direction === 'horizontal') {
                activeObject.flipX = !activeObject.flipX;
            } else {
                activeObject.flipY = !activeObject.flipY;
            }
            this.canvas.renderAll();
        }
    }

    centerObject(direction) {
        const activeObject = this.canvas.getActiveObject();
        if (activeObject) {
            if (direction === 'both') {
                activeObject.center();
            } else if (direction === 'horizontal') {
                activeObject.centerH();
            } else {
                activeObject.centerV();
            }
            activeObject.setCoords();
            this.canvas.renderAll();
        }
    }

    resetObjectSize() {
        const activeObject = this.canvas.getActiveObject();
        if (activeObject && activeObject.originalWidth && activeObject.originalHeight) {
            // 重置对象到原始尺寸
            const originalWidth = activeObject.originalWidth;
            const originalHeight = activeObject.originalHeight;
            
            // 计算比例
            const scaleX = originalWidth / activeObject.width;
            const scaleY = originalHeight / activeObject.height;
            
            // 应用变换
            activeObject.set({
                scaleX: 1,
                scaleY: 1,
                width: originalWidth,
                height: originalHeight
            });
            
            this.canvas.renderAll();

        }
    }
    
    // 设置图层混合模式
    setObjectCompositeMode(mode) {
        const activeObject = this.canvas.getActiveObject();
        if (activeObject) {
            activeObject.set({
                globalCompositeOperation: mode
            });
            this.canvas.renderAll();

        }
    }
    
    // 显示透明度调整对话框
    showOpacityDialog() {
        const activeObject = this.canvas.getActiveObject();
        if (!activeObject) return;
        
        // 创建对话框元素
        const dialog = document.createElement('div');
        dialog.className = 'opacity-dialog';
        dialog.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #2a2a2a;
            border: 1px solid #3f3f3f;
            border-radius: 6px;
            padding: 15px;
            z-index: 1000;
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            min-width: 250px;
        `;
        
        // 创建滑动条
        const slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '0';
        slider.max = '100';
        slider.value = Math.round(activeObject.opacity * 100);
        slider.style.cssText = `
            width: 100%;
            margin: 5px 0;
        `;
        
        // 当前透明度值显示
        const valueDisplay = document.createElement('div');
        valueDisplay.style.cssText = `
            margin: 5px 0;
            text-align: center;
            font-size: 14px;
        `;
        valueDisplay.textContent = `${Math.round(activeObject.opacity * 100)}%`;
        
        // 滑动时实时更新透明度
        slider.addEventListener('input', (e) => {
            const opacity = parseInt(e.target.value) / 100;
            activeObject.set({ opacity });
            valueDisplay.textContent = `${Math.round(opacity * 100)}%`;
            this.canvas.renderAll();
        });
        
        // 按钮容器
        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = `
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
        `;
        
        // 关闭对话框函数
        const closeDialog = () => {
            activeObject.set({ opacity: activeObject._originalOpacity });
            try {
                document.body.removeChild(dialog);
                // 移除ESC键监听器
                document.removeEventListener('keydown', handleKeyDown);
            } catch (error) {
                console.log("对话框已被移除");
            }
            this.canvas.renderAll();
        };
        
        // 应用更改并关闭对话框
        const applyChanges = () => {
            try {
                document.body.removeChild(dialog);
                // 移除所有事件监听器
                document.removeEventListener('keydown', handleKeyDown);
                document.removeEventListener('mousedown', handleClickOutside);
            } catch (error) {
                console.log("对话框已被移除");
            }

        };
        
        // ESC键监听器
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') {
                closeDialog();
            }
        };
        
        // 添加ESC键监听
        document.addEventListener('keydown', handleKeyDown);
        
        // 取消按钮
        const cancelButton = document.createElement('button');
        cancelButton.textContent = '取消';
        cancelButton.style.cssText = `
            background: #444;
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
        `;
        cancelButton.onclick = closeDialog;
        
        // 确认按钮
        const confirmButton = document.createElement('button');
        confirmButton.textContent = '确认';
        confirmButton.style.cssText = `
            background: #007bff;
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
        `;
        confirmButton.onclick = applyChanges;
        
        // 添加所有元素
        dialog.appendChild(valueDisplay);
        dialog.appendChild(slider);
        buttonContainer.appendChild(cancelButton);
        buttonContainer.appendChild(confirmButton);
        dialog.appendChild(buttonContainer);
        
        // 保存原始透明度以便取消时恢复
        activeObject._originalOpacity = activeObject.opacity;
        
        // 添加到文档
        document.body.appendChild(dialog);
        
        // 在函数内部创建并存储mousedown监听器的引用
        let handleClickOutside;

        // 定义并存储监听器引用
        handleClickOutside = function(e) {
            if (!dialog.contains(e.target)) {
                closeDialog();
                document.removeEventListener('mousedown', handleClickOutside);
            }
        };

        // 添加监听器
        document.addEventListener('mousedown', handleClickOutside);
    }

    updateContainerSize(width, height) {
        // 更新画布容器尺寸为缩放后的显示尺寸
        this.canvasContainer.style.width = `${width}px`;
        this.canvasContainer.style.height = `${height + CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px`;
        
        // 确保控制面板始终在最上层
        this.canvasContainer.style.position = 'relative'; // 添加这行
        this.canvasContainer.style.zIndex = '1'; // 添加这行
        
        // 更新控制面板尺寸
        this.controlPanel.updateSize(width);
        
        // 更新节点的 DOM 元素尺寸
        if (this.node.canvasElement) {
            this.node.canvasElement.style.width = `${width}px`;
            this.node.canvasElement.style.height = `${height + CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px`;
        }
        
        // 更新节点尺寸
        this.node.size = [
            width + CANVAS_SIZE.RIGHT_MARGIN, 
            height + CANVAS_SIZE.BOTTOM_MARGIN + CANVAS_SIZE.CONTROL_PANEL_HEIGHT
        ];
        
        // 强制更新节点
        this.node.setDirtyCanvas(true, true);
    }

    updateCanvasSize(displayWidth, displayHeight) {
        // 1. 计算缩放比例
        const scaleX = displayWidth / this.originalSize.width;
        const scaleY = displayHeight / this.originalSize.height;
        const scale = Math.min(scaleX, scaleY);  // 使用统一的缩放比例
        
        // 2. 设置画布为原始尺寸
        this.canvas.setDimensions({
            width: this.originalSize.width,
            height: this.originalSize.height
        });
        
        // 3. 更新画布包装器的CSS transform
        this.canvas.wrapperEl.style.transform = `scale(${scale})`;
        this.canvas.wrapperEl.style.transformOrigin = 'top left';
        this.canvas.wrapperEl.style.width = `${displayWidth}px`;
        this.canvas.wrapperEl.style.height = `${displayHeight}px`;
        
        // 4. 同步Fabric.js的缩放
        this.canvas.setZoom(1);  // 重置zoom，因为我们使用CSS缩放
        
        // 5. 如果有背景图片，确保它使用原始尺寸
        if (this.canvas.backgroundImage) {
            this.canvas.backgroundImage.set({
                width: this.originalSize.width,
                height: this.originalSize.height,
                scaleX: 1,
                scaleY: 1,
                left: 0,
                top: 0,
                originX: 'left',
                originY: 'top'
            });
            // 强制更新背景坐标
            this.canvas.backgroundImage.setCoords();
        }
        
        // 6. 更新容器尺寸
        this.updateContainerSize(displayWidth, displayHeight);
        
        // 7. 重新渲染
        this.canvas.renderAll();
        
        // 8. 更新事件监听区域
        this.canvas.calcOffset();
    }
    // 修改 updateDisplayScale 方法
    updateDisplayScale(maxSize) {
        this.maxDisplaySize = maxSize;
        if (this.originalSize?.width && this.originalSize?.height) {
            const scaledSize = this.calculateScaledSize(
                this.originalSize.width, 
                this.originalSize.height, 
                maxSize
            );
            
            // 更新显示尺寸，但保持画布原始尺寸
            this.updateCanvasSize(scaledSize.width, scaledSize.height);
            
            // 确保画布完全重置和更新
            this.canvas.requestRenderAll();
        }
    }

    // 计算缩放后的尺寸
    calculateScaledSize(width, height, maxSize) {
        if (width <= maxSize && height <= maxSize) {
            return { width, height };
        }

        const ratio = width / height;
        if (width > height) {
            return {
                width: maxSize,
                height: Math.round(maxSize / ratio)
            };
        } else {
            return {
                width: Math.round(maxSize * ratio),
                height: maxSize
            };
        }
    }



    setupPaste() {
        // 移除可能存在的旧事件监听器
        if (this._pasteHandler) {
            document.removeEventListener('paste', this._pasteHandler);
        }
    
        let canvasActive = false;
    
        // 创建新的粘贴处理函数
        this._pasteHandler = async (e) => {
            if (!canvasActive) return;  // 如果画布不是活动状态，不处理粘贴
    
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (const item of items) {
                if (item.type.indexOf('image') === 0) {
                    e.preventDefault();
                    e.stopPropagation();
                    const blob = item.getAsFile();
                    await this.handleImageUpload(blob, { center: true, autoScale: true });
                    break;
                }
            }
        };
    
        // 添加内部复制功能
        document.addEventListener('keydown', (e) => {
            // 检查是否在画布区域内且画布处于活动状态
            const activeElement = document.activeElement;
            const isInputActive = activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA';
            if (isInputActive || !canvasActive) return;
    
            // 复制 (Ctrl+C)
            if (e.ctrlKey && e.key === 'c') {
                const activeObjects = this.canvas.getActiveObjects();
                if (activeObjects.length > 0) {
                    activeObjects.forEach(obj => {
                        // 创建图层的副本
                        obj.clone((cloned) => {
                            // 生成新的图层ID
                            const newId = this.controlPanel.maxLayerId + 1;
                            this.controlPanel.maxLayerId = newId;
    
                            cloned.set({
                                left: obj.left + 20,
                                top: obj.top + 20,
                                id: newId
                            });
                            
                            // 添加到画布和图层管理
                            this.canvas.add(cloned);
                            this.layers.set(newId, cloned);
                        });
                    });
                    
                    this.canvas.renderAll();
                    this.controlPanel.updateLayerSelector();

                }
            }
        });
    
        // 添加新的事件监听器到 document
        document.addEventListener('paste', this._pasteHandler, true);
    
        // 点击画布时设置活动状态
        this.canvasContainer.addEventListener('mousedown', () => {
            canvasActive = true;
        });
    
        // 点击其他地方时取消活动状态
        document.addEventListener('mousedown', (e) => {
            if (!this.canvasContainer.contains(e.target)) {
                canvasActive = false;
            }
        });
    
        // 清理函数
        this.cleanup = () => {
            document.removeEventListener('paste', this._pasteHandler, true);
        };
    }

    setupDragAndDrop() {
        // 防止浏览器默认的拖放行为
        this.canvasContainer.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvasContainer.classList.add('drag-over');
        });

        this.canvasContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        this.canvasContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.target === this.canvasContainer) {
                this.canvasContainer.classList.remove('drag-over');
            }
        });

        this.canvasContainer.addEventListener('drop', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvasContainer.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    await this.handleImageUpload(file, { center: true, autoScale: true });
                }
            }
        });


        // 添加样式
        const style = document.createElement('style');
        style.textContent = `
            .fast-canvas-container.drag-over::after {
                content: '拖放图片到这里';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: ${CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px;
                background: rgba(0, 0, 0, 0.7);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                pointer-events: none;
                z-index: 1000;
            }
        `;
        document.head.appendChild(style);
    }

    async handleImageUpload(file, options = { center: true, autoScale: true }) {
        try {
            const reader = new FileReader();
            
            const imageLoadPromise = new Promise((resolve, reject) => {
                reader.onload = (e) => {
                    const dataUrl = e.target.result;
                    resolve(dataUrl);
                };
                reader.onerror = reject;
            });
    
            reader.readAsDataURL(file);
            const imageData = await imageLoadPromise;
    
            // 统一按图层方式加载图片
            fabric.Image.fromURL(imageData, (img) => {
                if (options.center && options.autoScale) {
                    // 计算适合画布的缩放比例
                    const scale = Math.min(
                        this.canvas.width / img.width,
                        this.canvas.height / img.height
                    );
                    
                    // 计算居中位置
                    const canvasCenter = this.canvas.getCenter();
                    img.set({
                        scaleX: scale,
                        scaleY: scale,
                        left: canvasCenter.left,
                        top: canvasCenter.top,
                        originX: 'center',
                        originY: 'center'
                    });
                } else {
                    // 不进行居中和缩放，使用原始尺寸和位置
                    img.set({
                        left: 0,
                        top: 0,
                        originX: 'left',
                        originY: 'top'
                    });
                }

                // 添加到画布并更新图层ID
                this.canvas.add(img);
                this.controlPanel.updateLayerIds();
                
                this.canvas.setActiveObject(img);
                this.canvas.renderAll();

            });
    
        } catch (error) {
            console.error('图片上传失败:', error);
            alert('图片上传失败，请重试');
        }
    }


    
    setupEventListeners() {

        // 添加一个计算相对缩放值的辅助方法
        const getRelativeScale = (obj) => {
            const backgroundImage = this.canvas.backgroundImage;
            if (!backgroundImage) return obj.scaleX;
            
            // 计算背景图片的缩放比例
            const bgScaleRatio = backgroundImage.width / backgroundImage.getScaledWidth();
            // 计算相对于背景图片的实际缩放比例
            return Number((obj.scaleX * bgScaleRatio).toFixed(2));
        };
        // 创建一个绑定到当前实例的事件处理函数
        this._handleKeyDown = (e) => {
            // 首先检查画布和节点是否还存在
            if (!this.canvas || !this.node) {
                // 如果画布或节点不存在，移除事件监听器
                document.removeEventListener('keydown', this._handleKeyDown);
                return;
            }

            // 检查是否有选中的对象且当前焦点不在输入框上
            if (this.canvas.getActiveObject() && 
                document.activeElement.tagName !== 'INPUT' && 
                document.activeElement.tagName !== 'TEXTAREA') {
                
                // Delete 键或 Backspace 键
                if (e.key === 'Delete' || e.key === 'Backspace') {
                    e.preventDefault();
                    
                    const activeObject = this.canvas.getActiveObject();
                    // 确保不是背景图片
                    if (activeObject !== this.canvas.backgroundImage) {
                        // 如果是图层，从图层映射中移除
                        if (activeObject.id) {
                            this.layers.delete(activeObject.id);
                        }
                        
                        // 从画布中移除
                        this.canvas.remove(activeObject);
                        this.canvas.renderAll();
                        

                    }
                }
            }
        };

        // 添加滚轮缩放
        this.canvas.on('mouse:wheel', (opt) => {
            const delta = opt.e.deltaY;
            const activeObject = this.canvas.getActiveObject();
            
            if (activeObject) {
                let scale = activeObject.scaleX;

                if (delta < 0) {
                    scale *= 1.1; // 放大
                } else {
                    scale *= 0.9; // 缩小
                }

                // 限制缩放范围
                scale = Math.min(Math.max(scale, 0.01), 10);

                activeObject.scale(scale);
                this.canvas.renderAll();

                // 更新缩放值显示
                const relativeScale = getRelativeScale(activeObject);
                this.scaleText.innerHTML = `scale: ${relativeScale}×`;
                this.scaleText.style.display = 'block';
                
                opt.e.preventDefault();
                opt.e.stopPropagation();
            }
        });
    }


    async updateCanvas(canvasData, transformData = null) {
        if (!canvasData) return;

        try {
            // 先获取当前所有图层的状态（只在没有传入transformData时保留状态）
            const currentLayers = !transformData ? this.canvas.getObjects().map(obj => ({
                object: obj,
                state: {
                    left: obj.left,
                    top: obj.top,
                    scaleX: obj.scaleX,
                    scaleY: obj.scaleY,
                    angle: obj.angle,
                    flipX: obj.flipX,
                    flipY: obj.flipY
                }
            })) : [];
    
            // 先设置背景
            if (canvasData.background) {
                await this.setBackground(canvasData.background);
            }
    
            // 再更新图层
            if (canvasData.layers) {
                await this.updateLayers(canvasData.layers);
                
                // 如果有transformData，应用变换数据
                if (transformData) {
                    this.applyTransformData(transformData);
                } else {
                    // 否则恢复图层状态
                    currentLayers.forEach(({object, state}) => {
                        const existingObj = this.canvas.getObjects().find(obj => obj.id === object.id);
                        if (existingObj) {
                            existingObj.set(state);
                        }
                    });
                }
            }
    
            // 确保画布完全渲染
            await new Promise(resolve => {
                requestAnimationFrame(() => {
                    this.canvas.renderAll();
                    resolve();
                });
            });
            await this.sendCanvasState();
        } catch (error) {
            console.error('[FastCanvas] updateCanvas error:', error);
        }
    }

    // 应用变换数据到图层
    applyTransformData(transformData) {
        if (!transformData) return;
        
        try {
            // 遍历transform_data中的每个图层
            for (const [layerId, trans] of Object.entries(transformData)) {
                // 跳过background
                if (layerId === 'background') continue;
                
                // 查找对应的图层对象
                const layerObj = this.layers.get(parseInt(layerId));
                if (!layerObj) {
                    console.warn(`[FastCanvas] Layer ${layerId} not found`);
                    continue;
                }
                
                // 应用变换参数
                const updates = {
                    originX: 'center',
                    originY: 'center'
                };
                
                // 设置中心点位置
                if (trans.centerX !== undefined) updates.left = trans.centerX;
                if (trans.centerY !== undefined) updates.top = trans.centerY;
                
                // 缩放
                if (trans.scaleX !== undefined) updates.scaleX = trans.scaleX;
                if (trans.scaleY !== undefined) updates.scaleY = trans.scaleY;
                
                // 旋转（注意：fabric.js使用正角度，transform_data可能需要调整）
                if (trans.angle !== undefined) updates.angle = trans.angle;
                
                // 翻转
                if (trans.flipX !== undefined) updates.flipX = trans.flipX;
                if (trans.flipY !== undefined) updates.flipY = trans.flipY;
                
                // 应用所有变换
                layerObj.set(updates);
                layerObj.setCoords(); // 更新对象坐标
            }
            
            // 渲染画布
            this.canvas.renderAll();
            
            console.log('[FastCanvas] Applied transform data to layers');
        } catch (error) {
            console.error('[FastCanvas] Error applying transform data:', error);
        }
    }

    // 添加一个通用的图像加载函数
    loadImage(imageUrl, timeout = 5000) {
        return new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                reject(new Error('Image loading timeout'));
            }, timeout);

            fabric.Image.fromURL(imageUrl, 
                (img) => {
                    clearTimeout(timeoutId);
                    if (!img) {

                        reject(new Error('Failed to create image'));
                        return;
                    }
                    resolve(img);
                },
                (error) => {
                    clearTimeout(timeoutId);
                    reject(error);
                },
                { crossOrigin: 'anonymous' }  // 添加跨域支持
            );
        });
    }

    setLayerAsBackground() {
        const activeObject = this.canvas.getActiveObject();
        if (!activeObject) return;
    
        // 移除当前图层
        this.canvas.remove(activeObject);
        // 移除图层ID
        this.controlPanel.removeLayerId(activeObject);
        // 准备背景数据
        const backgroundData = {
            image: activeObject.getSrc()
        };
    
        // 使用现有的 setBackground 方法设置背景
        this.setBackground(backgroundData);
    }

    async setBackground(backgroundData) {
        if (!backgroundData?.image) {
            return;
        }
    
        try {
            const img = await this.loadImage(backgroundData.image);
            
            // 保存当前所有图层的状态
            const layerStates = this.canvas.getObjects().map(obj => ({
                object: obj,
                state: {
                    left: obj.left,
                    top: obj.top,
                    scaleX: obj.scaleX,
                    scaleY: obj.scaleY,
                    angle: obj.angle
                }
            }));
            
            // 保存原始尺寸
            this.originalSize = {
                width: img.width,
                height: img.height
            };
            
            // 计算显示尺寸
            const scaledSize = this.calculateScaledSize(
                img.width, 
                img.height, 
                this.maxDisplaySize
            );
    
            // 设置画布尺寸为原始尺寸
            this.canvas.setDimensions({
                width: this.originalSize.width,
                height: this.originalSize.height
            });
    
            // 设置背景图片属性
            img.set({
                scaleX: 1,
                scaleY: 1,
                left: 0,
                top: 0,
                selectable: false,
                evented: false,
                originX: 'left',
                originY: 'top'
            });
    
            // 设置CSS缩放
            const scale = scaledSize.width / this.originalSize.width;
            this.canvas.wrapperEl.style.transform = `scale(${scale})`;
            this.canvas.wrapperEl.style.transformOrigin = 'top left';
            
            // 更新容器尺寸
            this.updateContainerSize(scaledSize.width, scaledSize.height);
    
            // 更新尺寸输入框
            if (this.node?.sizeOverlay?.updateSizeInputs) {
                this.node.sizeOverlay.updateSizeInputs(
                    this.originalSize.width,
                    this.originalSize.height
                );
            }
    
            // 直接设置背景图片
            return new Promise((resolve) => {
                this.canvas.setBackgroundImage(img, () => {
                    // 恢复所有图层的状态
                    layerStates.forEach(({object, state}) => {
                        object.set(state);
                    });

                    this.canvas.renderAll();
                    resolve();
                });
            });
            
        } catch (error) {

            this.sendErrorState("Background loading failed");
        }
    }


    addLayer(layerData) {
        return new Promise((resolve) => {
            fabric.Image.fromURL(layerData.image, (img) => {
                if (!img) {

                    resolve();
                    return;
                }
    
                // 计算适合画布的缩放比例
                const scale = Math.min(
                    this.canvas.width / img.width,
                    this.canvas.height / img.height
                );
    
                // 计算居中位置
                const centerX = (this.canvas.width - img.width * scale) / 2;
                const centerY = (this.canvas.height - img.height * scale) / 2;
    
                img.set({
                    id: layerData.id,
                    scaleX: scale,
                    scaleY: scale,
                    left: centerX,
                    top: centerY,
                    selectable: true,
                    hasControls: true,
                    hasBorders: true,
                    originalWidth: img.width,
                    originalHeight: img.height
                });
    
                this.canvas.add(img);
                this.layers.set(layerData.id, img);
                
                // 添加以下代码来保持选中状态
                this.canvas.setActiveObject(img);
                this.canvas.renderAll();
                
                resolve();
            });
        });
    }

    updateLayer(layerData) {
        return new Promise((resolve) => {
            const existingObj = this.layers.get(layerData.id);
            if (existingObj) {
                // 保存当前的变换状态（补充透明度与混合模式）
                const currentState = {
                    scaleX: existingObj.scaleX,
                    scaleY: existingObj.scaleY,
                    left: existingObj.left,
                    top: existingObj.top,
                    angle: existingObj.angle,
                    flipX: existingObj.flipX,
                    flipY: existingObj.flipY,
                    opacity: existingObj.opacity,
                    globalCompositeOperation: existingObj.globalCompositeOperation
                };
    
                // 检查当前对象是否被选中
                const wasActive = this.canvas.getActiveObject() === existingObj;
    
                fabric.Image.fromURL(layerData.image, (newImg) => {
                    if (!newImg) {
                        resolve();
                        return;
                    }
    
                    // 直接应用所有保存的变换状态
                    newImg.set({
                        ...currentState,
                        id: layerData.id,
                        selectable: true,
                        hasControls: true,
                        hasBorders: true,
                        originalWidth: newImg.width,
                        originalHeight: newImg.height
                    });
    
                    // 替换旧图层
                    const index = this.canvas.getObjects().indexOf(existingObj);
                    this.canvas.remove(existingObj);
                    this.canvas.insertAt(newImg, index);
                    this.layers.set(layerData.id, newImg);
    
                    // 如果原对象是选中状态，则保持新对象的选中状态
                    if (wasActive) {
                        this.canvas.setActiveObject(newImg);
                    }
                    
                    this.canvas.renderAll();
                    resolve();
                });
            } else {
                // 如果图层不存在，作为新图层添加
                this.addLayer(layerData).then(resolve);
            }
        });
    }

    async updateLayers(layersData) {
        try {
            // 异步移除不再需要的图层
            const currentIds = new Set(layersData.map(layer => layer.id));
            const removePromises = [];
            
            for (const [id, fabricObj] of this.layers.entries()) {
                if (!currentIds.has(id)) {
                    removePromises.push(
                        new Promise(resolve => {
                            this.canvas.remove(fabricObj);
                            this.layers.delete(id);
                            requestAnimationFrame(resolve);
                        })
                    );
                }
            }

            if (removePromises.length > 0) {

                await Promise.all(removePromises);
                this.canvas.renderAll();
            }

            for (const layerData of layersData) {
                if (this.layers.has(layerData.id)) {
                    await this.updateLayer(layerData);
                } else {
                    await this.addLayer(layerData);
                }
                // 更新最大图层ID
                this.controlPanel.maxLayerId = Math.max(
                    this.controlPanel.maxLayerId,
                    layerData.id
                );
            }
            
            this.controlPanel.updateLayerSelector();
            
            // 确保最终渲染
            return new Promise(resolve => {
                requestAnimationFrame(() => {
                    this.canvas.renderAll();
                    resolve();
                });
            });

        } catch (error) {
            throw error;
        }
    }

    async getLayerMask(selectedLayer, ignoreOcclusion = false) {
        if (!selectedLayer || !this.canvas.backgroundImage) {
            return null;
        }
    
        // 创建与原始画布相同尺寸的遮罩画布
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = this.originalSize.width;
        maskCanvas.height = this.originalSize.height;
        const ctx = maskCanvas.getContext('2d');
    
        // 清除画布，保持透明背景
        ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    
        // 渲染选中图层
        if (selectedLayer.visible !== false) {
            ctx.save();
            
            // 直接使用图层的变换矩阵
            const matrix = selectedLayer.calcTransformMatrix();
            ctx.transform(
                matrix[0],
                matrix[1],
                matrix[2],
                matrix[3],
                matrix[4],
                matrix[5]
            );
    
            // 绘制图层 - 图像对象
            if (selectedLayer._element) {
                ctx.drawImage(
                    selectedLayer._element,
                    -selectedLayer.width / 2,
                    -selectedLayer.height / 2,
                    selectedLayer.width,
                    selectedLayer.height
                );
            }
            else if (selectedLayer.type === 'text' || selectedLayer.type === 'i-text') {
                // 设置文本样式
                ctx.fillStyle = 'white';
                ctx.font = `${selectedLayer.fontSize}px ${selectedLayer.fontFamily}`;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                
                // 处理行间距和多行文本
                const lines = selectedLayer.text.split('\n');
                const lineHeight = selectedLayer.lineHeight || 1.16; // Fabric默认值
                
                for (let i = 0; i < lines.length; i++) {
                    // 计算每行Y位置，考虑行高
                    const y = -selectedLayer.height / 2 + i * lineHeight * selectedLayer.fontSize + lineHeight*1.8;
                    
                    // 绘制每行文本，保留字间距处理
                    ctx.fillText(lines[i], -selectedLayer.width / 2, y);
                }
            }
            ctx.restore();
        }
    
        // 上层遮挡处理部分也需要修改
        if (!ignoreOcclusion) {
            const allLayers = this.canvas.getObjects()
                .filter(obj => obj !== this.canvas.backgroundImage);
            const selectedLayerIndex = allLayers.indexOf(selectedLayer);
            const upperLayers = allLayers.slice(selectedLayerIndex + 1);
    
            if (upperLayers.length > 0) {
                ctx.globalCompositeOperation = 'destination-out';
                
                upperLayers.forEach(layer => {
                    if (layer.visible !== false) {
                        ctx.save();
                        
                        const matrix = layer.calcTransformMatrix();
                        ctx.transform(
                            matrix[0], matrix[1], matrix[2],
                            matrix[3], matrix[4], matrix[5]
                        );
    
                        if (layer._element) {
                            ctx.drawImage(
                                layer._element,
                                -layer.width / 2,
                                -layer.height / 2,
                                layer.width,
                                layer.height
                            );
                        }
                        else if (layer.type === 'text' || layer.type === 'i-text') {
                            // 设置文本样式
                            ctx.fillStyle = 'white';
                            ctx.font = `${layer.fontSize}px ${layer.fontFamily}`;
                            ctx.textAlign = 'left';
                            ctx.textBaseline = 'top';
                            
                            // 处理行间距和多行文本
                            const lines = layer.text.split('\n');
                            const lineHeight = layer.lineHeight || 1.16; // Fabric默认值
                            
                            for (let i = 0; i < lines.length; i++) {
                                // 计算每行Y位置，考虑行高
                                const y = -selectedLayer.height / 2 + i * lineHeight * selectedLayer.fontSize + lineHeight*1.8;
                                
                                // 绘制每行文本，保留字间距处理
                                ctx.fillText(lines[i], -layer.width / 2, y);
                            }
                        }
                        ctx.restore();
                    }
                });
            }
        }
    
        return maskCanvas;
    }

    async sendCanvasState() {
        if (!this.canvas || !this.canvas.contextContainer) {
            console.log('[FastCanvas] 画布或上下文不存在，跳过状态发送');
            return;
        }
    

        try {
            const timestamp = Date.now();

            // 创建临时画布来处理尺寸
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = this.originalSize.width;
            tempCanvas.height = this.originalSize.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            const activeObjects = this.canvas.getActiveObjects();
            const activeObject = this.canvas.getActiveObject();
    
            // 主图像处理
            this.canvas.discardActiveObject();
            this.canvas.renderAll();
            
            // 将高分辨率画布内容绘制到正确尺寸的临时画布
            tempCtx.drawImage(
                this.canvas.lowerCanvasEl,
                0, 0, this.canvas.lowerCanvasEl.width, this.canvas.lowerCanvasEl.height,  // 源尺寸（高分辨率）
                0, 0, this.originalSize.width, this.originalSize.height  // 目标尺寸
            );
    
            if (activeObject) {
                this.canvas.setActiveObject(activeObject);
                this.canvas.renderAll();
            }
    
            // 从临时画布获取正确尺寸的Blob
            const imageBlob = await new Promise(resolve => {
                tempCanvas.toBlob(async (blob) => {
                    const tempImg = await createImageBitmap(blob);

                    resolve(blob);
                }, 'image/png', 1.0);
            });
    
            // 将Blob转换为ArrayBuffer
            const imageBuffer = await imageBlob.arrayBuffer();
            
            // 遮罩处理
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = this.originalSize.width;
            maskCanvas.height = this.originalSize.height;
            const ctx = maskCanvas.getContext('2d');
        
            // 处理选中对象的遮罩
            if (activeObjects && activeObjects.length > 1) {
                // 设置混合模式为"lighter"，确保重叠的白色区域保持白色
                ctx.globalCompositeOperation = 'lighter';
                
                // 按照画布中的顺序获取选中的图层
                const allLayers = this.canvas.getObjects().filter(obj => obj !== this.canvas.backgroundImage);
                const orderedSelectedObjects = allLayers.filter(obj => activeObjects.includes(obj));
                
                // 按顺序处理每个选中的对象
                for (const obj of orderedSelectedObjects) {
                    // 获取带遮挡关系的遮罩（注意这里传入 false 表示不忽略遮挡）
                    const layerMask = await this.getLayerMask(obj, false);
                    if (layerMask) {
                        ctx.drawImage(layerMask, 0, 0);
                    }
                }
            } else if (activeObject) {
                const layerMask = await this.getLayerMask(activeObject, false);
                if (layerMask) {
                    ctx.drawImage(layerMask, 0, 0);
                }
            } else {
                ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            }
        
            // 主遮罩转换为二进制
            const maskBlob = await new Promise(resolve => {
                maskCanvas.toBlob(resolve, 'image/png', 1.0);
            });
            const maskBuffer = await maskBlob.arrayBuffer();
            
            // 收集变换信息（使用之前获取的activeObjects）
            const layerTransforms = {};
            if (activeObjects && activeObjects.length > 0) {
                activeObjects.forEach(obj => {
                    if (obj.id) {
                        // 获取对象的中心点坐标
                        const centerPoint = obj.getCenterPoint();
                        
                        layerTransforms[obj.id] = {
                            centerX: centerPoint.x,  // 使用中心点 X 坐标
                            centerY: centerPoint.y,  // 使用中心点 Y 坐标
                            scaleX: obj.scaleX,
                            scaleY: obj.scaleY,
                            angle: obj.angle,
                            width: obj.width,
                            height: obj.height,
                            flipX: obj.flipX,
                            flipY: obj.flipY
                        };
                    }
                });
            }

            // 构建发送的数据对象
            const data = {
                node_id: this.node.id.toString(),
                timestamp: timestamp.toString(),
                type: 'temp',
                subfolder: 'fast_canvas',
                overwrite: 'true',
                main_image: Array.from(new Uint8Array(imageBuffer)),
                main_mask: Array.from(new Uint8Array(maskBuffer)),
                layer_transforms: layerTransforms  // 替换原来的 layer_masks
            };

    
            // 发送数据
            const response = await api.fetchApi('/fast_canvas_all', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
    
            console.log('[FastCanvas] 数据发送成功，服务器响应:', response);
            
            await new Promise(resolve => setTimeout(resolve, 200));
    
        } catch (error) {
            console.error('[FastCanvas] 更新画布状态失败:', error);
            throw error;
        }
    }

    // 使用fabric.js内置方法序列化画布（不包含图像数据）
    serializeCanvasJSON() {
        if (!this.canvas) {
            return null;
        }
        
        try {
            // 保存当前选中状态
            const activeObjectIds = [];
            const activeObjects = this.canvas.getActiveObjects();
            if (activeObjects) {
                activeObjects.forEach(obj => {
                    if (obj.id) activeObjectIds.push(obj.id);
                });
            }
            
            // 序列化画布，包含自定义属性（混合模式/透明度/文字相关）
            const json = this.canvas.toJSON([
                'id', 'originalWidth', 'originalHeight',
                'globalCompositeOperation', 'opacity',
                'text', 'fontFamily', 'fontSize', 'fill', 'backgroundColor',
                'fontWeight', 'fontStyle', 'underline',
                'charSpacing', 'lineHeight', 'stroke', 'strokeWidth', 'textAlign'
            ]);
            
            // 添加元数据
            json.metadata = {
                version: '1.0',
                timestamp: Date.now(),
                canvasSize: this.originalSize,
                activeObjectIds: activeObjectIds
            };
            
            // 移除图像数据，仅保留引用
            if (json.objects) {
                json.objects.forEach(obj => {
                    if (obj.type === 'image') {
                        obj.imageRef = `layer_${obj.id}`;
                        delete obj.src;
                    }
                });
            }
            
            if (json.backgroundImage) {
                json.backgroundImage.imageRef = 'background';
                delete json.backgroundImage.src;
            }
            
            return JSON.stringify(json);
        } catch (error) {
            return null;
        }
    }
    // 从JSON恢复画布状态（需要提供图像数据）
    async deserializeCanvasJSON(jsonString, imageDataProvider) {
        if (!this.canvas || !jsonString) {
            return false;
        }
        
        try {
            // 解析JSON
            const json = JSON.parse(jsonString);
            
            // 处理对象中的图像引用
            if (json.objects) {
                for (let i = 0; i < json.objects.length; i++) {
                    const obj = json.objects[i];
                    if (obj.type === 'image' && obj.imageRef && imageDataProvider) {
                        // 获取图像数据
                        const imageData = await imageDataProvider(obj.imageRef);
                        if (imageData) {
                            obj.src = imageData;
                        }
                    }
                }
            }
            
            // 处理背景图像引用
            if (json.backgroundImage && json.backgroundImage.imageRef && imageDataProvider) {
                const bgImageData = await imageDataProvider(json.backgroundImage.imageRef);
                if (bgImageData) {
                    json.backgroundImage.src = bgImageData;
                }
            }
            
            // 使用fabric.js的loadFromJSON方法恢复画布
            return new Promise(resolve => {
                this.canvas.loadFromJSON(json, () => {
                    // 恢复选中状态
                    if (json.metadata && json.metadata.activeObjectIds && json.metadata.activeObjectIds.length > 0) {
                        const objectsToSelect = [];
                        json.metadata.activeObjectIds.forEach(id => {
                            const obj = this.canvas.getObjects().find(o => o.id === id);
                            if (obj) objectsToSelect.push(obj);
                        });
                        
                        if (objectsToSelect.length === 1) {
                            this.canvas.setActiveObject(objectsToSelect[0]);
                        } else if (objectsToSelect.length > 1) {
                            const selection = new fabric.ActiveSelection(objectsToSelect, {
                                canvas: this.canvas
                            });
                            this.canvas.setActiveObject(selection);
                        }
                    }
                    
                    // 设置画布尺寸
                    if (json.metadata && json.metadata.canvasSize) {
                        this.originalSize = json.metadata.canvasSize;
                    }
                    
                    // 更新图层引用
                    this.layers = new Map();
                    this.canvas.getObjects().forEach(obj => {
                        if (obj.id) {
                            this.layers.set(obj.id, obj);
                        }
                    });
                    
                    this.canvas.renderAll();

                    resolve(true);
                });
            });
        } catch (error) {
            return false;
        }
    }
}



// 控制面板类
class ControlPanel {
    constructor(canvas, node) {
        this.canvas = canvas;  // fabric.js canvas 实例
        this.node = node;     // ComfyUI 节点实例
        this.container = this.createContainer();
        this.currentLayerId = null; // 当前选中的图层ID
        this.maxLayerId = 0; // 最大图层ID
        this.textPanelVisible = false; // 文字面板可见性状态
        this.initializeControls();
    }

    // 创建控制面板容器
    createContainer() {
        const container = document.createElement('div');
        container.className = 'fast-canvas-control-panel';
        container.style.cssText = `
            position: absolute;
            bottom: -5px;  // 将bottom设为负值，使控制面板向下移动
            left: 0;
            width: 100%;
            height: ${CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px;
            background-color: #353535;
            padding: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            box-sizing: border-box;
        `;
        return container;
    }

    // 创建按钮
    createButton(text, onClick) {
        const button = document.createElement('button');
        button.textContent = text;
        button.style.cssText = `
            padding: 5px 10px;
            height: 28px;           
            line-height: 18px;      
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            font-size: 14px;
        `;
        button.onclick = onClick;
        return button;
    }

    // 创建数字输入框
    createNumberInput(placeholder, defaultValue, onInput) {
        const input = document.createElement('input');
        input.type = 'number';
        input.placeholder = placeholder;
        
        input.style.cssText = `
            padding: 5px 10px;
            height: 28px;
            line-height: 18px;
            background: #333;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            width: 80px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            font-size: 14px;
            text-align: center;
        `;

        const style = document.createElement('style');
        style.textContent = `
            input[type="number"]::-webkit-inner-spin-button,
            input[type="number"]::-webkit-outer-spin-button {
                -webkit-appearance: none;
                margin: 0;
            }
            input[type="number"] {
                -moz-appearance: textfield;
            }
            input[type="number"]::placeholder {
                color: #888;
                opacity: 1;
                text-align: center;
            }
        `;
        document.head.appendChild(style);

        input.dataset.defaultValue = defaultValue;
        input.oninput = (e) => {
            const value = parseInt(e.target.value);
            if (value > 0) {
                onInput(e);
            } else {
                e.target.value = input.dataset.defaultValue;
                onInput({ target: { value: input.dataset.defaultValue } });
            }
        };
        return input;
    }
    // 创建图层选择器
    createLayerSelector() {
        const container = document.createElement('div');
        container.className = 'layer-selector';
        container.style.cssText = `
            position: relative;
            display: inline-block;
        `;

        const button = document.createElement('button');
        button.className = 'layer-selector-btn';
        button.style.cssText = `
            padding: 5px 8px;
            height: 28px;
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            min-width: 80px;
            justify-content: space-between;
            font-size: 14px;
        `;

        const buttonText = document.createElement('span');
        buttonText.textContent = '图层';
        button.appendChild(buttonText);

        const arrow = document.createElement('span');
        arrow.textContent = '▼';
        arrow.style.marginLeft = '5px';
        button.appendChild(arrow);

        const dropdown = document.createElement('div');
        dropdown.className = 'layer-dropdown';
        dropdown.style.cssText = `
            position: absolute;
            top: 100%;
            left: 0;
            background: #2a2a2a;
            border: 1px solid #666;
            border-radius: 4px;
            display: none;
            z-index: 1000;
            min-width: 100%;
            max-height: 200px;
            overflow-y: auto;
        `;

        // 点击按钮显示/隐藏下拉菜单
        button.onclick = (e) => {
            e.stopPropagation();
            dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
        };

        // 点击其他地方关闭下拉菜单
        document.addEventListener('click', () => {
            dropdown.style.display = 'none';
        });

        container.appendChild(button);
        container.appendChild(dropdown);
        
        // 保存引用以便更新
        this.layerButton = button;
        this.layerButtonText = buttonText;
        this.layerDropdown = dropdown;
        
        return container;
    }

    // 更新图层选择器
    updateLayerSelector() {
        const dropdown = this.layerDropdown;
        dropdown.innerHTML = '';
        
        // 获取所有图层，保持原有顺序
        const layers = Array.from(this.canvas.getObjects())
            .filter(obj => obj !== this.canvas.backgroundImage && obj.id);

        // 获取当前选中的图层
        const activeObject = this.canvas.getActiveObject();
        if (activeObject && activeObject !== this.canvas.backgroundImage) {
            this.currentLayerId = activeObject.id;
        }

        // 更新按钮文本
        this.layerButtonText.textContent = this.currentLayerId ? 
            `图层 ${this.currentLayerId}` : '图层';

        // 创建下拉选项
        layers.forEach(layer => {
            if (layer.id !== this.currentLayerId) {
                const option = document.createElement('div');
                option.className = 'layer-option';
                option.textContent = `图层 ${layer.id}`;
                option.style.cssText = `
                    padding: 5px 10px;
                    cursor: pointer;
                    white-space: nowrap;
                    &:hover {
                        background: #444;
                    }
                `;
                
                option.onclick = (e) => {
                    e.stopPropagation();
                    
                    // 只选中图层，不改变叠加顺序
                    this.canvas.setActiveObject(layer);
                    this.currentLayerId = layer.id;
                    this.updateLayerSelector();
                    
                    // 触发画布更新，但不改变图层顺序
                    this.canvas.requestRenderAll();
                    
                    this.layerDropdown.style.display = 'none';
                };
                
                dropdown.appendChild(option);
            }
        });
    }

    // 创建设置弹窗
    createSettingsDialog() {
        const dialog = document.createElement('dialog');
        dialog.style.cssText = `
            background: #353535;
            border: 1px solid #666;
            border-radius: 4px;
            color: #fff;
            padding: 20px;
            min-width: 300px;
        `;

        const title = document.createElement('h3');
        title.textContent = '设置';
        title.style.cssText = `
            margin: 0 0 20px 0;
            font-size: 16px;
        `;

        // 创建设置项容器
        const settingsContainer = document.createElement('div');
        settingsContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 15px;
        `;

        // 窗口最大尺寸设置项
        const maxSizeInput = this.createNumberInput('最大尺寸', 1024, (e) => {
            const maxSize = parseInt(e.target.value);
            if (maxSize > 0) {
                this.node.canvasInstance.updateDisplayScale(maxSize);
            }
        });
        const maxSizeItem = this.createSettingItem(
            '窗口最大尺寸:', 
            '限制画布显示尺寸',
            maxSizeInput
        );

        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = `
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 20px;
        `;

        const saveButton = this.createButton('保存', () => {
            dialog.close();
        });

        const cancelButton = this.createButton('取消', () => {
            dialog.close();
        });

        settingsContainer.appendChild(maxSizeItem);
        buttonContainer.appendChild(cancelButton);
        buttonContainer.appendChild(saveButton);

        dialog.appendChild(title);
        dialog.appendChild(settingsContainer);
        dialog.appendChild(buttonContainer);

        return dialog;
    }

    // 创建单个设置项
    createSettingItem(labelText, tooltipText, inputElement) {
        const container = document.createElement('div');
        container.style.cssText = `
            display: flex;
            align-items: center;
            gap: 10px;
            position: relative;
        `;

        const label = document.createElement('label');
        label.style.cssText = `
            flex: 1;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
        `;
        label.textContent = labelText;

        // 添加问号图标
        const helpIcon = document.createElement('span');
        helpIcon.textContent = '?';
        helpIcon.style.cssText = `
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #666;
            font-size: 12px;
            cursor: help;
        `;

        // 创建tooltip
        const tooltip = document.createElement('div');
        tooltip.textContent = tooltipText;
        tooltip.style.cssText = `
            position: absolute;
            background: #1a1a1a;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 12px;
            color: #fff;
            width: max-content;
            max-width: 200px;
            left: 0;
            top: -45px;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
            z-index: 1000;
            line-height: 1.4;
            pointer-events: none;
        `;

        // 添加悬停事件
        helpIcon.addEventListener('mouseenter', () => {
            tooltip.style.opacity = '1';
            tooltip.style.visibility = 'visible';
        });

        helpIcon.addEventListener('mouseleave', () => {
            tooltip.style.opacity = '0';
            tooltip.style.visibility = 'hidden';
        });

        label.appendChild(helpIcon);
        container.appendChild(label);
        container.appendChild(inputElement);
        container.appendChild(tooltip);

        return container;
    }

    // 创建文字功能面板
    createTextPanel() {
        const panel = document.createElement('div');
        panel.className = 'text-function-panel';
        panel.style.cssText = `
            position: absolute;
            top: calc(100% + 5px);
            left: 0;
            width: 100%;
            background-color: #353535;
            padding: 10px;
            display: none;
            flex-wrap: wrap;
            gap: 10px;
            border-top: 1px solid #666;
            border-radius: 8px;
            z-index: 100;
            box-sizing: border-box;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        `;
        
        // 添加文字按钮
        const addTextBtn = this.createButton('添加文字', () => {
            const text = new fabric.IText('双击编辑文字', {
                fontSize: 48,
                fill: '#ffffff',
                fontFamily: 'Arial',
                originX: 'center',
                originY: 'center'
            });
            
            // 为新添加的文本对象分配ID
            this.maxLayerId++;
            text.id = this.maxLayerId;
            
            // 获取画布中心点
            const canvasCenter = this.canvas.getCenter();
            text.set({
                left: canvasCenter.left,
                top: canvasCenter.top
            });
            
            this.canvas.add(text);
            this.canvas.setActiveObject(text);
            this.currentLayerId = text.id;
            this.updateLayerSelector();
            this.canvas.renderAll();
        });
        
        // 字体选择器
        const fontSelector = document.createElement('select');
        fontSelector.style.cssText = `
            padding: 5px 10px;
            height: 28px;
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            min-width: 100px;
        `;
        
        const fonts = ['Arial', 'Times New Roman', 'Courier New', '宋体', '黑体', '微软雅黑'];
        fonts.forEach(font => {
            const option = document.createElement('option');
            option.value = font;
            option.textContent = font;
            fontSelector.appendChild(option);
        });
        
        fontSelector.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('fontFamily', fontSelector.value);
                this.canvas.renderAll();
            }
        };
        
        // 字号选择器
        const sizeSelector = document.createElement('select');
        sizeSelector.style.cssText = `
            padding: 5px 10px;
            height: 28px;           
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            min-width: 60px;
        `;
        
        const sizes = [12, 14, 16, 18, 20, 24, 28, 32, 36, 42, 48, 56, 64, 72];
        sizes.forEach(size => {
            const option = document.createElement('option');
            option.value = size;
            option.textContent = size;
            sizeSelector.appendChild(option);
        });
        
        sizeSelector.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('fontSize', parseInt(sizeSelector.value));
                this.canvas.renderAll();
            }
        };
        
        // 文字颜色选择器
        const colorPicker = document.createElement('input');
        colorPicker.type = 'color';
        colorPicker.value = '#ffffff';
        colorPicker.style.cssText = `
            width: 28px;
            height: 28px;
            padding: 0;
            border: 1px solid #666;
            border-radius: 4px;
            background: #444;
            cursor: pointer;
        `;
        
        colorPicker.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('fill', colorPicker.value);
                this.canvas.renderAll();
            }
        };
        
        // 背景色选择器
        const bgColorPicker = document.createElement('input');
        bgColorPicker.type = 'color';
        bgColorPicker.value = '#000000';
        bgColorPicker.style.cssText = `
            width: 28px;
            height: 28px;
            padding: 0;
            border: 1px solid #666;
            border-radius: 4px;
            background: #444;
            cursor: pointer;
        `;
        
        const bgColorLabel = document.createElement('span');
        bgColorLabel.textContent = '背景';
        bgColorLabel.style.cssText = `
            color: #fff;
            font-size: 12px;
            margin-right: 4px;
        `;
        
        const bgColorContainer = document.createElement('div');
        bgColorContainer.style.cssText = `
            display: flex;
            align-items: center;
        `;
        bgColorContainer.appendChild(bgColorLabel);
        bgColorContainer.appendChild(bgColorPicker);
        
        bgColorPicker.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('backgroundColor', bgColorPicker.value);
                this.canvas.renderAll();
            }
        };
        
        // 文字样式按钮组
        const styleContainer = document.createElement('div');
        styleContainer.style.cssText = `
            display: flex;
            gap: 5px;
        `;
        
        // 加粗按钮
        const boldBtn = this.createButton('B', () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('fontWeight', activeObj.fontWeight === 'bold' ? 'normal' : 'bold');
                this.canvas.renderAll();
                boldBtn.style.background = activeObj.fontWeight === 'bold' ? '#666' : '#444';
            }
        });
        boldBtn.style.fontWeight = 'bold';
        boldBtn.style.padding = '5px 8px';
        
        // 斜体按钮
        const italicBtn = this.createButton('I', () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('fontStyle', activeObj.fontStyle === 'italic' ? 'normal' : 'italic');
                this.canvas.renderAll();
                italicBtn.style.background = activeObj.fontStyle === 'italic' ? '#666' : '#444';
            }
        });
        italicBtn.style.fontStyle = 'italic';
        italicBtn.style.padding = '5px 8px';
        
        // 下划线按钮
        const underlineBtn = this.createButton('U', () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('underline', !activeObj.underline);
                this.canvas.renderAll();
                underlineBtn.style.background = activeObj.underline ? '#666' : '#444';
            }
        });
        underlineBtn.style.textDecoration = 'underline';
        underlineBtn.style.padding = '5px 8px';
        
        styleContainer.appendChild(boldBtn);
        styleContainer.appendChild(italicBtn);
        styleContainer.appendChild(underlineBtn);
        
        // 字间距控制
        const charSpacingLabel = document.createElement('span');
        charSpacingLabel.textContent = '字间距';
        charSpacingLabel.style.cssText = `
            color: #fff;
            font-size: 12px;
            margin-right: 4px;
        `;
        
        const charSpacingInput = document.createElement('input');
        charSpacingInput.type = 'number';
        charSpacingInput.min = '0';
        charSpacingInput.max = '1000';
        charSpacingInput.value = '0';
        charSpacingInput.style.cssText = `
            width: 50px;
            padding: 3px;
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
        `;
        
        const charSpacingContainer = document.createElement('div');
        charSpacingContainer.style.cssText = `
            display: flex;
            align-items: center;
        `;
        charSpacingContainer.appendChild(charSpacingLabel);
        charSpacingContainer.appendChild(charSpacingInput);
        
        charSpacingInput.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('charSpacing', parseInt(charSpacingInput.value));
                this.canvas.renderAll();
            }
        };
        
        // 行间距控制
        const lineHeightLabel = document.createElement('span');
        lineHeightLabel.textContent = '行间距';
        lineHeightLabel.style.cssText = `
            color: #fff;
            font-size: 12px;
            margin-right: 4px;
        `;
        
        const lineHeightInput = document.createElement('input');
        lineHeightInput.type = 'number';
        lineHeightInput.min = '0';
        lineHeightInput.max = '10';
        lineHeightInput.step = '0.1';
        lineHeightInput.value = '1';
        lineHeightInput.style.cssText = `
            width: 50px;
            padding: 3px;
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
        `;
        
        const lineHeightContainer = document.createElement('div');
        lineHeightContainer.style.cssText = `
            display: flex;
            align-items: center;
        `;
        lineHeightContainer.appendChild(lineHeightLabel);
        lineHeightContainer.appendChild(lineHeightInput);
        
        lineHeightInput.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('lineHeight', parseFloat(lineHeightInput.value));
                this.canvas.renderAll();
            }
        };
        
        // 描边控制
        const strokeLabel = document.createElement('span');
        strokeLabel.textContent = '描边';
        strokeLabel.style.cssText = `
            color: #fff;
            font-size: 12px;
            margin-right: 4px;
        `;
        
        const strokeColorPicker = document.createElement('input');
        strokeColorPicker.type = 'color';
        strokeColorPicker.value = '#000000';
        strokeColorPicker.style.cssText = `
            width: 28px;
            height: 28px;
            padding: 0;
            border: 1px solid #666;
            border-radius: 4px;
            background: #444;
            cursor: pointer;
            margin-right: 4px;
        `;
        
        const strokeWidthInput = document.createElement('input');
        strokeWidthInput.type = 'number';
        strokeWidthInput.min = '0';
        strokeWidthInput.max = '10';
        strokeWidthInput.step = '0.5';
        strokeWidthInput.value = '0';
        strokeWidthInput.style.cssText = `
            width: 40px;
            padding: 3px;
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
        `;
        
        const strokeContainer = document.createElement('div');
        strokeContainer.style.cssText = `
            display: flex;
            align-items: center;
        `;
        strokeContainer.appendChild(strokeLabel);
        strokeContainer.appendChild(strokeColorPicker);
        strokeContainer.appendChild(strokeWidthInput);
        
        strokeColorPicker.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('stroke', strokeColorPicker.value);
                this.canvas.renderAll();
            }
        };
        
        strokeWidthInput.oninput = () => {
            const activeObj = this.canvas.getActiveObject();
            if (activeObj && activeObj.type === 'i-text') {
                activeObj.set('strokeWidth', parseFloat(strokeWidthInput.value));
                this.canvas.renderAll();
            }
        };


        
        // 添加所有控件到面板
        panel.appendChild(addTextBtn);
        panel.appendChild(fontSelector);
        panel.appendChild(sizeSelector);
        panel.appendChild(colorPicker);
        panel.appendChild(bgColorContainer);
        panel.appendChild(styleContainer);
        panel.appendChild(charSpacingContainer);
        panel.appendChild(lineHeightContainer);
        panel.appendChild(strokeContainer);


        
        // 监听文本对象选择事件，更新控件状态
        this.canvas.on('selection:created', (e) => {
            const obj = e.selected[0];
            if (obj && obj.type === 'i-text') {
                fontSelector.value = obj.fontFamily;
                sizeSelector.value = obj.fontSize;
                colorPicker.value = obj.fill;
                bgColorPicker.value = obj.backgroundColor || '#000000';
                boldBtn.style.background = obj.fontWeight === 'bold' ? '#666' : '#444';
                italicBtn.style.background = obj.fontStyle === 'italic' ? '#666' : '#444';
                underlineBtn.style.background = obj.underline ? '#666' : '#444';
                charSpacingInput.value = obj.charSpacing || 0;
                lineHeightInput.value = obj.lineHeight || 1.2;
                strokeWidthInput.value = obj.strokeWidth || 0;
                strokeColorPicker.value = obj.stroke || '#000000';
            }
        });
        
        this.canvas.on('selection:updated', (e) => {
            const obj = e.selected[0];
            if (obj && obj.type === 'i-text') {
                fontSelector.value = obj.fontFamily;
                sizeSelector.value = obj.fontSize;
                colorPicker.value = obj.fill;
                bgColorPicker.value = obj.backgroundColor || '#000000';
                boldBtn.style.background = obj.fontWeight === 'bold' ? '#666' : '#444';
                italicBtn.style.background = obj.fontStyle === 'italic' ? '#666' : '#444';
                underlineBtn.style.background = obj.underline ? '#666' : '#444';
                charSpacingInput.value = obj.charSpacing || 0;
                lineHeightInput.value = obj.lineHeight || 1.2;
                strokeWidthInput.value = obj.strokeWidth || 0;
                strokeColorPicker.value = obj.stroke || '#000000';
            }
        });
        
        return panel;
    }
    
    // 切换文字功能面板显示状态
    toggleTextPanel() {
        if (!this.textPanel) {
            this.textPanel = this.createTextPanel();
            this.node.canvasElement.appendChild(this.textPanel);
        }
        
        this.textPanelVisible = !this.textPanelVisible;
        this.textPanel.style.display = this.textPanelVisible ? 'flex' : 'none';
    }

    initializeControls() {
        const layerSelector = this.createLayerSelector();
        this.container.appendChild(layerSelector);

        const loadButton = this.createButton('加载', () => {
            if (this.node.id) {
                queueManager.queueOutputNodes([this.node.id]);
            }
        });
        
        // 添加文字功能按钮
        const textButton = this.createButton('文字', () => {
            this.toggleTextPanel();
        });

        const resetButton = this.createButton('重置', () => {
            if (this.node.id) {
                // 保存当前节点的状态
                const currentNode = this.node;
                const currentSize = [...currentNode.size];
                const currentPos = [...currentNode.pos];
                const currentOriginalSize = {...currentNode.canvasInstance.originalSize};
                const currentMaxDisplaySize = currentNode.canvasInstance.maxDisplaySize;
                
                // 获取当前节点的存储键和尺寸
                const STORAGE_KEY = 'canvas_size_';
                const savedSize = localStorage.getItem(STORAGE_KEY + currentNode.id);
                
                // 创建新节点
                let new_node = LiteGraph.createNode("FastCanvas");
                new_node.pos = currentPos;
                app.graph.add(new_node, false);
                
                requestAnimationFrame(() => {
                    // 复制节点信息（包括连接）
                    node_info_copy(currentNode, new_node, true);
                    
                    if (new_node.canvasInstance) {
                        // 设置画布实例的属性
                        new_node.canvasInstance.originalSize = currentOriginalSize;
                        new_node.canvasInstance.maxDisplaySize = currentMaxDisplaySize;
                        
                        // 保持原有尺寸
                        new_node.size = currentSize;
                        
                        // 如果有保存的尺寸，转移到新节点
                        if (savedSize) {
                            localStorage.setItem(STORAGE_KEY + new_node.id, savedSize);
                            // 删除旧节点的存储
                            localStorage.removeItem(STORAGE_KEY + currentNode.id);
                        }
                        
                        // 设置画布容器尺寸
                        if (new_node.canvasElement) {
                            const width = currentOriginalSize.width;
                            const height = currentOriginalSize.height;
                            
                            // 计算缩放后的尺寸
                            const scaledSize = new_node.canvasInstance.calculateScaledSize(
                                width,
                                height,
                                currentMaxDisplaySize
                            );
                            
                            // 更新画布容器尺寸
                            new_node.canvasElement.style.width = `${scaledSize.width}px`;
                            new_node.canvasElement.style.height = `${scaledSize.height + CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px`;
                            
                            // 更新画布尺寸
                            if (new_node.canvasInstance.canvas) {
                                new_node.canvasInstance.updateCanvasSize(
                                    scaledSize.width,
                                    scaledSize.height
                                );
                            }
                            
                            // 更新尺寸输入框显示
                            if (new_node.sizeOverlay?.updateSizeInputs) {
                                new_node.sizeOverlay.updateSizeInputs(width, height);
                            }
                        }
                    }
                    
                    // 更新画布
                    app.graph.setDirtyCanvas(true, true);
                    // 移除旧节点
                    app.graph.remove(currentNode);
                });
            }
        });

        const settingsButton = this.createButton('⚙️', () => {
            const dialog = this.createSettingsDialog();
            document.body.appendChild(dialog);
            dialog.showModal();
            dialog.addEventListener('close', () => {
                document.body.removeChild(dialog);
            });
        });
        settingsButton.style.padding = '5px 8px';

        // 添加控件到容器
        this.container.appendChild(loadButton);
        this.container.appendChild(textButton);  // 添加文字按钮
        this.container.appendChild(resetButton);
        this.container.appendChild(settingsButton);
        
        // 监听画布选择变化
        this.canvas.on('selection:created', (e) => {
            const obj = e.selected[0];
            if (obj && obj !== this.canvas.backgroundImage) {
                this.currentLayerId = obj.id;
                this.updateLayerSelector();
            }
        });

        this.canvas.on('selection:updated', (e) => {
            const obj = e.selected[0];
            if (obj && obj !== this.canvas.backgroundImage) {
                this.currentLayerId = obj.id;
                this.updateLayerSelector();
            }
        });

        this.canvas.on('selection:cleared', () => {
            this.currentLayerId = null;
            this.updateLayerSelector();
        });
    }
    // 获取控制面板容器
    getContainer() {
        return this.container;
    }

    // 更新控制面板尺寸
    updateSize(width) {
        this.container.style.width = `${width}px`;
    }
    // 更新图层ID
    updateLayerIds() {
        const objects = this.canvas.getObjects();
        let maxId = this.maxLayerId;

        objects.forEach(obj => {
            if (obj !== this.canvas.backgroundImage) {
                if (!obj.id) {
                    // 为新图层分配ID
                    maxId++;
                    obj.id = maxId;
                }
            }
        });

        this.maxLayerId = maxId;
        
        // 更新当前选中的图层ID
        const activeObject = this.canvas.getActiveObject();
        if (activeObject && activeObject !== this.canvas.backgroundImage) {
            this.currentLayerId = activeObject.id;
        }
        
        this.updateLayerSelector();
    }

    // 设置背景时移除图层ID
    removeLayerId(obj) {
        if (obj.id) {
            delete obj.id;
            if (this.currentLayerId === obj.id) {
                this.currentLayerId = null;
            }
            this.updateLayerSelector();
        }
    }
}

// 创建覆盖层扩展
class FastCanvasOverlay {
    static createOverlay(node) {

        // 创建覆盖层容器
        const overlayContainer = document.createElement("div");
        overlayContainer.style.cssText = `
            position: absolute;
            left: 0;
            top: -35px;
            width: 100%;
            height: ${CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px;
            background: #353535;
            display: flex;
            gap: 10px;
            padding: 10px;
            z-index: 10;
            align-items: center;
            box-sizing: border-box;
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
        `;

        // 创建输入框样式
        const inputStyle = `
            padding: 5px 10px;
            height: 28px;
            line-height: 18px;
            background: #333;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            width: 60px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            font-size: 14px;
            text-align: center;
        `;

        // 创建按钮样式
        const buttonStyle = `
            padding: 5px 10px;
            height: 28px;           
            line-height: 18px;      
            background: #444;
            border: 1px solid #666;
            color: #fff;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            font-size: 14px;
        `;
        
        const STORAGE_KEY = 'canvas_size_';

        // 从本地存储中获取保存的尺寸
        const savedSize = localStorage.getItem(STORAGE_KEY + node.id);
        const initialSize = savedSize ? JSON.parse(savedSize) : {
            width: node.canvasInstance.originalSize.width,
            height: node.canvasInstance.originalSize.height
        };
        
        // 创建宽度输入框时使用保存的值
        const widthInput = document.createElement("input");
        widthInput.type = "number";
        widthInput.id = "canvas-width";
        widthInput.placeholder = "宽度";
        widthInput.value = initialSize.width; // 使用保存的宽度
        widthInput.style.cssText = inputStyle;
        
        // 创建高度输入框时使用保存的值
        const heightInput = document.createElement("input");
        heightInput.type = "number";
        heightInput.id = "canvas-height";
        heightInput.placeholder = "高度";
        heightInput.value = initialSize.height; // 使用保存的高度
        heightInput.style.cssText = inputStyle;
        // 更新尺寸显示函数（显示实际尺寸）
        const updateSizeInputs = (width, height) => {
            widthInput.value = width;
            heightInput.value = height;
        };
        
        // 创建更新按钮
        const updateButton = document.createElement("button");
        updateButton.textContent = "更新尺寸";
        updateButton.style.cssText = buttonStyle;
        updateButton.onclick = async () => {
            if (!node.canvasInstance?.canvas) return;
            
            let customWidth = parseInt(widthInput.value, 10);
            let customHeight = parseInt(heightInput.value, 10);
            
            if (isNaN(customWidth) && isNaN(customHeight)) {
                console.warn('[FastCanvas] 无效的尺寸输入');
                return;
            }
            
            // 获取当前背景尺寸
            const currentBg = node.canvasInstance.canvas.backgroundImage;
            const currentWidth = currentBg ? currentBg.width : node.canvasInstance.originalSize.width;
            const currentHeight = currentBg ? currentBg.height : node.canvasInstance.originalSize.height;
            
            // 处理按边等比例缩放
            if (customWidth === 0 && customHeight > 0) {
                // 按高度等比例计算宽度
                const ratio = customHeight / currentHeight;
                customWidth = Math.round(currentWidth * ratio);
            } else if (customHeight === 0 && customWidth > 0) {
                // 按宽度等比例计算高度
                const ratio = customWidth / currentWidth;
                customHeight = Math.round(currentHeight * ratio);
            } else if (customWidth === 0 && customHeight === 0) {
                console.warn('[FastCanvas] 宽度和高度不能同时为0');
                return;
            }
            
            try {
                // 1. 确保尺寸能被8整除
                const adjustedWidth = Math.floor(customWidth / 8) * 8;
                const adjustedHeight = Math.floor(customHeight / 8) * 8;
                
                // 2. 更新原始尺寸
                node.canvasInstance.originalSize = {
                    width: adjustedWidth,
                    height: adjustedHeight
                };
            
                // 3. 计算显示尺寸（应用限制窗口大小的功能）
                const scaledSize = node.canvasInstance.calculateScaledSize(
                    adjustedWidth,
                    adjustedHeight,
                    node.canvasInstance.maxDisplaySize
                );
            
                // 4. 如果有现有背景，创建新的背景
                const currentBg = node.canvasInstance.canvas.backgroundImage;
                let newBackground;
            
                if (currentBg) {
                    // 创建临时画布
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = adjustedWidth;
                    tempCanvas.height = adjustedHeight;
                    const ctx = tempCanvas.getContext('2d');
            
                    // 获取当前背景的原始尺寸
                    const origWidth = currentBg.width;
                    const origHeight = currentBg.height;
            
                    // 计算填充比例
                    const scaleX = adjustedWidth / origWidth;
                    const scaleY = adjustedHeight / origHeight;
                    const scale = Math.max(scaleX, scaleY); // 使用较大的缩放比例以填充
            
                    // 计算居中位置
                    const scaledWidth = origWidth * scale;
                    const scaledHeight = origHeight * scale;
                    const left = (adjustedWidth - scaledWidth) / 2;
                    const top = (adjustedHeight - scaledHeight) / 2;
            
                    // 先填充黑色背景
                    ctx.fillStyle = '#000000';
                    ctx.fillRect(0, 0, adjustedWidth, adjustedHeight);
            
                    // 绘制并居中裁剪背景
                    if (currentBg instanceof fabric.Image) {
                        // 如果是 Image 对象，使用 getElement()
                        ctx.drawImage(
                            currentBg.getElement(),
                            left, top, scaledWidth, scaledHeight
                        );
                    } else if (currentBg instanceof fabric.Rect) {
                        // 如果是 Rect 对象，创建一个新的临时画布来渲染矩形
                        const tempRectCanvas = document.createElement('canvas');
                        tempRectCanvas.width = origWidth;
                        tempRectCanvas.height = origHeight;
                        const tempCtx = tempRectCanvas.getContext('2d');
                        
                        // 绘制矩形
                        tempCtx.fillStyle = currentBg.fill || '#000000';
                        tempCtx.fillRect(0, 0, origWidth, origHeight);
                        
                        // 使用临时画布作为源
                        ctx.drawImage(
                            tempRectCanvas,
                            left, top, scaledWidth, scaledHeight
                        );
                    } else if (currentBg._element) {
                        // 如果有 _element 属性（某些 fabric 对象的情况）
                        ctx.drawImage(
                            currentBg._element,
                            left, top, scaledWidth, scaledHeight
                        );
                    } else {
                        // 如果都不是，填充黑色
                        ctx.fillStyle = '#000000';
                        ctx.fillRect(0, 0, adjustedWidth, adjustedHeight);
                    }
            
                    // 创建新的背景图像
                    newBackground = new fabric.Image(tempCanvas, {
                        width: adjustedWidth,
                        height: adjustedHeight,
                        left: 0,
                        top: 0,
                        selectable: false,
                        evented: false,
                        excludeFromExport: false,
                        originX: 'left',
                        originY: 'top'
                    });
                } else {
                    // 如果没有背景，创建黑色矩形
                    newBackground = new fabric.Rect({
                        width: adjustedWidth,
                        height: adjustedHeight,
                        left: 0,
                        top: 0,
                        fill: '#000000',
                        selectable: false,
                        evented: false,
                        excludeFromExport: false
                    });
                }
            
                // 5. 设置新背景并等待完成
                await new Promise(resolve => {
                    node.canvasInstance.canvas.setBackgroundImage(newBackground, resolve);
                });
                
                // 6. 更新背景引用
                node.canvasInstance.background = newBackground;
            
                // 7. 更新画布尺寸和缩放
                node.canvasInstance.updateCanvasSize(scaledSize.width, scaledSize.height);
            
                // 8. 更新输入框显示实际尺寸
                widthInput.value = adjustedWidth;
                heightInput.value = adjustedHeight;
                localStorage.setItem(STORAGE_KEY + node.id, JSON.stringify({
                    width: adjustedWidth,
                    height: adjustedHeight
                }));

            
            } catch (error) {
                console.error('[FastCanvas] 更新尺寸失败:', error);
            }
        };

        // 创建颜色选择器
        const colorInput = document.createElement("input");
        colorInput.type = "color";
        colorInput.value = "#000000";
        colorInput.style.cssText = `
            padding: 0;
            width: 28px;
            height: 28px;
            background: #333;
            border: 1px solid #666;
            border-radius: 4px;
            cursor: pointer;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
        `;

        colorInput.oninput = (e) => {
            const newColor = e.target.value;
            if (node.canvasInstance?.canvas) {
                // 创建新的纯色矩形背景
                const newBackground = new fabric.Rect({
                    width: node.canvasInstance.originalSize.width,
                    height: node.canvasInstance.originalSize.height,
                    left: 0,
                    top: 0,
                    fill: newColor,
                    selectable: false,
                    evented: false,
                    excludeFromExport: false,
                    originX: 'left',
                    originY: 'top'
                });
                
                // 设置新背景
                node.canvasInstance.canvas.setBackgroundImage(newBackground);
                node.canvasInstance.background = newBackground;
                node.canvasInstance.canvas.renderAll();
            }
        };

        // 统一的样式规则
        if (!document.getElementById('fast-canvas-overlay-styles')) {
            const style = document.createElement('style');
            style.id = 'fast-canvas-overlay-styles'; // 添加ID避免重复
            style.textContent = `
                input[type="number"]::-webkit-inner-spin-button,
                input[type="number"]::-webkit-outer-spin-button {
                    -webkit-appearance: none;
                    margin: 0;
                }
                input[type="number"] {
                    -moz-appearance: textfield;
                }
                input[type="number"]::placeholder {
                    color: #888;
                    opacity: 1;
                    text-align: center;
                }
                input[type="color"]::-webkit-color-swatch-wrapper {
                    padding: 0;
                }
                input[type="color"]::-webkit-color-swatch {
                    border: none;
                    border-radius: 2px;
                }
                input[type="color"]::-moz-color-swatch {
                    border: none;
                    border-radius: 2px;
                }
            `;
            document.head.appendChild(style);
        }

        // 创建文件输入框（隐藏）
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.style.display = 'none';
        
        // 创建上传按钮
        const uploadButton = document.createElement("button");
        uploadButton.textContent = "上传图片";
        uploadButton.style.cssText = buttonStyle;  // 使用已有的按钮样式
        
        // 处理点击上传按钮
        uploadButton.onclick = () => {
            fileInput.click();
        };
        
        // 处理文件选择
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (file && node.canvasInstance) {
                await node.canvasInstance.handleImageUpload(file);
            }
            // 清除文件选择，允许选择相同文件
            fileInput.value = '';
        };
        const lockButton = document.createElement("button");
        lockButton.innerHTML = '🔓'; // 默认未锁定状态
        lockButton.style.cssText = buttonStyle;
        lockButton.style.marginLeft = 'auto'; // 将按钮推到最右边
        lockButton.dataset.locked = 'false';
        
        const updateLockStyle = (isLocked) => {
            if (isLocked) {
                lockButton.innerHTML = '🔒';
                lockButton.style.background = '#663c3c';
                overlayContainer.classList.add('canvas-locked');
                
                // 禁用控制按钮和输入
                colorInput.disabled = true;
                widthInput.disabled = true;
                heightInput.disabled = true;
                updateButton.disabled = true;
                uploadButton.disabled = true;
                
                if (node.canvasInstance?.canvas) {
                    const canvas = node.canvasInstance.canvas;
                    
                    // 创建透明覆盖层
                    if (!canvas.lockOverlay) {
                        const overlayDiv = document.createElement('div');
                        overlayDiv.style.cssText = `
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                            z-index: 1000;
                            background: transparent;
                            cursor: default;
                        `;
                        canvas.wrapperEl.appendChild(overlayDiv);
                        canvas.lockOverlay = overlayDiv;
                    }
                    
                    canvas.selection = false;
                    canvas.skipTargetFind = true;
                    canvas.renderAll();
                }
            } else {
                lockButton.innerHTML = '🔓';
                lockButton.style.background = '#444';
                overlayContainer.classList.remove('canvas-locked');
                
                // 启用控制按钮和输入
                colorInput.disabled = false;
                widthInput.disabled = false;
                heightInput.disabled = false;
                updateButton.disabled = false;
                uploadButton.disabled = false;
                
                if (node.canvasInstance?.canvas) {
                    const canvas = node.canvasInstance.canvas;
                    
                    // 移除覆盖层
                    if (canvas.lockOverlay) {
                        canvas.wrapperEl.removeChild(canvas.lockOverlay);
                        delete canvas.lockOverlay;
                    }
                    
                    canvas.selection = true;
                    canvas.skipTargetFind = false;
                    canvas.renderAll();
                }
            }
        };
        
        // 添加点击事件
        lockButton.onclick = () => {
            const isLocked = lockButton.dataset.locked === 'true';
            const newLockState = !isLocked;
            lockButton.dataset.locked = newLockState.toString();
            updateLockStyle(newLockState);
            
            // 保存锁定状态到节点实例
            if (node.canvasInstance) {
                node.canvasInstance.isLocked = newLockState;
            }
        };
        
        // 添加禁用状态的样式
        const additionalStyles = `
            .canvas-locked button:disabled,
            .canvas-locked input:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .canvas-locked input[type="color"]:disabled {
                pointer-events: none;
            }
        `;
        
        // 添加样式到已有的样式表中
        if (document.getElementById('fast-canvas-overlay-styles')) {
            const styleElement = document.getElementById('fast-canvas-overlay-styles');
            styleElement.textContent += additionalStyles;
        }
        // 将元素添加到容器
        overlayContainer.appendChild(colorInput);
        overlayContainer.appendChild(widthInput);
        overlayContainer.appendChild(heightInput);
        overlayContainer.appendChild(updateButton);
        overlayContainer.appendChild(uploadButton); 
        overlayContainer.appendChild(fileInput); 
        overlayContainer.appendChild(lockButton);
        
        // 保存更新函数引用
        overlayContainer.updateSizeInputs = updateSizeInputs;
        if (node.canvasInstance?.isLocked) {
            lockButton.dataset.locked = 'false';
            updateLockStyle(false);
        }
        return overlayContainer;
    }
}

function node_info_copy(src, dest, connect_both) {
	// copy input connections
	for(let i in src.inputs) {
		let input = src.inputs[i];
		if (input.widget !== undefined) {
			const destWidget = dest.widgets.find(x => x.name === input.widget.name);
			dest.convertWidgetToInput(destWidget);
		}
		if(input.link) {
			let link = app.graph.links[input.link];
			let src_node = app.graph.getNodeById(link.origin_id);
			src_node.connect(link.origin_slot, dest.id, input.name);
		}
	}

	// copy output connections
	if(connect_both) {
		let output_links = {};
		for(let i in src.outputs) {
			let output = src.outputs[i];
			if(output.links) {
				let links = [];
				for(let j in output.links) {
					links.push(app.graph.links[output.links[j]]);
				}
				output_links[output.name] = links;
			}
		}

		for(let i in dest.outputs) {
			let links = output_links[dest.outputs[i].name];
			if(links) {
				for(let j in links) {
					let link = links[j];
					let target_node = app.graph.getNodeById(link.target_id);
					dest.connect(parseInt(i), target_node, link.target_slot);
				}
			}
		}
	}

	dest.color = src.color;
	dest.bgcolor = src.bgcolor;
	dest.size = src.size;

	app.graph.afterChange();
}

// 修改节点执行处理
app.registerExtension({
    name: "Custom.FastCanvas",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "FastCanvas") {           
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 添加随机种子输入框
                const seedWidget = this.addWidget(
                    "text",
                    "seed",
                    "initial_seed",  // 给一个初始值
                    (v) => {
                        this.seed = v;
                    },
                    {
                        multiline: false,
                        readonly: true
                    }
                );

                // 生成随机种子的方法
                this.generateSeed = () => {
                    const timestamp = Date.now();
                    const random = Math.floor(Math.random() * 1000000);
                    return `${timestamp}_${random}`;
                };

                // 更新种子值的方法
                this.updateSeed = () => {
                    const newSeed = this.generateSeed();
                    const seedWidget = this.widgets?.find(w => w.name === "seed");
                    if (seedWidget) {
                        seedWidget.value = newSeed;
                        this.seed = newSeed;  // 确保也更新节点的属性
                        
                        if (!this.widgets_values) {
                            this.widgets_values = [];
                        }
                        const widgetIndex = this.widgets.findIndex(w => w.name === "seed");
                        if (widgetIndex !== -1) {
                            this.widgets_values[widgetIndex] = newSeed;
                        }
                        
                        this.setDirtyCanvas(true, false);
                    }
                };
                
                // 添加队列执行前的钩子到种子widget上
                seedWidget.beforeQueued = () => {
                    // 如果画布未锁定，则更新种子
                    if (!this.canvasInstance?.isLocked) {
                        this.updateSeed();
                    }
                };

                return result;
            };

            nodeType.prototype.onAdded = function() {
                if (this.id !== undefined && this.id !== -1) {

                    // 从本地存储获取保存的尺寸
                    const STORAGE_KEY = 'canvas_size_';
                    const savedSize = localStorage.getItem(STORAGE_KEY + this.id);
                    const size = savedSize ? JSON.parse(savedSize) : {
                        width: CANVAS_SIZE.WIDTH,
                        height: CANVAS_SIZE.HEIGHT
                    };
                    
                    // 创建画布容器
                    const element = document.createElement("div");
                    element.style.position = "relative";
                    element.style.width = "100%";
                    element.style.height = "100%";
                    
                    // 存储 element 引用
                    this.canvasElement = element;
                    
                    // 先创建画布实例
                    this.canvasInstance = new FastCanvas(this, size);
                    
                    // 计算缩放后的尺寸
                    const scaledSize = this.canvasInstance.calculateScaledSize(
                        size.width,
                        size.height,
                        this.canvasInstance.maxDisplaySize
                    );
            
                    // 更新容器尺寸
                    element.style.minWidth = `${scaledSize.width}px`;
                    element.style.minHeight = `${scaledSize.height + CANVAS_SIZE.CONTROL_PANEL_HEIGHT}px`;
                    
                    // 更新画布尺寸
                    this.canvasInstance.updateCanvasSize(scaledSize.width, scaledSize.height);
                    
                    // 更新节点尺寸计算方法
                    this.computeSize = () => [
                        scaledSize.width + CANVAS_SIZE.RIGHT_MARGIN,
                        scaledSize.height + CANVAS_SIZE.BOTTOM_MARGIN + CANVAS_SIZE.CONTROL_PANEL_HEIGHT
                    ];
            
                    // 创建覆盖层
                    const overlay = FastCanvasOverlay.createOverlay(this);
                    element.appendChild(overlay);
                    this.sizeOverlay = overlay;
            
                    // 添加画布组件
                    this.canvasWidget = this.addDOMWidget("canvas", "canvas", element);
                    
                    // 添加画布到容器
                    this.canvasElement.appendChild(this.canvasInstance.canvasContainer);
                    
                    // 强制更新节点尺寸
                    requestAnimationFrame(() => {
                        this.size = this.computeSize();
                        this.setDirtyCanvas(true, true);
                        
                        // 如果有保存的尺寸，立即更新输入框显示
                        if (this.sizeOverlay?.updateSizeInputs) {
                            this.sizeOverlay.updateSizeInputs(size.width, size.height);
                        }
                    });
                }
            };


            nodeType.prototype.onRemoved = function() {
                if (!this) return;
                
                const nodeId = this.id;
                
                // 清理实例
                cleanup(nodeId);
                
                // 清除引用
                if (this.canvasInstance) {
                    this.canvasInstance = null;
                }

            };
            // 在FastCanvas节点定义中添加
            nodeType.prototype.onConfigure = function(o) {
                // 不要调用原始的onConfigure方法，因为会导致递归
                
                // 发送need_update=true标记到后端
                setTimeout(() => {
                    api.fetchApi('/fast_canvas_refresh_signal', {
                        method: 'POST',
                        body: JSON.stringify({
                            node_id: this.id.toString(),
                            need_update: true
                        })
                    }).then(() => {
                        console.log(`[FastCanvas] 已发送刷新信号给节点 ${this.id}`);
                    }).catch(error => {
                        console.error(`[FastCanvas] 发送刷新信号失败:`, error);
                    });
                }, 0);
                

            };


        }
    }
});


app.registerExtension({
    name: "Custom.FastCanvasTool",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "FastCanvasTool") {
            // 处理连接变化
            nodeType.prototype.onConnectionsChange = async function(type, index, connected, link_info) {
                if (!link_info || type == LiteGraph.OUTPUT) return;
                
                const stackTrace = new Error().stack;
                
                // 如果是背景图片端口，不进行删除操作
                if (this.inputs[index]?.name === "bg_img") {
                    return;
                }

                // 处理断开连接
                if (!connected) {
                    if (!stackTrace.includes('LGraphNode.prototype.connect') && 
                        !stackTrace.includes('LGraphNode.connect') && 
                        !stackTrace.includes('loadGraphData')) {
                        
                        // 只删除非背景图片的输入端口
                        if (this.inputs[index]?.name.startsWith("img_")) {
                            this.removeInput(index);
                        }
                    }
                }
        
                // 处理新连接
                if (connected) {
                    await app.queuePromise;
                }
        
                // 重新编号图层输入（跳过背景图片端口）
                let imgIndex = 1;
                this.inputs.forEach(input => {
                    if (input.name.startsWith("img_")) {
                        const newName = `img_${imgIndex}`;
                        if (input.name !== newName) {
                            input.name = newName;
                        }
                        imgIndex++;
                    }
                });
        
                // 只有在最后一个非背景端口被连接时添加新端口
                const nonBgInputs = this.inputs.filter(input => input.name.startsWith("img_"));
                const lastInput = nonBgInputs[nonBgInputs.length - 1];
                
                if (lastInput?.link != null) {
                    const newIndex = `img_${imgIndex}`;
                    this.addInput(newIndex, "IMAGE");
                }

                this.setDirtyCanvas(true, true);
            };

            // 添加初始化方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // 确保始终有一个背景图片端口和一个图层端口
                if (!this.inputs.find(input => input.name === "bg_img")) {
                    this.addInput("bg_img", "IMAGE");
                }
                if (!this.inputs.find(input => input.name === "img_1")) {
                    this.addInput("img_1", "IMAGE");
                }
            };
        }
    }
});
