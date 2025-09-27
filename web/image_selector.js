import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Comfyui_LG_Tools.ImageSelector",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImageSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                this.selected_images = new Set();
                this.anti_selected = new Set();
                this.currentMode = "always_pause";
                this.isWaitingSelection = false;
                this.isCancelling = false;
                this.imageData = [];
                
                this.isChooser = true;
                this.imageIndex = null;
                
                this.confirmButton = this.addWidget("button", "确认选择", "", () => {
                    this.executeSelection();
                });
                
                this.cancelButton = this.addWidget("button", "取消", "", () => {
                    this.cancelSelection();
                });
                
                this.confirmButton.serialize = false;
                this.cancelButton.serialize = false;
                
                Object.defineProperty(this.confirmButton, "clicked", {
                    get: function() {
                        return this._clicked;
                    },
                    set: function(value) {
                        this._clicked = value && "" != this.name;
                    }
                });
                
                Object.defineProperty(this.cancelButton, "clicked", {
                    get: function() {
                        return this._clicked;
                    },
                    set: function(value) {
                        this._clicked = value && "" != this.name;
                    }
                });
                
                Object.defineProperty(this, "imageIndex", {
                    get: function() {
                        return null;
                    },
                    set: function(value) {
                    }
                });
                
                this.ensurePropertiesValid();
                this.updateWidgets();
                
                const canvas = app.canvas.canvas;
                if (canvas && !this._globalMouseHandler) {
                    this._globalMouseHandler = (event) => {
                        if (!this.imgs || this.imgs.length === 0 || !this.imageRects) {
                            return;
                        }
                        
                        const mouse = app.canvas.graph_mouse;
                        
                        for (let i = 0; i < this.imageRects.length; i++) {
                            const [rectX, rectY, rectWidth, rectHeight] = this.imageRects[i];
                            
                            const absoluteX = rectX + this.pos[0];
                            const absoluteY = rectY + this.pos[1];
                            
                            const isInside = LiteGraph.isInsideRectangle(
                                mouse[0], mouse[1],
                                absoluteX, absoluteY,
                                rectWidth, rectHeight
                            );
                            
                            if (isInside) {
                                event.preventDefault();
                                event.stopPropagation();
                                this.toggleImageSelection(i);
                                this.setDirtyCanvas(true, true);
                                return;
                            }
                        }
                    };
                    
                    canvas.addEventListener('click', this._globalMouseHandler, true);
                }
                
                return result;
            };

            nodeType.prototype.ensurePropertiesValid = function() {
                if (!(this.selected_images instanceof Set)) {
                    this.selected_images = new Set();
                }
                
                if (!(this.anti_selected instanceof Set)) {
                    this.anti_selected = new Set();
                }
            };

            const onDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                this.ensurePropertiesValid();
                
                this.pointerDown = null;
                this.overIndex = null;
                
                const originalImgs = this.imgs;
                this.imgs = null;
                
                const result = onDrawBackground?.apply(this, arguments);
                
                this.imgs = originalImgs;
                
                this.drawImagesAndSelection(ctx);
                
                return result;
            };
            
            nodeType.prototype.drawImagesAndSelection = function(ctx) {
                if (!this.imgs || this.imgs.length === 0) return;
                
                if (this.imageRects) {
                    for (let i = 0; i < this.imgs.length; i++) {
                        if (i >= this.imageRects.length) break;
                        
                        const [rectX, rectY, rectWidth, rectHeight] = this.imageRects[i];
                        const img = this.imgs[i];
                        
                        ctx.fillStyle = "#000";
                        ctx.fillRect(rectX, rectY, rectWidth, rectHeight);
                        
                        // 检查图像是否有效加载
                        if (img && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
                            let scaleX = rectWidth / img.naturalWidth;
                            let scaleY = rectHeight / img.naturalHeight;
                            let scale = Math.min(scaleX, scaleY);
                            
                            let imgHeight = scale * img.naturalHeight;
                            let imgWidth = scale * img.naturalWidth;
                            
                            const imgX = rectX + (rectWidth - imgWidth) / 2;
                            const imgY = rectY + (rectHeight - imgHeight) / 2;
                            
                            const margin = 2;
                            ctx.drawImage(img, imgX + margin, imgY + margin, imgWidth - 2 * margin, imgHeight - 2 * margin);
                        }
                    }
                }
                
                ctx.lineWidth = 2;
                this.selected_images.forEach(index => {
                    if (index < this.imageRects.length) {
                        const [x, y, width, height] = this.imageRects[index];
                        
                        ctx.strokeStyle = "green";
                        ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
                        
                        const checkSize = 20;
                        const checkX = x + width - checkSize - 5;
                        const checkY = y + 5;
                        
                        ctx.fillStyle = '#4CAF50';
                        ctx.beginPath();
                        ctx.arc(checkX + checkSize/2, checkY + checkSize/2, checkSize/2, 0, 2 * Math.PI);
                        ctx.fill();
                        
                        ctx.strokeStyle = '#ffffff';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(checkX + 6, checkY + checkSize/2);
                        ctx.lineTo(checkX + checkSize/2, checkY + checkSize - 6);
                        ctx.lineTo(checkX + checkSize - 4, checkY + 6);
                        ctx.stroke();
                        
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                        ctx.fillRect(x + 2, y + 2, 25, 18);
                        ctx.fillStyle = '#ffffff';
                        ctx.font = '12px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText((index + 1).toString(), x + 14, y + 15);
                    }
                });
                
                if (this.overIndex !== null && this.overIndex >= 0 && this.overIndex < this.imageRects.length) {
                    const [x, y, width, height] = this.imageRects[this.overIndex];
                    
                    if (!this.selected_images.has(this.overIndex)) {
                        ctx.lineWidth = 2;
                        ctx.strokeStyle = "rgba(0, 255, 0, 0.5)";
                        ctx.setLineDash([5, 5]);
                        ctx.strokeRect(x + 1, y + 1, width - 2, height - 2);
                        ctx.setLineDash([]);
                        
                        ctx.fillStyle = 'rgba(0, 255, 0, 0.3)';
                        ctx.fillRect(x + 2, y + 2, 60, 18);
                        ctx.fillStyle = '#ffffff';
                        ctx.font = '10px Arial';
                        ctx.textAlign = 'left';
                        ctx.fillText('点击选择', x + 5, y + 14);
                    }
                }
            };
            
            const onMouseDown = nodeType.prototype.onMouseDown;
            nodeType.prototype.onMouseDown = function(event, localPos, graphCanvas) {
                this.ensurePropertiesValid();
                
                if (event.isPrimary && this.imgs && this.imgs.length > 0) {
                    if (!this.imageRects || this.imageRects.length === 0) {
                        return onMouseDown?.apply(this, arguments);
                    }
                    
                    const imageIndex = this.getImageIndexFromClick(localPos);
                    
                    if (imageIndex >= 0) {
                        event.preventDefault();
                        event.stopPropagation();
                        
                        this.pointerDown = { imageIndex: imageIndex, localPos: [...localPos] };
                        this.overIndex = imageIndex;
                        
                        this.setDirtyCanvas(true, true);
                        
                        return true;
                    } else {
                        this.pointerDown = null;
                        this.overIndex = null;
                    }
                }
                
                return onMouseDown?.apply(this, arguments);
            };
            
            const onMouseMove = nodeType.prototype.onMouseMove;
            nodeType.prototype.onMouseMove = function(event, localPos, graphCanvas) {
                if (this.isChooser && this.pointerDown) {
                    const currentImageIndex = this.getImageIndexFromClick(localPos);
                    this.overIndex = (currentImageIndex === this.pointerDown.imageIndex) ? currentImageIndex : null;
                    this.setDirtyCanvas(true, true);
                } else if (this.isChooser) {
                    this.overIndex = null;
                    this.pointerDown = null;
                }
                
                return onMouseMove?.apply(this, arguments);
            };
            
            const onMouseUp = nodeType.prototype.onMouseUp;
            nodeType.prototype.onMouseUp = function(event, localPos, graphCanvas) {
                if (this.isChooser && this.pointerDown) {
                    const currentImageIndex = this.getImageIndexFromClick(localPos);
                    
                    if (currentImageIndex === this.pointerDown.imageIndex) {
                        this.toggleImageSelection(currentImageIndex);
                    }
                    
                    this.pointerDown = null;
                    this.overIndex = null;
                    this.setDirtyCanvas(true, true);
                } else if (this.isChooser) {
                    this.pointerDown = null;
                    this.overIndex = null;
                }
                
                return onMouseUp?.apply(this, arguments);
            };

            const originalUpdate = nodeType.prototype.update;
            nodeType.prototype.update = function() {
                this.ensurePropertiesValid();
                
                this.updateWidgets();
                this.calculateImageLayout();
                this.setDirtyCanvas(true, true);
                
                if (originalUpdate && originalUpdate !== this.update) {
                    originalUpdate.apply(this, arguments);
                }
            };

            const originalOnResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                this.ensurePropertiesValid();
                
                if (this.imgs && this.imgs.length > 0) {
                    this.calculateImageLayout();
                }
                
                if (originalOnResize) {
                    originalOnResize.apply(this, arguments);
                }
            };
            
            nodeType.prototype.calculateImageLayout = function() {
                if (!this.imgs || this.imgs.length === 0) {
                    this.imageRects = [];
                    return;
                }
                
                const titleHeight = LiteGraph.NODE_TITLE_HEIGHT || 30;
                const widgetHeight = 30;
                const imageTextHeight = 15;
                const margin = 4;
                
                let widgetsHeight = 0;
                if (this.widgets && this.widgets.length > 0) {
                    const visibleWidgets = this.widgets.filter(w => !w.hidden);
                    widgetsHeight = visibleWidgets.length * widgetHeight + margin;
                }
                
                const shiftY = titleHeight + widgetsHeight;
                
                const dw = this.size[0] - margin * 2;
                const dh = this.size[1] - shiftY - imageTextHeight - margin;
                
                this.imageRects = [];
                
                if (this.imgs.length === 1) {
                    const img = this.imgs[0];
                    if (img.naturalWidth && img.naturalHeight) {
                        let w = img.naturalWidth;
                        let h = img.naturalHeight;
                        
                        const scaleX = dw / w;
                        const scaleY = dh / h;
                        const scale = Math.min(scaleX, scaleY, 1);
                        
                        w *= scale;
                        h *= scale;
                        
                        const x = margin + (dw - w) / 2;
                        const y = shiftY + (dh - h) / 2;
                        
                        this.imageRects.push([x, y, w, h]);
                    }
                } else {
                    const numImages = this.imgs.length;
                    
                    let w = this.imgs[0].naturalWidth;
                    let h = this.imgs[0].naturalHeight;
                    
                    let bestLayout = null;
                    let bestArea = 0;
                    
                    for (let cols = 1; cols <= numImages; cols++) {
                        const rows = Math.ceil(numImages / cols);
                        
                        const availableWidthPerImage = dw / cols;
                        const availableHeightPerImage = dh / rows;
                        
                        const scaleX = availableWidthPerImage / w;
                        const scaleY = availableHeightPerImage / h;
                        const scale = Math.min(scaleX, scaleY, 1);
                        
                        const imageWidth = w * scale;
                        const imageHeight = h * scale;
                        
                        const actualGridWidth = cols * imageWidth;
                        const actualGridHeight = rows * imageHeight;
                        
                        if (actualGridWidth <= dw && actualGridHeight <= dh) {
                            const area = imageWidth * imageHeight * numImages;
                            
                            if (area > bestArea) {
                                bestArea = area;
                                bestLayout = {
                                    cols: cols,
                                    rows: rows,
                                    imageWidth: imageWidth,
                                    imageHeight: imageHeight,
                                    actualGridWidth: actualGridWidth,
                                    actualGridHeight: actualGridHeight,
                                    scale: scale
                                };
                            }
                        }
                    }
                    
                    if (bestLayout) {
                        const gridOffsetX = margin + (dw - bestLayout.actualGridWidth) / 2;
                        const gridOffsetY = shiftY + (dh - bestLayout.actualGridHeight) / 2;
                        
                        for (let i = 0; i < numImages; i++) {
                            const row = Math.floor(i / bestLayout.cols);
                            const col = i % bestLayout.cols;
                            
                            const imgX = gridOffsetX + col * bestLayout.imageWidth;
                            const imgY = gridOffsetY + row * bestLayout.imageHeight;
                            
                            this.imageRects.push([imgX, imgY, bestLayout.imageWidth, bestLayout.imageHeight]);
                        }
                    } else {
                        const imageHeight = Math.min(dh / numImages, 100);
                        const imageWidth = imageHeight * (w / h);
                        
                        for (let i = 0; i < numImages; i++) {
                            const imgX = margin + (dw - imageWidth) / 2;
                            const imgY = shiftY + i * imageHeight;
                            this.imageRects.push([imgX, imgY, imageWidth, imageHeight]);
                        }
                    }
                }
            };
            
            nodeType.prototype.getImageIndexFromClick = function(pos) {
                if (!this.imageRects) {
                    return -1;
                }
                
                for (let i = 0; i < this.imageRects.length; i++) {
                    const [x, y, width, height] = this.imageRects[i];
                    const isInside = pos[0] >= x && pos[0] <= x + width &&
                                   pos[1] >= y && pos[1] <= y + height;
                    
                    if (isInside) {
                        return i;
                    }
                }
                
                return -1;
            };
            
            nodeType.prototype.toggleImageSelection = function(index) {
                this.ensurePropertiesValid();
                
                const wasSelected = this.selected_images.has(index);
                
                if (wasSelected) {
                    this.selected_images.delete(index);
                } else {
                    this.selected_images.add(index);
                }
                
                this.update();
            };
            
            nodeType.prototype.executeSelection = function() {
                if (!this.isWaitingSelection) {
                    return;
                }
                
                this.ensurePropertiesValid();
                
                const selectedIndices = Array.from(this.selected_images);
                
                fetch('/image_selector/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        node_id: this.id.toString(),
                        action: 'select',
                        selected_indices: selectedIndices
                    })
                }).then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        console.error(`选择请求失败:`, data.error);
                    }
                }).catch(error => {
                    console.error(`选择请求异常:`, error);
                });

                this.isWaitingSelection = false;
                this.update();
                this.setExecutingState(false);
            };
            
            nodeType.prototype.cancelSelection = function(source = "manual") {
                if (!this.isWaitingSelection) {
                    return;
                }
                
                this.isCancelling = true;
                this.update();
                
                fetch('/image_selector/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        node_id: this.id.toString(),
                        action: 'cancel',
                        selected_indices: []
                    })
                }).then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        console.error(`取消请求失败:`, data.error);
                    }
                }).catch(error => {
                    console.error(`取消请求异常:`, error);
                }).finally(() => {
                    this.isCancelling = false;
                    this.update();
                    this.setExecutingState(false);
                });
            };
            
            nodeType.prototype.updateWidgets = function() {
                if (!this.confirmButton || !this.cancelButton) return;
                
                this.ensurePropertiesValid();
                
                const selectedCount = this.selected_images.size;
                const totalCount = this.imgs ? this.imgs.length : 0;
                
                if (this.isCancelling) {
                    this.confirmButton.name = "正在取消...";
                    this.cancelButton.name = "";
                    this.confirmButton.disabled = true;
                    this.cancelButton.disabled = true;
                } else if (this.isWaitingSelection) {
                    if (selectedCount > 0) {
                        this.confirmButton.name = selectedCount > 1 ? 
                            `确认选择 (${selectedCount}/${totalCount})` : 
                            "确认选择";
                        this.confirmButton.disabled = false;
                    } else {
                        this.confirmButton.name = "请选择图像";
                        this.confirmButton.disabled = true;
                    }
                    this.cancelButton.name = "取消运行";
                    this.cancelButton.disabled = false;
                } else {
                    const modeText = {
                        "always_pause": "等待选择",
                        "keep_last_selection": "自动使用上次选择",
                        "passthrough": "自动通过"
                    }[this.currentMode] || "未知模式";
                    
                    this.confirmButton.name = modeText;
                    this.cancelButton.name = "";
                    this.confirmButton.disabled = true;
                    this.cancelButton.disabled = true;
                }
            };
            nodeType.prototype.setExecutingState = function(isExecuting) {
                this.isExecuting = isExecuting;
                this.strokeStyles = this.strokeStyles || {};
                this.strokeStyles['customExecuting'] = function() {
                    if (this.isExecuting) {
                        return { color: '#0f0' }; 
                    }
                    return null;
                };

                if (app.graph) {
                    app.graph.setDirtyCanvas(true, false);
                }
            };

            nodeType.prototype.serialize = function() {
                const data = LiteGraph.LGraphNode.prototype.serialize.call(this);
                
                data.isWaitingSelection = this.isWaitingSelection;
                data.currentMode = this.currentMode;
                
                if (this.selected_images && this.selected_images.size > 0) {
                    data.selected_images = Array.from(this.selected_images);
                }
                
                data.isExecuting = this.isExecuting || false;
                
                return data;
            };

            nodeType.prototype.configure = function(data) {
                LiteGraph.LGraphNode.prototype.configure.call(this, data);
                
                this.isWaitingSelection = data.isWaitingSelection || false;
                this.currentMode = data.currentMode || "always_pause";
                
                this.ensurePropertiesValid();
                if (data.selected_images && Array.isArray(data.selected_images)) {
                    this.selected_images.clear();
                    data.selected_images.forEach(index => {
                        this.selected_images.add(index);
                    });
                }
                
                if (data.isExecuting) {
                    setTimeout(() => this.setExecutingState(true), 100);
                }
                
                this.updateWidgets();
            };

        }
    },
    
    setup() {
        const originalInterrupt = api.interrupt;
        
        api.interrupt = function() {
            if (app.graph && app.graph._nodes_by_id) {
                Object.values(app.graph._nodes_by_id).forEach(node => {
                    if (node.isChooser && node.isWaitingSelection) {
                        node.cancelSelection("interrupt");
                    }
                });
            }
            
            originalInterrupt.apply(this, arguments);
        };

        api.addEventListener("image_selector_update", (event) => {
            const data = event.detail;
            
            const node = app.graph._nodes_by_id[data.id];
            if (!node || !node.isChooser) {
                return;
            }
            node.setExecutingState(false);
            
            const imageData = data.urls.map((url, index) => ({
                index: index,
                filename: url.filename,
                subfolder: url.subfolder,
                type: url.type
            }));
            
            const modeWidget = node.widgets.find(w => w.name === "mode");
            const currentMode = modeWidget ? modeWidget.value : "always_pause";
            
            node.currentMode = currentMode;
            node.isWaitingSelection = (currentMode === "always_pause" || currentMode === "keep_last_selection");
            node.isCancelling = false;
            node.imageData = imageData;
            
            node.ensurePropertiesValid();
            node.selected_images.clear();
            node.anti_selected.clear();
            
            node.pointerDown = null;
            node.overIndex = null;
            
            node.imgs = [];
            let loadedCount = 0;
            
            imageData.forEach((imgData, i) => {
                const img = new Image();
                img.onload = () => {
                    loadedCount++;
                    
                    if (loadedCount === imageData.length) {
                        node.calculateImageLayout();
                        app.graph.setDirtyCanvas(true);
                    }
                };
                
                img.src = api.apiURL(`/view?filename=${encodeURIComponent(imgData.filename)}&type=${imgData.type}&subfolder=${imgData.subfolder}&rand=${Math.random()}`);
                node.imgs.push(img);
            });

            node.update();
            if (currentMode === "always_pause") {
                node.setExecutingState(true);
            }
        });
        
        api.addEventListener("image_selector_selection", (event) => {
            const data = event.detail;
            
            const node = app.graph._nodes_by_id[data.id];
            if (node && node.isChooser) {
                node.isWaitingSelection = false;
                node.isCancelling = false;
                
                if (data.selected_indices && Array.isArray(data.selected_indices)) {
                    node.ensurePropertiesValid();
                    node.selected_images.clear();
                    data.selected_indices.forEach(index => {
                        node.selected_images.add(index);
                    });
                }
                
                node.update();
            }
        });
    }
}); 