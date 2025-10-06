import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

class EventManager {
  constructor() {
    this.listeners = new Map();
  }

  addEventListener(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  removeEventListener(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  dispatchEvent(event, detail = {}) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      callbacks.forEach(callback => {
        try {
          callback({ detail });
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }
}

class QueueManager {
  constructor() {
    this.eventManager = new EventManager();
    this.queueNodeIds = null;
    this.processingQueue = false;
    this.lastAdjustedMouseEvent = null;
    this.isLGTriggered = false; // 标记是否由 LG 扩展触发
    this.initializeHooks();
  }

  initializeHooks() {
    const originalQueuePrompt = app.queuePrompt;
    const originalGraphToPrompt = app.graphToPrompt;
    const originalApiQueuePrompt = api.queuePrompt;

    app.queuePrompt = async function() {
      this.processingQueue = true;
      this.eventManager.dispatchEvent("queue");
      try {
        await originalQueuePrompt.apply(app, [...arguments]);
      } finally {
        this.processingQueue = false;
        this.eventManager.dispatchEvent("queue-end");
      }
    }.bind(this);

    app.graphToPrompt = async function() {
      this.eventManager.dispatchEvent("graph-to-prompt");
      let promise = originalGraphToPrompt.apply(app, [...arguments]);
      await promise;
      this.eventManager.dispatchEvent("graph-to-prompt-end");
      return promise;
    }.bind(this);

    api.queuePrompt = async function(index, prompt, ...args) {
      // 仅在 LG 扩展触发时才修改 prompt.output
      if (this.isLGTriggered && this.queueNodeIds && this.queueNodeIds.length && prompt.output) {
        const oldOutput = prompt.output;
        let newOutput = {};
        for (const queueNodeId of this.queueNodeIds) {
          this.recursiveAddNodes(String(queueNodeId), oldOutput, newOutput);
        }
        prompt.output = newOutput;
      }
      
      this.eventManager.dispatchEvent("comfy-api-queue-prompt-before", {
        workflow: prompt.workflow,
        output: prompt.output,
      });
      
      const response = originalApiQueuePrompt.apply(api, [index, prompt, ...args]);
      this.eventManager.dispatchEvent("comfy-api-queue-prompt-end");
      return response;
    }.bind(this);

    const originalProcessMouseDown = LGraphCanvas.prototype.processMouseDown;
    const originalAdjustMouseEvent = LGraphCanvas.prototype.adjustMouseEvent;
    const originalProcessMouseMove = LGraphCanvas.prototype.processMouseMove;

    LGraphCanvas.prototype.processMouseDown = function(e) {
      const result = originalProcessMouseDown.apply(this, [...arguments]);
      queueManager.lastAdjustedMouseEvent = e;
      return result;
    };

    LGraphCanvas.prototype.adjustMouseEvent = function(e) {
      originalAdjustMouseEvent.apply(this, [...arguments]);
      queueManager.lastAdjustedMouseEvent = e;
    };

    LGraphCanvas.prototype.processMouseMove = function(e) {
      const result = originalProcessMouseMove.apply(this, [...arguments]);
      if (e && !e.canvasX && !e.canvasY) {
        const canvas = app.canvas;
        const offset = canvas.convertEventToCanvasOffset(e);
        e.canvasX = offset[0];
        e.canvasY = offset[1];
      }
      queueManager.lastAdjustedMouseEvent = e;
      return result;
    };
  }
  recursiveAddNodes(nodeId, oldOutput, newOutput) {
    let currentId = nodeId;
    let currentNode = oldOutput[currentId];
    if (newOutput[currentId] == null) {
      newOutput[currentId] = currentNode;
      for (const inputValue of Object.values(currentNode.inputs || [])) {
        if (Array.isArray(inputValue)) {
          this.recursiveAddNodes(inputValue[0], oldOutput, newOutput);
        }
      }
    }
    return newOutput;
  }
  async queueOutputNodes(nodeIds) {
    try {
      this.queueNodeIds = nodeIds;
      this.isLGTriggered = true; // 设置 LG 触发标记
      await app.queuePrompt();
    } catch (e) {
      console.error("队列节点时出错:", e);
    } finally {
      this.queueNodeIds = null;
      this.isLGTriggered = false; // 清除 LG 触发标记
    }
  }
  getLastMouseEvent() {
    return this.lastAdjustedMouseEvent;
  }
  addEventListener(event, callback) {
    this.eventManager.addEventListener(event, callback);
  }
  removeEventListener(event, callback) {
    this.eventManager.removeEventListener(event, callback);
  }
}

function getOutputNodes(nodes) {
  return (nodes?.filter((n) => {
    return (n.mode != LiteGraph.NEVER &&
      n.constructor.nodeData?.output_node);
  }) || []);
}
const queueManager = new QueueManager();
function queueSelectedOutputNodes() {
  const selectedNodes = app.canvas.selected_nodes;
  if (!selectedNodes || Object.keys(selectedNodes).length === 0) {
    console.log("[LG]队列: 没有选中的节点");
    return;
  }

  const outputNodes = getOutputNodes(Object.values(selectedNodes));
  if (!outputNodes || outputNodes.length === 0) {
    console.log("[LG]队列: 选中的节点中没有输出节点");
    return;
  }

  console.log(`[LG]队列: 执行 ${outputNodes.length} 个输出节点`);
  queueManager.queueOutputNodes(outputNodes.map((n) => n.id));
}

function queueGroupOutputNodes() {
  const lastMouseEvent = queueManager.getLastMouseEvent();
  if (!lastMouseEvent) {
    return;
  }

  let canvasX = lastMouseEvent.canvasX;
  let canvasY = lastMouseEvent.canvasY;
  
  if (!canvasX || !canvasY) {
    const canvas = app.canvas;
    const mousePos = canvas.getMousePos();
    canvasX = mousePos[0];
    canvasY = mousePos[1];
  }

  const group = app.graph.getGroupOnPos(canvasX, canvasY);

  if (!group) {
    return;
  }

  group.recomputeInsideNodes();

  if (!group._nodes || group._nodes.length === 0) {
    return;
  }
  
  const outputNodes = getOutputNodes(group._nodes);
  if (!outputNodes || outputNodes.length === 0) {
    return;
  }

  queueManager.queueOutputNodes(outputNodes.map((n) => n.id));
}

app.registerExtension({
  name: "LG.QueueNodes",
  commands: [
    {
      id: "LG.QueueSelectedOutputNodes",
      icon: "pi pi-play",
      label: "执行选中的输出节点",
      function: queueSelectedOutputNodes
    },
    {
      id: "LG.QueueGroupOutputNodes", 
      icon: "pi pi-sitemap",
      label: "执行组内输出节点",
      function: queueGroupOutputNodes
    }
  ]
});

export { queueManager, getOutputNodes, queueSelectedOutputNodes, queueGroupOutputNodes }; 

