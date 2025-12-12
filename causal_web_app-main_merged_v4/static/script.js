let needRecommendation = false;
let ifbuttonRecommendation = false;
function showRecommendation(needRec) {
    needRecommendation = needRec; // 设置needRecommendation的值
    console.log(needRecommendation)
    if (needRec) {
        openRecommendationTypeModal();
    } else {
        showModifiedGraph();
        ifbuttonRecommendation = true;
    }
}

// 全局变量存储tom-select实例
let featureQualitySelect = null;

function openRecommendationTypeModal() {
    const modal = document.getElementById('recommendationTypeModal');
    if (modal) {
        // 获取节点列表填充特征品质下拉框
        fetch('/get_nodes')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('featureQuality');
                
                // 如果已有tom-select实例，先销毁
                if (featureQualitySelect) {
                    featureQualitySelect.destroy();
                    featureQualitySelect = null;
                }
                
                // 填充选项
                select.innerHTML = '';
                data.nodes.forEach(node => {
                    select.innerHTML += `<option value="${node}">${node}</option>`;
                });
                
                // 初始化tom-select多选下拉框
                featureQualitySelect = new TomSelect('#featureQuality', {
                    plugins: ['remove_button'],
                    placeholder: '选择特征品质（可多选）',
                    maxItems: null
                });
            })
            .catch(err => console.error('获取节点列表失败:', err));
        modal.style.display = 'block';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

//符磊于2025/4/9修改 添加获取原始因果图的Nodes和edges
//start
function getOriginNodes_Edges(){
        const iframe = document.getElementById("leftGraph");
        try {

                // 获取 iframe 内部的 document 对象
                const iframeDoc = iframe.contentWindow;
                // 在 iframe 内部查询节点（例如查找 class="node" 的元素）
                let nodes_obj = iframeDoc.allNodes
                let result = Object.entries(nodes_obj).map(([key, value]) => ({
  ...value,
  "node_name":key
}))

                const node_propertiesToExclude = ["color", "physics", "shape", "x", "y"];


                 const nodes =    result.map(node =>{
                    const entries = Object.entries(node).filter(([key]) => !node_propertiesToExclude.includes(key));
                    return Object.fromEntries(entries);
                });


                let edges_obj = iframeDoc.allEdges
                let edges_result = Object.entries(edges_obj).map(([key, value]) => ({
  ...value,

}))




                 const edges =    edges_result.map(node =>{
                    const entries = Object.entries(node).filter(([key]) => key!=="id");
                    return Object.fromEntries(entries);
                });


                if (nodes && edges) {

                        return JSON.stringify( {"nodes":nodes,"edges":edges})



                } else {
                    alert("没找到")
                    return null
                }
            } catch (error) {
            alert(error)
                console.error('访问 iframe 失败:', error);
                return null
            }


}






//end
// function confirmRecommendation() {
//     const recommendationType = document.querySelector('input[name="recommendationType"]:checked');
//     if (!recommendationType) {
//         alert('请选择推荐类型');
//         return;
//     }
//     const featureQuality = document.getElementById('featureQuality').value; // 获取特征品质
//     console.log("特征品质:", featureQuality);

//     let dataFile, dataAlgorithm;
//     if (recommendationType.value === 'dataBased') {
//         dataFile = document.getElementById('dataFile').files[0];
//         dataAlgorithm = document.getElementById('dataAlgorithm').value;
//     } else {
//         dataFile = document.getElementById('knowledgeFile').files[0];
//         dataAlgorithm = document.getElementById('knowledgeAlgorithm').value;
//     }

//     // 符磊于2025/4/9修改 获得候选因果图
//     const t = getOriginNodes_Edges();

//     // 创建 FormData 对象来发送文件和数据
//     const formData = new FormData();
//     formData.append('featureQuality', featureQuality);
//     formData.append('causal_recommendation', 'yes');
//     formData.append('File', dataFile);
//     formData.append('Algorithm', dataAlgorithm);
//     formData.append('candidates', t);

//     // 发送 POST 请求到 Flask 后端
//     fetch('/get_graph', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.json())
//     .then(data => {
//         alert(data.message);
//         // 刷新右图的显示
//         refreshModifiedIframe();
//         loadPendingList();
//     })
//     .catch(error => {
//         console.error('Error:', error);
//         alert('Failed to fetch data');
//     });

//     // 修改右侧框的src为modified_graph
//     const rightGraph = document.getElementById('rightGraph');
//     if (rightGraph) {
//         rightGraph.src = staticUrls.modifiedGraph + "?v=" + Date.now();
//     }
//     closeModal('recommendationTypeModal');
//     const recommendationPrompt = document.getElementById('recommendationPrompt');
//     if (recommendationPrompt) {
//         recommendationPrompt.style.display = 'none';
//     }
// }

function confirmRecommendation() {
    const recommendationType = document.querySelector('input[name="recommendationType"]:checked');
    if (!recommendationType) {
        alert('请选择推荐类型');
        return;
    }
    // 获取多选的特征品质（从tom-select实例获取）
    const featureQuality = featureQualitySelect ? featureQualitySelect.getValue().join(',') : '';
    console.log("特征品质:", featureQuality);

    let dataFile, dataAlgorithm;
    if (recommendationType.value === 'dataBased') {
        dataFile = document.getElementById('dataFile').files[0];
        dataAlgorithm = document.getElementById('dataAlgorithm').value;
    } else {
        dataFile = document.getElementById('knowledgeFile').files[0];
        dataAlgorithm = document.getElementById('knowledgeAlgorithm').value;
    }

    // 显示“正在生成中”的提示
    const loadingModal = document.getElementById('loadingModal');
    if (loadingModal) {
        loadingModal.style.display = 'block';
    }
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.style.display = 'block';
    document.body.appendChild(overlay);

    // 符磊于2025/4/9修改 获得候选因果图
    const t = getOriginNodes_Edges();

    // 创建 FormData 对象来发送文件和数据
    const formData = new FormData();
    formData.append('featureQuality', featureQuality);
    formData.append('causal_recommendation', 'yes');
    formData.append('File', dataFile);
    formData.append('Algorithm', dataAlgorithm);
    formData.append('candidates', t);

    // 发送 POST 请求到 Flask 后端
    fetch('/get_graph', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隐藏等待提示
        if (loadingModal) {
            loadingModal.style.display = 'none';
        }
        overlay.style.display = 'none';
        document.body.removeChild(overlay);
    
        // 后端显示alert框，提示后端返回的信息
        // alert(data.message);
        console.log(data.message);
    
        // 在alert框关闭后刷新右侧框的显示
        refreshModifiedIframe();
        loadPendingList();
        ifbuttonRecommendation = true;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('推荐失败，请稍后再试');
        if (loadingModal) {
            loadingModal.style.display = 'none';
        }
        overlay.style.display = 'none';
        document.body.removeChild(overlay);
    });

    closeModal('recommendationTypeModal');
    const recommendationPrompt = document.getElementById('recommendationPrompt');
    if (recommendationPrompt) {
        recommendationPrompt.style.display = 'none';
    }
}


// function showModifiedGraph() {
//     const rightGraph = document.getElementById('rightGraph');
//     if (rightGraph) {
//         rightGraph.src = staticUrls.modifiedGraph + "?v=" + Date.now();
//     }
//     const recommendationPrompt = document.getElementById('recommendationPrompt');
//     if (recommendationPrompt) {
//         recommendationPrompt.style.display = 'none';
//     }
// }
function showModifiedGraph() {
    const selectedVariety = sessionStorage.getItem('selectedVariety');
    if (!selectedVariety) {
        alert("未选择品种，请返回主界面选择品种");
        return;
    }

    // 调用后端接口，将原始图重新加载到修改图
    fetch('/cp_modified', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variety_name: selectedVariety })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        refreshOriginalIframe();
        refreshModifiedIframe();
    })
    .catch(err => console.error(err));

    // 隐藏推荐提示
    const recommendationPrompt = document.getElementById('recommendationPrompt');
    if (recommendationPrompt) {
        recommendationPrompt.style.display = 'none';
    }
}

const rightGraph = document.getElementById('rightGraph');
if (rightGraph) {
    rightGraph.onload = function () {
        const iframeDocument = rightGraph.contentDocument || rightGraph.contentWindow.document;
        const canvas = iframeDocument.querySelector('canvas');

        if (canvas) {
            canvas.addEventListener('dblclick', function (event) {
                const rect = canvas.getBoundingClientRect();
                const x = event.clientX - rect.left;
                const y = event.clientY - rect.top;

                const network = iframeDocument.defaultView.network;
                const clickedNode = getNodeAtPosition(network, x, y);
                const clickedEdge = getEdgeAtPosition(network, x, y);

                if (clickedNode) {
                    openNodeModal('update', clickedNode);
                } else if (clickedEdge) {
                    openEdgeModal('update', clickedEdge.from, clickedEdge.to);
                }
            });
        }
    };
}

function getNodeAtPosition(network, x, y) {
    const nodeId = network.getNodeAt({ x: x, y: y });
    return nodeId || null;
}

function getEdgeAtPosition(network, x, y) {
    const edgeId = network.getEdgeAt({ x: x, y: y });
    if (edgeId) {
        const edge = network.body.data.edges.get(edgeId);
        return edge || null;
    }
    return null;
}

function openNodeModal(action, nodeName = '') {
    if (!ifbuttonRecommendation) {
        alert('请先选择是否需要因果推荐');
    } else {
        const modal = document.getElementById('nodeModal');
        const title = document.getElementById('nodeModalTitle');
        const nodeInput = document.getElementById('nodeNameInput');
        const nodeSelect = document.getElementById('nodeNameSelect');
        
        if (modal && title) {
            title.textContent = action === 'add' ? '添加节点' : action === 'delete' ? '删除节点' : '更新节点';
            
            // 添加节点用输入框，删除/更新用下拉框
            if (action === 'add') {
                nodeInput.style.display = 'block';
                nodeSelect.style.display = 'none';
                nodeInput.value = nodeName || '';
            } else {
                nodeInput.style.display = 'none';
                nodeSelect.style.display = 'block';
                // 获取节点列表填充下拉框
                fetch('/get_nodes')
                    .then(r => r.json())
                    .then(data => {
                        nodeSelect.innerHTML = '<option value="" disabled selected>选择节点 (必填)</option>';
                        data.nodes.forEach(node => {
                            nodeSelect.innerHTML += `<option value="${node}">${node}</option>`;
                        });
                        if (nodeName) nodeSelect.value = nodeName;
                    });
            }
            
            document.getElementById('nodeType').value = '';
            document.getElementById('nodeSource').value = '';
            modal.style.display = 'block';
        }
    }
}

function openEdgeModal(action, startNode = '', endNode = '') {
    if (!ifbuttonRecommendation) {
        alert('请先选择是否需要因果推荐');
    } else {
        const modal = document.getElementById('edgeModal');
        const title = document.getElementById('edgeModalTitle');
        if (modal && title) {
            title.textContent = action === 'add' ? '添加边' : action === 'delete' ? '删除边' : '更新边';
            
            // 获取节点列表并填充下拉框
            fetch('/get_nodes')
                .then(response => response.json())
                .then(data => {
                    const startSelect = document.getElementById('startNode');
                    const endSelect = document.getElementById('endNode');
                    
                    // 清空并重新填充选项
                    startSelect.innerHTML = '<option value="" disabled selected>起始节点 (必填)</option>';
                    endSelect.innerHTML = '<option value="" disabled selected>终止节点 (必填)</option>';
                    
                    data.nodes.forEach(node => {
                        startSelect.innerHTML += `<option value="${node}">${node}</option>`;
                        endSelect.innerHTML += `<option value="${node}">${node}</option>`;
                    });
                    
                    // 如果有预设值则选中
                    if (startNode && endNode) {
                        startSelect.value = startNode;
                        endSelect.value = endNode;
                        const edgeInfo = getEdgeInfo(startNode, endNode);
                        if (edgeInfo) {
                            document.getElementById('edgeRelation').value = edgeInfo.relation || '';
                            document.getElementById('edgeSource').value = edgeInfo.source || '';
                        }
                    } else {
                        document.getElementById('edgeRelation').value = '';
                        document.getElementById('edgeSource').value = '';
                    }
                })
                .catch(err => console.error('获取节点列表失败:', err));
            
            modal.style.display = 'block';

            const changeDirectionButton = modal.querySelector('button[onclick="changeEdgeDirection()"]');
            if (changeDirectionButton) {
                changeDirectionButton.style.display = action === 'update' ? 'inline-block' : 'none';
            }
        }
    }
    
}

// 提交“更新节点”或“添加节点”表单
function submitNodeForm() {
    const title = document.getElementById('nodeModalTitle').textContent;
    // 根据操作类型从不同元素获取节点名称
    let nodeName;
    if (title === '添加节点') {
        nodeName = document.getElementById('nodeNameInput').value;
    } else {
        nodeName = document.getElementById('nodeNameSelect').value;
    }
    
    if (!nodeName) {
        alert('节点名称是必填项！');
        return;
    }
    const nodeType = document.getElementById('nodeType').value;    // 节点类型
    const nodeSource = document.getElementById('nodeSource').value; // 节点来源

    let action = '';
    if (title === '添加节点') {
        action = 'add_node';
    } else if (title === '删除节点') {
        action = 'remove_node';
    } else if (title === '更新节点') {
        action = 'update_node';
    }

    // 提交后端
    fetch('/modify_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            action: action,
            start_node: nodeName,
            attrs: {
                type: nodeType,   // ✅ 修改为 "type"
                source: nodeSource
            }
        })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        closeModal('nodeModal');
        refreshModifiedIframe();
        loadPendingList();
    })
    .catch(err => console.error(err));
}

// 同理，提交流/更新/删除边时，也可传 relation、source 等
// 切换自定义关系输入框显示
function toggleCustomRelation() {
    const select = document.getElementById('edgeRelation');
    const customInput = document.getElementById('customRelation');
    if (select.value === '其他') {
        customInput.style.display = 'block';
    } else {
        customInput.style.display = 'none';
        customInput.value = '';
    }
}

function submitEdgeForm() {
    const startNode = document.getElementById('startNode').value;
    const endNode = document.getElementById('endNode').value;
    if (!startNode || !endNode) {
        alert('起始节点和终止节点是必填项！');
        return;
    }
    // 检查起点和终点不能相同
    const title = document.getElementById('edgeModalTitle').textContent;
    if ((title === '添加边' || title === '更新边') && startNode === endNode) {
        alert('起始节点和终止节点不能相同！');
        return;
    }
    // 处理关系类型：如果选择"其他"则使用自定义输入
    let edgeRelation = document.getElementById('edgeRelation').value;
    if (edgeRelation === '其他') {
        const customRelation = document.getElementById('customRelation').value.trim();
        if (!customRelation) {
            alert('请输入自定义关系类型！');
            return;
        }
        edgeRelation = customRelation;
    }
    const edgeSource = document.getElementById('edgeSource').value;

    let action = '';
    if (title === '添加边') {
        action = 'add_edge';
    } else if (title === '删除边') {
        action = 'remove_edge';
    } else if (title === '更新边') {
        action = 'update_edge';
    }

    fetch('/modify_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            action: action,
            start_node: startNode,
            end_node: endNode,
            attrs: { relation: edgeRelation, source: edgeSource }
        })
    })
    .then(r => r.json())
    .then(data => {
        alert(data.message);
        closeModal('edgeModal');
        refreshModifiedIframe();
        loadPendingList();
    })
    .catch(err => console.error(err));
}

function getNodeInfo(nodeName) {
    const nodes = {
        'SoilAcidity': { type: 'EnvironmentalFactor', source: 'SoilTest' },
        'SoilFertility': { type: 'EnvironmentalFactor', source: 'SoilTest' },
    };
    return nodes[nodeName] || null;
}

function getEdgeInfo(startNode, endNode) {
    const edges = {
        'SoilAcidity-SoilFertility': { relation: 'Negative', source: 'AgriculturalResearch' },
        'SoilFertility-CropYield': { relation: 'Positive', source: 'AgriculturalResearch' },
    };
    return edges[`${startNode}-${endNode}`] || null;
}

function refreshModifiedIframe() {
    const ifr = document.getElementById("rightGraph");
    if (ifr) {
        ifr.src = staticUrls.modifiedGraph + "?v=" + Date.now();
        console.log("刷新:",ifr.src)
    }
}

function loadPendingList() {
    fetch('/list_pending_changes')
        .then(r => r.json())
        .then(data => {
            const tbody = document.getElementById("pendingChangesList");
            if (tbody) {
                tbody.innerHTML = "";
                data.pending_changes.forEach(item => {
                    // item = { id, change_type, start, end, ... }
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td>${item.change_type}</td>
                        <td>${item.start || ''}</td>
                        <td>${item.end || ''}</td>
                        <!-- 撤销按钮传递 item.id -->
                        <td>
                          <button onclick="rejectChange('${item.id}')">撤销</button>
                        </td>
                    `;
                    tbody.appendChild(row);
                });
            }
        });
}


// 撤销修改
function rejectChange(pendingId) {
    fetch('/reject_change', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pending_id: pendingId })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        loadPendingList();
        refreshModifiedIframe();
    });
}

function loadVarieties() {
  fetch('/get_varieties')
    .then(r => r.json())
    .then(data => {
      const sel = document.getElementById("variety_select");
      if (sel) {
        sel.innerHTML = "";
        data.variety_list.forEach(v => {
            const opt = document.createElement("option");
            opt.value = v;
            opt.innerText = v;
            sel.appendChild(opt);
        });

        // 优先级：URL参数 > localStorage中的最近选择 > 默认品种
        const urlParams = new URLSearchParams(window.location.search);
        const varietyParam = urlParams.get('variety');
        const lastSelectedVariety = localStorage.getItem('lastSelectedVariety');
        
        let selectedVariety = '';
        
        if (varietyParam) {
            // URL参数优先级最高
            const option = Array.from(sel.options).find(opt => opt.value === varietyParam);
            if (option) {
                selectedVariety = varietyParam;
            }
        } else if (lastSelectedVariety) {
            // 其次使用localStorage中保存的最近选择
            const option = Array.from(sel.options).find(opt => opt.value === lastSelectedVariety);
            if (option) {
                selectedVariety = lastSelectedVariety;
            }
        } else {
            // 最后使用默认品种
            selectedVariety = data.default_variety || '';
        }
        
        if (selectedVariety) {
            sel.value = selectedVariety;
            const defaultVarietySpan = document.getElementById("default_variety_span");
            if (defaultVarietySpan) {
                defaultVarietySpan.innerText = selectedVariety;
            }
            showVarietyImage();
        }
      }
    });
}

function deleteVariety() {
  const sel = document.getElementById("variety_select");
  if (sel) {
    const vName = sel.value;
    if (!vName) {
      alert("请选择要删除的品种");
      return;
    }

    const confirmed = confirm(`确定要删除品种 "${vName}" 吗？此操作将删除其所有相关文件。`);
    if (!confirmed) {
      alert("已取消删除");
      return;
    }

    fetch('/delete_variety', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variety_name: vName })
    })
    .then(r => r.json())
    .then(data => {
        alert(data.message);
        loadVarieties();
    })
    .catch(err => console.log(err));
  }
}

function showVarietyImage() {
  const sel = document.getElementById("variety_select");
  if (sel) {
    const vName = sel.value;
    
    // 保存最近选择的品种到localStorage
    if (vName) {
        localStorage.setItem('lastSelectedVariety', vName);
    }
    
    const ifr = document.getElementById("variety_html_iframe");
    if (ifr) {
        // 先重新生成HTML（自动计算分散布局），再加载
        fetch('/regenerate_variety_html/' + encodeURIComponent(vName))
        .then(() => {
            ifr.src = "static/variety_htmls/" + vName + ".html?" + Date.now();
        })
        .catch(() => {
            ifr.src = "static/variety_htmls/" + vName + ".html?" + Date.now();
        });
    }
  }

}
// 更新可视化函数
function visualizeVarietyInBoth() {
    const sel = document.getElementById("variety_select");
    const vName = sel.value;
    fetch('/visualize_variety_in_both', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({variety_name: vName})
    })
    .then(response => {
        refreshOriginalIframe();
        refreshModifiedIframe();
    });
}

// function visualizeVarietyInBoth() {
//     const sel = document.getElementById("variety_select");
//     if (sel) {
//         const vName = sel.value;
//         if (!vName) {
//             alert("请选择品种");
//             return;
//         }
//         fetch('/visualize_variety_in_both', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ variety_name: vName })
//         })
//         .then(response => response.json())
//         .then(data => {
//             alert(data.message);
//             refreshOriginalIframe();
//             refreshModifiedIframe();
//         })
//         .catch(err => console.error(err));
//     }
// }

function refreshOriginalIframe() {
    const ifr = document.getElementById("leftGraph");
    if (ifr) {
        ifr.src = ifr.src.split("?")[0] + "?v=" + Date.now();
    }
}

function downloadChanges() {
    fetch('/list_pending_changes')
    .then(response => response.json())
    .then(data => {
        const changes = data.pending_changes.map(item => ({
            type: item.type,
            start: item.start,
            end: item.end
        }));
        const blob = new Blob([JSON.stringify(changes, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'changes.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
}

function showFilePathDialog(elementId) {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.onchange = function() {
        const inputElement = document.getElementById(elementId + '_input');
        if (inputElement) {
            inputElement.value = this.value;
        }
    };
    fileInput.click();
}

function changeEdgeDirection() {
    const startNode = document.getElementById('startNode').value;
    const endNode = document.getElementById('endNode').value;
    if (startNode && endNode) {
        fetch('/modify_graph', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                action: 'change_direction',
                start_node: startNode,
                end_node: endNode,
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.message) {
                alert(data.message);
                closeModal('edgeModal');
                refreshModifiedIframe();
                loadPendingList();
            }
        });
    } else {
        alert('请先选择边！');
    }
}

window.onload = function() {
    bindTreeIframeEvents();
    loadVarieties();
    loadPendingList();
    showDefaultVarietyGraph();
};

function showDefaultVarietyGraph() {
    const sel = document.getElementById("variety_select");
    if (sel) {
        const defaultVariety = sel.value;
        if (defaultVariety) {
            showVarietyImage();
        }
    }
}



// 修改 goToIndex 函数
function goToIndex() {
    if (!checkVarietySelected()) {
        return;
    }
    const sel = document.getElementById("variety_select");
    if (sel) {
        const vName = sel.value;
        if (!vName) {
            alert("请选择品种");
            return;
        }
        
        // 保存最近选择的品种
        localStorage.setItem('lastSelectedVariety', vName);
        
        fetch('/visualize_variety_in_both', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ variety_name: vName })
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
            window.location.href = '/index';
        })
        .catch(err => console.error(err));
    }
}



// 修改 checkVarietyAndNavigate 函数
function checkVarietyAndNavigate(url) {
    if (!checkVarietySelected()) {
        return;
    }
    window.location.href = url;
}

// 修改 returnToMain 函数，在返回主界面时清空已选品种
function returnToMain() {
    sessionStorage.removeItem('selectedVariety');
    const selectedVarietyName = document.getElementById("selected_variety_name");
    if (selectedVarietyName) {
        selectedVarietyName.innerText = '';
    }
    window.location.href = '/';
}

function saveModifiedGraph() {
    const selectedVariety = sessionStorage.getItem('selectedVariety');
    if (!selectedVariety) {
        alert("未找到品种信息");
        return;
    }
    fetch('/save_modified_graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variety_name: selectedVariety })
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
    })
    .catch(err => console.error(err));
}

function confirmVarietySelection() {
    const sel = document.getElementById("variety_select");
    if (sel) {
        const vName = sel.value;
        if (!vName) {
            alert("请选择品种");
            return;
        }
        showVarietyImage();
        
        // 保存到sessionStorage和localStorage
        sessionStorage.setItem('selectedVariety', vName);
        localStorage.setItem('lastSelectedVariety', vName); // 保存最近选择的品种
        
        const selectedVarietyName = document.getElementById("selected_variety_name");
        if (selectedVarietyName) {
            selectedVarietyName.innerText = vName;
        }
        const urlParams = new URLSearchParams(window.location.search);
        urlParams.set('variety', vName);
        window.history.replaceState({}, '', `${window.location.pathname}?${urlParams.toString()}`);
        
        // 确认选择后：隐藏选择文件组，显示上传候选图按钮
        const beforeConfirmGroup = document.getElementById("before_confirm_group");
        const uploadCandidateBtn = document.getElementById("upload_candidate_btn");
        if (beforeConfirmGroup) {
            beforeConfirmGroup.style.display = "none";
        }
        if (uploadCandidateBtn) {
            uploadCandidateBtn.style.display = "inline-block";
        }
    }
}


function addVariety() {
    const vName = document.getElementById("variety_name_input").value.trim();

    if (!vName) {
        alert("请输入品种名称");
        return;
    }

    const formData = new FormData();
    formData.append("variety_name", vName);

    fetch('/add_variety', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        loadVarieties();
        document.getElementById("variety_name_input").value = '';
    })
    .catch(err => console.log(err));
}

function openUploadCandidateModal() {
    const selectedVariety = sessionStorage.getItem('selectedVariety');
    if (!selectedVariety) {
        alert("请先确认一个品种，再上传候选图");
        return;
    }

    // 直接打开模态框
    const modal = document.getElementById('candidateUploadModal');
    const overlay = document.getElementById('candidateUploadOverlay');
    if (!modal) {
        return;
    }

    const varietyInfo = document.getElementById('candidate_modal_variety');
    if (varietyInfo) {
        varietyInfo.innerText = `当前品种：${selectedVariety}`;
    }

    const fileInput = document.getElementById('candidate_file_input');
    const fileDisplay = document.getElementById('candidate_file_name');
    if (fileInput) {
        fileInput.value = "";
    }
    if (fileDisplay) {
        fileDisplay.innerText = "未选择任何文件";
        fileDisplay.style.color = '#888';
    }

    modal.style.display = 'block';
    if (overlay) {
        overlay.style.display = 'block';
    }
}

function closeCandidateUploadModal() {
    const modal = document.getElementById('candidateUploadModal');
    const overlay = document.getElementById('candidateUploadOverlay');
    if (modal) {
        modal.style.display = 'none';
    }
    if (overlay) {
        overlay.style.display = 'none';
    }

    const fileInput = document.getElementById('candidate_file_input');
    const fileDisplay = document.getElementById('candidate_file_name');
    if (fileInput) {
        fileInput.value = "";
    }
    if (fileDisplay) {
        fileDisplay.innerText = "未选择任何文件";
        fileDisplay.style.color = '#888';
    }
}

function confirmCandidateUpload() {
    const selectedVariety = sessionStorage.getItem('selectedVariety');
    if (!selectedVariety) {
        alert("未检测到已确认的品种");
        return;
    }

    const fileInput = document.getElementById('candidate_file_input');
    if (!fileInput || !fileInput.files[0]) {
        alert("请先选择候选图 JSON 文件");
        return;
    }

    const confirmUpload = confirm(`确定将候选图覆盖到 "${selectedVariety}" 吗？`);
    if (!confirmUpload) {
        return;
    }

    const formData = new FormData();
    formData.append('variety_name', selectedVariety);
    formData.append('json_file', fileInput.files[0]);

    fetch('/upload_candidate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        closeCandidateUploadModal();
        showVarietyImage();
    })
    .catch(err => {
        console.log(err);
        alert('上传候选图失败，请稍后再试');
    });
}

// 添加一个函数来检查品种是否已确认
function checkVarietySelected() {
    const selectedVariety = sessionStorage.getItem('selectedVariety');
    if (!selectedVariety) {
        alert("请确定品种！");
        return false;
    }
    return true;
}


function toggleAlgorithmOptions() {
    const dataBased = document.getElementById('dataBased');
    const knowledgeBased = document.getElementById('knowledgeBased');
    const dataBasedOptions = document.getElementById('dataBasedOptions');
    const knowledgeBasedOptions = document.getElementById('knowledgeBasedOptions');

    if (dataBased.checked) {
        dataBasedOptions.style.display = 'block';
        knowledgeBasedOptions.style.display = 'none';
    } else if (knowledgeBased.checked) {
        knowledgeBasedOptions.style.display = 'block';
        dataBasedOptions.style.display = 'none';
    }
}

// 版本管理树节点选择事件
// function handleNodeSelect(versionName) {
//     console.log("handleNodeSelect called with:", versionName);
    
//     const selectedVariety = sessionStorage.getItem('selectedVariety');
//     if (!selectedVariety) {
//         alert("未选择品种，请返回主界面选择品种");
//         return;
//     }
    
//     fetch('/select_version', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//             variety_name: selectedVariety,
//             version_name: versionName
//         })
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error(`HTTP error! status: ${response.status}`);
//         }
//         return response.json();
//     })
//     .then(data => {
//         console.log("Data from /select_version:", data); // 调试信息
        
//         // 更新版本管理树的 iframe
//         const treeIframe = document.getElementById('treeIframe');
//         if (treeIframe) {
//             treeIframe.src = `static/version_control/${encodeURIComponent(selectedVariety)}/version_tree.html?v=${Date.now()}`;
//         }
        
//         // 更新当前版本因果图的 iframe
//         const rightGraph = document.getElementById('rightGraph');
//         if (rightGraph) {
//             rightGraph.src = data.path + "?v=" + Date.now();
//         }
//     })
//     .catch(error => {
//         console.error("Error in handleNodeSelect:", error); // 调试信息
//     });
// }
function handleNodeSelect(versionName) {
    console.log("handleNodeSelect called with:", versionName);
    
    const selectedVariety = sessionStorage.getItem('selectedVariety');
    if (!selectedVariety) {
        alert("未选择品种，请返回主界面选择品种");
        return;
    }
    
    fetch('/select_version', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            variety_name: selectedVariety,
            version_name: versionName
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Data from /select_version:", data); // 调试信息
        
        // 更新版本管理树的 iframe
        const treeIframe = document.getElementById('treeIframe');
        if (treeIframe) {
            treeIframe.src = `static/version_control/${encodeURIComponent(selectedVariety)}/version_tree.html?v=${Date.now()}`;
        }
        
        // 更新当前版本因果图的 iframe
        const rightGraph = document.getElementById('rightGraph');
        if (rightGraph) {
            rightGraph.src = data.path + "?v=" + Date.now();
            
            // 更新当前选中版本的文本
            document.getElementById('currentVersionSpan').innerText = versionName;
        }
    })
    .catch(error => {
        console.error("Error in handleNodeSelect:", error); // 调试信息
    });
}

// 为版本管理树的 iframe 绑定双击事件
function bindTreeIframeEvents() {
    console.log("bindTreeIframeEvents called");
    const treeIframe = document.getElementById('treeIframe');
    
    if (treeIframe) {
        console.log("treeIframe found");
        
        treeIframe.onload = function() {
            console.log("treeIframe loaded");
            
            try {
                const iframeDocument = treeIframe.contentDocument || treeIframe.contentWindow.document;
                console.log("iframeDocument accessed");
                
                // 获取ECharts实例
                const chartContainers = iframeDocument.getElementsByClassName('chart-container');
                if (chartContainers.length > 0) {
                    console.log("chartContainers found");
                    
                    // 通过ECharts API获取所有实例
                    const echarts = treeIframe.contentWindow.echarts;
                    const instances = echarts.getInstanceByDom(chartContainers[0]);
                    
                    if (instances) {
                        console.log("ECharts instance found");
                        
                        // 添加点击事件监听
                        instances.on('click', function(params) {
                            console.log("Node clicked:", params);
                            
                            // 获取版本名称
                            const versionName = params.name || params.data.name;
                            if (versionName) {
                                console.log("Selected version:", versionName);
                                handleNodeSelect(versionName);
                            }
                        });
                    }
                }
            } catch (error) {
                console.error("Error accessing iframe content:", error);
            }
        };
    }
}


