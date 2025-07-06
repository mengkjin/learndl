import sys , pathlib
file_path = str(pathlib.Path(__file__).absolute())
assert 'learndl' in file_path , f'learndl path not found , do not know where to find src file : {file_path}'
path = file_path.removesuffix(file_path.split('learndl')[-1])
if not path in sys.path: sys.path.append(path)

import os, platform, subprocess, yaml, re, time, base64, glob, json, signal
import streamlit as st
import streamlit.components.v1 as components
import psutil
from typing import Any, Literal
from pathlib import Path
from datetime import datetime

from src_runs.util import terminal_cmd

def load_queue():
    """加载运行队列"""
    queue_file = "run_queue.json"
    if os.path.exists(queue_file):
        try:
            with open(queue_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_queue(queue):
    """保存运行队列"""
    queue_file = "run_queue.json"
    with open(queue_file, 'w') as f:
        json.dump(queue, f, indent=2)

def add_to_queue(script_name, cmd):
    """添加到运行队列"""
    queue = load_queue()
    queue_item = {
        'id': f"{script_name}_{int(time.time())}",
        'script_name': script_name,
        'cmd': cmd,
        'status': 'starting',
        'created_time': time.time(),
        'pid': None,
        'start_time': None,
        'end_time': None
    }
    queue.append(queue_item)
    save_queue(queue)
    return queue_item

def update_queue_item(item_id, updates):
    """更新队列项状态"""
    queue = load_queue()
    for item in queue:
        if item['id'] == item_id:
            item.update(updates)
            break
    save_queue(queue)

def remove_from_queue(item_id):
    """从队列中移除项目"""
    queue = load_queue()
    queue = [item for item in queue if item['id'] != item_id]
    save_queue(queue)

def check_process_status(pid):
    """检查进程状态"""
    try:
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            return proc.status() in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]
        return False
    except:
        return False

def kill_process(pid):
    """终止进程"""
    try:
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            proc.terminate()
            time.sleep(2)
            if proc.is_running():
                proc.kill()
            return True
    except:
        pass
    return False

def load_output_manifest(script_name):
    """加载脚本输出文件清单"""
    manifest_file = "output_manifest.json"
    if os.path.exists(manifest_file):
        try:
            with open(manifest_file, 'r') as f:
                data = json.load(f)
                if data.get('script') == script_name:
                    return data.get('files', [])
        except:
            pass
    return []

def show_run_report(queue_item):
    """显示运行报告"""
    if queue_item['status'] != 'completed':
        st.warning("脚本还未完成，无法生成完整报告")
        return
    
    duration = queue_item.get('end_time', 0) - queue_item.get('start_time', 0)
    
    st.subheader("📊 运行报告")
    
    # 基本信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("运行时间", f"{duration:.2f}秒")
    with col2:
        st.metric("脚本名称", queue_item['script_name'])
    with col3:
        start_time_str = datetime.fromtimestamp(queue_item['start_time']).strftime('%H:%M:%S')
        st.metric("开始时间", start_time_str)
    
    # 检查脚本输出的文件清单
    output_files = load_output_manifest(queue_item['script_name'])
    
    # 显示生成的文件
    if output_files:
        st.subheader("📁 生成的文件")
        for file_path in output_files:
            if os.path.exists(file_path):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file_path}")
                with col2:
                    if file_path.endswith('.html'):
                        if st.button("预览", key=f"preview_{file_path}_{queue_item['id']}"):
                            preview_html_file(file_path)
                    elif file_path.endswith('.pdf'):
                        if st.button("预览", key=f"preview_{file_path}_{queue_item['id']}"):
                            preview_pdf_file(file_path)
                with col3:
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                "下载", 
                                data=f.read(),
                                file_name=os.path.basename(file_path),
                                key=f"download_{file_path}_{queue_item['id']}"
                            )
                    except:
                        st.error("文件读取失败")
            else:
                st.warning(f"⚠️ 文件不存在: {file_path}")
    else:
        st.info("💡 **提示**: 脚本可以通过创建 `output_manifest.json` 文件来报告生成的文件")
        with st.expander("📖 如何在脚本中输出文件清单", expanded=False):
            st.code('''
import json
from datetime import datetime

# 在脚本结束前添加以下代码：
output_files = ["output1.html", "output2.pdf"]  # 您的输出文件列表

manifest = {
    "script": "your_script_name",
    "files": output_files,
    "timestamp": datetime.now().isoformat()
}

with open("output_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
            ''', language='python')

def preview_html_file(file_path):
    """预览HTML文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.subheader(f"📄 {os.path.basename(file_path)}")
        components.html(html_content, height=600, scrolling=True)
    except Exception as e:
        st.error(f"无法预览HTML文件: {str(e)}")

def preview_pdf_file(file_path):
    """预览PDF文件"""
    try:
        with open(file_path, 'rb') as f:
            pdf_data = f.read()
        
        # 使用base64编码PDF
        pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{pdf_base64}" 
                width="100%" height="600px" type="application/pdf">
        </iframe>
        '''
        st.subheader(f"📄 {os.path.basename(file_path)}")
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"无法预览PDF文件: {str(e)}")
        st.info("您可以下载文件后查看")

class StreamlitScriptRunner:
    def __init__(self, script_path: Path | str):
        self.script = Path(script_path).absolute()
        self.header = self.parse_script_header()
        
    def parse_script_header(self, verbose=False, include_starter='#', exit_starter='', 
                          ignore_starters=('#!', '# coding:')):
        header_dict = {}
        yaml_lines: list[str] = []
        
        try:
            with open(self.script, 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line.startswith(ignore_starters): 
                        continue
                    elif stripped_line.startswith(include_starter):
                        yaml_lines.append(stripped_line)
                    elif stripped_line.startswith(exit_starter):
                        break

            yaml_str = '\n'.join(line.removeprefix(include_starter) for line in yaml_lines)
            header_dict = yaml.safe_load(yaml_str) or {}
            
        except FileNotFoundError:
            header_dict = {
                'disabled': True,
                'description': '文件未找到',
                'content': f'文件未找到: {self.script}'
            }
        except yaml.YAMLError as e:
            header_dict = {
                'disabled': True,
                'description': 'YAML解析错误',
                'content': f'错误信息: {e}'
            }
        except Exception as e:
            header_dict = {
                'disabled': True,
                'description': '文件读取错误',
                'content': f'错误信息: {e}'
            }

        if 'description' not in header_dict:
            header_dict['description'] = self.script.name
            
        return header_dict

    def get_param_inputs(self):
        """生成参数输入控件并返回参数值"""
        param_inputs = self.header.get('param_inputs', {})
        if not param_inputs:
            return {}
            
        st.subheader("参数设置")
        params = {}
        
        # 创建3列布局 - 所有参数同时显示
        param_items = list(param_inputs.items())
        num_cols = min(3, len(param_items))
        param_cols = st.columns(num_cols)
        
        # 先收集所有参数，避免依赖关系导致的逐步显示
        all_widgets = []
        for i, (pname, pdef) in enumerate(param_items):
            col_idx = i % num_cols
            all_widgets.append((col_idx, pname, pdef))
        
        # 同时渲染所有参数
        for col_idx, pname, pdef in all_widgets:
            with param_cols[col_idx]:
                try:
                    # 解析参数定义
                    ptype = pdef.get("type") or pdef.get("enum")
                    if isinstance(ptype, str):
                        ptype = eval(ptype)
                    elif isinstance(ptype, (list, tuple)):
                        ptype = list(ptype)
                        
                    required = pdef.get('required', False)
                    default = pdef.get('default')
                    desc = pdef.get('desc', pname)
                    prefix = pdef.get('prefix', '')
                    
                    # 生成唯一key
                    key = f"{self.script.name}_{pname}"
                    
                    # 创建输入控件
                    if isinstance(ptype, list):
                        # 下拉选择
                        options = [f'{prefix}{e}' for e in ptype]
                        placeholder = f"请选择{desc}" if required else f"可选: {desc}"
                        
                        if default is not None:
                            default_idx = ptype.index(default) if default in ptype else 0
                            value = st.selectbox(
                                f"**{desc}**" if required else desc,
                                options,
                                index=default_idx,
                                key=key
                            )
                        else:
                            value = st.selectbox(
                                f"**{desc}**" if required else desc,
                                [placeholder] + options,
                                key=key
                            )
                        
                        # 处理选择结果
                        if value == placeholder:
                            if required:
                                st.error(f"请为 [{desc}] 选择一个有效值！")
                                return None
                            else:
                                params[pname] = None
                        else:
                            # 去除prefix并找到原始值
                            enum_idx = options.index(value)
                            params[pname] = ptype[enum_idx]
                            
                    elif ptype == bool:
                        # 布尔开关
                        default_val = bool(eval(str(default))) if default is not None else False
                        value = st.toggle(
                            f"**{desc}**" if required else desc,
                            value=default_val,
                            key=key
                        )
                        params[pname] = value
                        
                    elif ptype in [str, int, float]:
                        # 文本/数字输入
                        placeholder = f"请输入{desc}" if required else f"可选: {desc}"
                        
                        if ptype == str:
                            value = st.text_input(
                                f"**{desc}**" if required else desc,
                                value=str(default) if default is not None else "",
                                placeholder=placeholder,
                                key=key
                            )
                            if required and (not value or value.strip() == ""):
                                st.error(f"请为 [{desc}] 输入一个有效值！")
                                return None
                            params[pname] = value if value.strip() else None
                            
                        elif ptype in [int, float]:
                            min_val = pdef.get('min')
                            max_val = pdef.get('max')
                            
                            if ptype == int:
                                value = st.number_input(
                                    f"**{desc}**" if required else desc,
                                    value=int(default) if default is not None else (min_val or 0),
                                    min_value=min_val,
                                    max_value=max_val,
                                    step=1,
                                    key=key
                                )
                            else:  # float
                                value = st.number_input(
                                    f"**{desc}**" if required else desc,
                                    value=float(default) if default is not None else (min_val or 0.0),
                                    min_value=min_val,
                                    max_value=max_val,
                                    step=0.1,
                                    key=key
                                )
                            params[pname] = value
                            
                except Exception as e:
                    st.error(f"参数 [{pname}] 配置错误: {str(e)}")
                    return None
                
        return params

    def render(self):
        """渲染脚本界面（保持兼容性）"""
        self.render_details()
    
    def render_details(self):
        """渲染脚本详细界面"""
        # TODO信息
        if todo := self.header.get('TODO'):
            st.info(f"📝 TODO: {todo}")
            
        # 检查是否禁用
        if self.header.get('disabled', False):
            st.error("该脚本已禁用")
            return
            
        # 获取参数输入
        params = self.get_param_inputs()
        if params is None:  # 参数验证失败
            return
            
        # 运行按钮
        if st.button("🚀 运行脚本", key=f"run_{self.script.name}", type="primary", use_container_width=True):
            # 添加默认参数
            run_params = {
                'email': int(self.header.get('email', 0)),
                'close_after_run': bool(self.header.get('close_after_run', False))
            }
            run_params.update({k: v for k, v in params.items() if v is not None})
            
            # 运行脚本
            self.run_script(**run_params)

    @staticmethod
    def run_script(script : str | Path , close_after_run = False , **kwargs):
        cmd = terminal_cmd(script, kwargs, close_after_run=close_after_run)
        script_name = Path(script).stem
        
        # 添加到运行队列
        queue_item = add_to_queue(script_name, cmd)
        st.info(f"✅ 已添加到队列: {queue_item['id']}")
        
        try:
            # 启动进程
            process = subprocess.Popen(cmd, shell=True, encoding='utf-8')
            
            # 更新队列状态
            update_queue_item(queue_item['id'], {
                'pid': process.pid,
                'status': 'running',
                'start_time': time.time()
            })
            
            st.success(f'✅ 脚本已启动！PID: {process.pid}')
            st.info('📊 请点击上方队列区域的"🔄 刷新"按钮查看最新状态')
            
            # 显示命令信息
            with st.expander("🔧 执行命令详情", expanded=False):
                st.code(cmd)
            
        except Exception as e:
            # 更新队列状态为失败
            update_queue_item(queue_item['id'], {
                'status': 'failed',
                'error': str(e),
                'end_time': time.time()
            })
            st.error(f'❌ 脚本启动失败: {str(e)}')

def show_folder(folder_path: Path | str, level: int = 0):
    """递归展示文件夹内容"""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        st.error(f"文件夹不存在: {folder_path}")
        return
        
    # 获取所有项目并排序
    items = []
    for item in folder_path.iterdir():
        if item.name.startswith(('.', '_')):
            continue
        items.append(item)
    
    items.sort(key=lambda x: (x.is_file(), x.name))
    
    # 显示文件夹标题（更紧凑的样式）
    if level > 0:
        folder_name = folder_path.name.replace('_', ' ').title()
        st.markdown(f"**📁 {folder_name}**")
    
    # 处理子文件夹和文件
    folders = [item for item in items if item.is_dir()]
    files = [item for item in items if item.is_file() and item.suffix == '.py']
    
    # 显示子文件夹（默认展开）
    for folder in folders:
        if level < 3:  # 限制递归深度
            show_folder(folder, level + 1)
    
    # 显示Python脚本（紧凑模式）
    if files:
        for script_file in files:
            show_script(script_file)

def show_run_queue():
    """显示运行队列"""
    queue = load_queue()
    
    # 队列标题和刷新按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        st.subheader("🔄 运行队列")
    with col2:
        if st.button("🔄 刷新", key="refresh_queue"):
            st.rerun()
    
    if not queue:
        st.info("🈳 队列为空，点击下方脚本运行后会在此显示进度")
        return
    
    # 更新队列状态
    updated_queue = []
    status_changed = False
    
    for item in queue:
        if item.get('pid') and item['status'] == 'running':
            if check_process_status(item['pid']):
                # 进程还在运行
                item['status'] = 'running'
            else:
                # 进程已结束
                old_status = item['status']
                item['status'] = 'completed'
                item['end_time'] = time.time()
                if old_status != 'completed':
                    status_changed = True
        updated_queue.append(item)
    
    if status_changed:
        save_queue(updated_queue)
        queue = updated_queue
    
    # 显示队列统计信息
    running_count = len([item for item in queue if item['status'] == 'running'])
    completed_count = len([item for item in queue if item['status'] == 'completed'])
    failed_count = len([item for item in queue if item['status'] == 'failed'])
    
    if running_count > 0 or completed_count > 0 or failed_count > 0:
        st.caption(f"📊 运行中: {running_count} | 已完成: {completed_count} | 失败: {failed_count}")
    
    # 显示队列项
    for item in queue:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                status_icon = {
                    'starting': '🟡',
                    'running': '🟢', 
                    'completed': '✅',
                    'failed': '❌'
                }.get(item['status'], '❓')
                
                st.markdown(f"**{status_icon} {item['script_name']}**")
                
            with col2:
                if item['status'] == 'running' and item.get('start_time'):
                    duration = time.time() - item['start_time']
                    st.write(f"⏱️ {duration:.0f}秒")
                elif item['status'] == 'completed' and item.get('end_time') and item.get('start_time'):
                    duration = item['end_time'] - item['start_time']
                    st.write(f"✅ {duration:.1f}秒")
                else:
                    st.write(f"📊 {item['status']}")
                    
            with col3:
                if item['status'] == 'completed':
                    if st.button("📊 查看报告", key=f"report_{item['id']}", use_container_width=True):
                        with st.expander(f"📊 {item['script_name']} 运行报告", expanded=True):
                            show_run_report(item)
                elif item['status'] in ['running', 'starting'] and item.get('pid'):
                    st.write(f"PID: {item['pid']}")
                    
            with col4:
                if st.button("❌", key=f"remove_{item['id']}", help="移除/终止"):
                    if item.get('pid') and item['status'] == 'running':
                        if kill_process(item['pid']):
                            st.success(f"✅ 已终止进程 {item['pid']}")
                        else:
                            st.warning("⚠️ 终止进程失败")
                    remove_from_queue(item['id'])
                    st.success(f"✅ 已从队列移除: {item['script_name']}")
                    time.sleep(0.5)  # 短暂延迟确保状态更新
                    st.rerun()
            
            st.markdown("---")

def show_script(script_file: Path):
    """显示单个脚本（紧凑模式，支持互斥展开）"""
    script_key = str(script_file.absolute())
    
    # 初始化session state
    if 'current_script' not in st.session_state:
        st.session_state.current_script = None
    
    runner = StreamlitScriptRunner(script_file)
    
    # 脚本标题行（紧凑布局，垂直居中）
    col1, col2 = st.columns([5, 1])
    with col1:
        script_name = script_file.stem.replace('_', ' ').title()
        
        # 检查是否是当前展开的脚本
        is_current = st.session_state.current_script == script_key
        button_text = f"{'🔽' if is_current else '▶️'} 🐍 {script_name}"
        
        # 使用水平布局，确保垂直居中
        button_col, desc_col = st.columns([2, 3])
        with button_col:
            # 使用container确保对齐
            with st.container():
                # 添加缩进
                st.markdown('<div style="display: inline-block; width: 20px;"></div>', 
                           unsafe_allow_html=True)
                if st.button(button_text, key=f"toggle_{script_key}"):
                    # 切换脚本展开状态
                    if st.session_state.current_script == script_key:
                        st.session_state.current_script = None
                    else:
                        st.session_state.current_script = script_key
                        st.rerun()
        
        with desc_col:
            # 描述在按钮右侧，使用相同高度的容器
            with st.container():
                content = runner.header.get('content', '')
                if content and len(content) > 0:
                    st.markdown(f'<div style="display: flex; align-items: center; height: 28px; font-size: 11px; color: #666;">'
                               f'💬 {content[:80]}{"..." if len(content) > 80 else ""}</div>', 
                               unsafe_allow_html=True)
    
    with col2:
        # 显示脚本状态
        if runner.header.get('disabled', False):
            st.markdown("🚫")
        else:
            st.markdown("✅")
    
    # 只有当前脚本才显示详细内容
    if st.session_state.current_script == script_key:
        st.markdown("---")
        runner.render_details()

def main():
    """主函数"""
    st.set_page_config(
        page_title="脚本运行器",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 添加紧凑的CSS样式
    st.markdown("""
    <style>
    .stButton > button {
        height: 28px;
        padding: 1px 6px;
        font-size: 12px;
        margin-bottom: 1px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stSelectbox > div > div {
        height: 28px;
        font-size: 12px;
    }
    .stTextInput > div > div > input {
        height: 28px;
        font-size: 12px;
    }
    .stNumberInput > div > div > input {
        height: 28px;
        font-size: 12px;
    }
    .element-container {
        margin-bottom: 2px;
        display: flex;
        align-items: center;
    }
    .stMarkdown {
        margin-bottom: 1px;
        line-height: 1.1;
        display: flex;
        align-items: center;
    }
    .stMarkdown p {
        margin-top: 0px;
        margin-bottom: 1px;
        line-height: 1.1;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 4px;
        border-radius: 3px;
        margin: 0px;
    }
    .stContainer {
        padding-top: 0px;
        padding-bottom: 0px;
    }
    div[data-testid="column"] {
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 项目脚本运行器")
    
    # 显示运行队列
    show_run_queue()
    
    st.markdown("---")
    
    # 侧边栏信息
    with st.sidebar:
        st.markdown("### 📋 使用说明")
        st.markdown("""
        1. 📁 文件夹已自动展开，无需点击
        2. ▶️ 点击脚本按钮展开参数设置
        3. 📝 填写必要参数后点击运行
        4. 📊 查看运行报告和生成的文件
        5. 👁️ 可预览生成的HTML/PDF文件
        """)
        
        st.markdown("### 🔧 系统信息")
        st.info(f"**操作系统:** {platform.system()}")
        st.info(f"**内存:** {psutil.virtual_memory().percent:.1f}%")
        st.info(f"**CPU:** {psutil.cpu_percent():.1f}%")
        
        st.markdown("### 🎯 新功能")
        st.success("✅ 文件夹默认展开")
        st.success("✅ 紧凑界面设计") 
        st.success("✅ 脚本互斥展开")
        st.success("✅ 运行报告生成")
        st.success("✅ 文件预览功能")
        
        st.markdown("### 🐛 调试信息")
        if st.button("查看队列文件", key="debug_queue"):
            queue_file = "run_queue.json"
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    queue_content = f.read()
                st.code(queue_content, language='json')
            else:
                st.info("队列文件不存在")
    
    # 主内容区域
    src_runs_path = Path("src_runs")
    if src_runs_path.exists():
        show_folder(src_runs_path)
    else:
        st.error("src_runs 文件夹不存在！")

if __name__ == '__main__':
    main() 