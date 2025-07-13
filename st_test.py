import streamlit as st

def initialize_session_state():
    """初始化 session_state 中的筛选状态"""
    if 'selected_types' not in st.session_state:
        st.session_state.selected_types = []
    if 'selected_names' not in st.session_state:
        st.session_state.selected_names = []

def get_unique_types(data):
    """从数据中获取所有唯一的 type 值"""
    return sorted(list({item['type'] for item in data}))

def get_names_by_types(data, selected_types):
    """根据已选的 types 获取可选的 names"""
    if not selected_types:
        return []
    return sorted(list({item['name'] for item in data if item['type'] in selected_types}))

def render_filters(data):
    """渲染两级筛选器 UI"""
    # 获取所有唯一的 type 选项
    all_types = get_unique_types(data)
    
    # Type 多选筛选器
    selected_types = st.multiselect(
        "选择 Type:",
        options=all_types,
        default=st.session_state.selected_types,
        key='type_selector'
    )
    
    # 更新 session_state 中的 selected_types
    st.session_state.selected_types = selected_types
    
    # 基于已选 type 获取可选的 names
    available_names = get_names_by_types(data, selected_types)
    
    # Name 多选筛选器 (仅在选择了 type 后显示)
    selected_names = st.multiselect(
            "选择 Name:",
            options=available_names,
            default=st.session_state.selected_names,
            key='name_selector',
            on_change=lambda: st.write(st.session_state)
        )

def main():
    st.write(f"Hello {st.session_state}")
    st.title("列表筛选器")
    
    # 假设这是你的自定义列表数据
    # 实际应用中替换为你的真实数据
    custom_list = [
        {"type": "水果", "name": "苹果"},
        {"type": "水果", "name": "香蕉"},
        {"type": "蔬菜", "name": "胡萝卜"},
        {"type": "蔬菜", "name": "菠菜"},
        {"type": "肉类", "name": "牛肉"},
        {"type": "肉类", "name": "鸡肉"}
    ]
    
    # 初始化 session_state
    initialize_session_state()
    
    # 渲染筛选器 UI
    render_filters(custom_list)
    
    # 这里可以添加你的列表展示逻辑
    # filtered_list = apply_filters(custom_list) 
    # display_list(filtered_list)

if __name__ == "__main__":
    main()