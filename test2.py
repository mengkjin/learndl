import streamlit as st

def on_selectbox_change(key , additional_param):
    # 通过 session_state 获取当前选择的值
    current_value = st.session_state[key]
    st.write(f"Selected: {current_value}, Additional: {additional_param}")

# 使用functools.partial时只需要传递额外参数
from functools import partial
st.selectbox(
    "选择一个选项",
    ["选项1", "选项2", "选项3"],
    on_change=partial(on_selectbox_change, key = "my_selectbox-abc" , additional_param="extra info"),
    key="my_selectbox-abc"
)