import streamlit_antd_components as sac

sac.buttons([
    sac.ButtonsItem(label='button'),
    sac.ButtonsItem(icon='apple'),
    sac.ButtonsItem(label='google', icon='google', color='#25C3B0'),
    sac.ButtonsItem(label='wechat', icon='wechat'),
    sac.ButtonsItem(label='disabled', disabled=True),
    sac.ButtonsItem(label='link', icon='share-fill', href='https://ant.design/components/button'),
], label='label', align='center')