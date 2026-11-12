"""Interactive, existing-results-only model comparison."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

__all__ = ['main', 'calculate']


@st.cache_data(max_entries=8, show_spinner=False)
def calculate(sources, parameters):
    from src.res.model.analytic.compare import CompareConfig, CompareModelSpec, compare_models
    names = [Path(path).parent.parent.name if Path(path).name == 'detailed_alpha_data.xlsx' else Path(path).stem for path, _, _ in sources]
    labels = [name if names.count(name) == 1 else f'{name} ({i + 1})' for i, name in enumerate(names)]
    return compare_models([CompareModelSpec(path, name=name) for (path, _, _), name in zip(sources, labels)],
                          config=CompareConfig(**parameters), display=False, generate_figures=False, export=False)


def main():
    from src.proj import PATH
    from src.res.model.analytic.compare import CompareConfig
    st.set_page_config(page_title='Model Compare', layout='wide')
    zh = st.selectbox('Language', ['中文', 'English']) == '中文'
    def t(chinese, english):
        return chinese if zh else english
    st.title(t('模型对比', 'Model Compare'))
    candidates = sorted(PATH.model.rglob('detailed_alpha_data.xlsx'))
    with st.form('comparison'):
        selected = st.multiselect(t('选择至少两个模型', 'Select at least two models'), candidates,
                                  format_func=lambda p: str(p.parent.parent.relative_to(PATH.model)))
        extra = st.text_area(t('其他服务器报告或模型目录（每行一个）', 'Other server report/model paths (one per line)'))
        left, right = st.columns(2)
        start = left.text_input(t('开始日期 YYYYMMDD（可空）', 'Start YYYYMMDD (optional)'))
        end = right.text_input(t('结束日期 YYYYMMDD（可空）', 'End YYYYMMDD (optional)'))
        periods = st.multiselect(t('分期', 'Periods'), ['all', 'year', 'recent_year', 'quarter', 'month'], ['all', 'year', 'recent_year'])
        templates = st.multiselect(t('分析模板', 'Templates'), ['ic', 'top', 'complementarity'], ['ic', 'top', 'complementarity'])
        rolling = st.number_input(t('滚动窗口', 'Rolling window'), min_value=1, value=20)
        method = st.selectbox(t('相关性方法', 'Correlation method'), ['pearson', 'spearman'])
        config = asdict(CompareConfig())
        with st.expander(t('高级设置', 'Advanced settings')):
            custom_periods = st.text_area(t('自定义分期 JSON', 'Custom periods JSON'), value='{}')
            config['top_n'] = st.number_input('top_n', min_value=1, value=50)
            for key in ('submodel', 'ic_benchmark', 'top_sheet', 'top_strategy', 'top_benchmark', 'top_suffix'):
                config[key] = st.text_input(key, value=config[key])
        submitted = st.form_submit_button(t('比较 / 重新读取', 'Compare / reread'))
    if submitted:
        try:
            paths = [*selected, *[Path(line.strip()).expanduser() for line in extra.splitlines() if line.strip()]]
            paths = list(dict.fromkeys((p / 'results/detailed_alpha_data.xlsx' if p.is_dir() else p).resolve() for p in paths))
            if len(paths) < 2:
                raise ValueError(t('请至少选择两个不同模型。', 'Select at least two distinct models.'))
            config['custom_periods'] = json.loads(custom_periods)
            if not isinstance(config['custom_periods'], dict):
                raise ValueError('Custom periods must be an object')
            config.update(start=int(start) if start else None, end=int(end) if end else None,
                          periods=tuple(periods), templates=tuple(templates), rolling_window=rolling, corr_method=method,
                          analyze_pred=False, analyze_hidden=False)
            CompareConfig(**config).validate()
            with st.spinner(t('读取并比较已有结果…', 'Comparing existing results…')):
                sources = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
                calculate.clear()
                st.session_state['comparison_result'] = calculate(sources, config)
                st.session_state.pop('exports', None)
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.get('comparison_result')
    if result is None:
        return
    st.dataframe(result.summary, hide_index=True)
    for key in ('ic_series', 'top_series'):
        frame = result.tables.get(key)
        if frame is None or frame.empty:
            continue
        st.subheader(t('IC 曲线', 'IC curves') if key == 'ic_series' else t('收益与回撤', 'Returns and drawdown'))
        available = [c for c in ('cum_ic', 'ic', 'rolling_ic', 'cum_pf', 'cum_bm', 'cum_excess', 'excess_drawdown') if c in frame]
        metric = st.selectbox(t('指标', 'Metric'), available, key=key)
        chart = frame.copy()
        chart['date'] = pd.to_datetime(chart['date'].astype(str), format='%Y%m%d')
        if key == 'ic_series' and metric == 'cum_ic':
            origin = chart['date'].min() - pd.Timedelta(days=1)
            baseline = pd.DataFrame({'date': origin, 'model': chart['model'].unique(), 'cum_ic': 0.0})
            chart = pd.concat([baseline, chart], ignore_index=True).sort_values('date')
        st.line_chart(chart, x='date', y=metric, color='model')
    with st.expander(t('分期指标', 'Period metrics')):
        for key, frame in result.tables.items():
            if key.endswith('_periods'):
                st.write(key)
                st.dataframe(frame, hide_index=True)
    if result.correlations:
        key = st.selectbox(t('相关性矩阵', 'Correlation matrix'), [k for k in result.correlations if not k.endswith('_count')])
        if key is None:
            return
        matrix = result.correlations[key]
        cells = matrix.rename_axis('model_a').reset_index().melt(id_vars='model_a', var_name='model_b', value_name='correlation')
        st.altair_chart(alt.Chart(cells).mark_rect().encode(x='model_a:N', y='model_b:N', color=alt.Color('correlation:Q', scale=alt.Scale(domain=[-1, 1], scheme='redblue')), tooltip=['model_a', 'model_b', 'correlation']), width='stretch')
        with st.expander(t('配对样本数', 'Pairwise sample counts')):
            st.dataframe(result.correlations.get(key + '_count'))
    with st.expander(t('数据诊断与计算口径', 'Diagnostics and methodology')):
        st.dataframe(result.diagnostics)
        st.json(result.metadata)
    if st.button(t('生成 Excel / PDF', 'Generate Excel / PDF')):
        from src.res.model.analytic.compare.report import build_figures
        try:
            with st.spinner(t('生成报告…', 'Exporting…')):
                build_figures(result)
                result.export()
                st.session_state['exports'] = dict(result.output_paths)
                import matplotlib.pyplot as plt
                for figure in result.figures.values():
                    plt.close(figure)
                result.figures.clear()
        except Exception as exc:
            st.error(str(exc))
    for kind, path in st.session_state.get('exports', {}).items():
        st.download_button(kind, path.read_bytes(), file_name=path.name)


if __name__ == '__main__':
    main()
