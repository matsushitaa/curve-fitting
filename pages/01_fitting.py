"""
streamlit run Hello.py --server.enableCORS false --server.enableXsrfProtection false
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
import ast
import plotly.graph_objects as go
import inspect
from bayes_opt import BayesianOptimization
from enum import Enum, auto


# 'data.csv' ファイルの相対パス
file_path = os.path.join(os.path.dirname(__file__), 'data.csv')


class Status(Enum):
    PROCESSING = auto()
    STOPPED = auto()
    SUCCESS = auto()
    ABORTED = auto()


@st.cache_data
def read_csv_file(uploaded_file):
    return pd.read_csv(uploaded_file)


def get_function_arguments(func):
    argspec = inspect.getfullargspec(func)
    return argspec.args


def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


circle_function_text = """def fit_function(x, a, b, c, d):
    x = np.where(x<b-c, 0, x)
    x = np.where(x>b+c, 0, x)
    return np.sqrt(np.abs(a*(x-b)**2-c**2))+d"""

combined_function_text = """# Combination of Sigmoid and Gaussian Functions
def fit_function(x, a1, b1, c1, d1, a2, b2, c2, d2):
    def sigmoid(x, a, b, c, d):
        return a / (b + np.exp(-c * (x - d)))
    def gaussian(x, a, b, c, d):
        return a * np.exp(-((x - b)**2) / (2 * c**2)) + d
    return sigmoid(x, a1, b1, c1, d1) + gaussian(x, a2, b2, c2, d2)
"""


def reset_initial_params():
    st.session_state.optimizer = None
    st.session_state.initial_params = []


def get_fit_function(st, st2, len_data):
    """フィッティング関数取得
    Returns:
        function: フィッティング関数
    """
    col11,col12 = st2.columns(2)
    # 選択された近似式のテキストを定義
    fitting_options = ('exponential', 'linear', 'logarithmic', 'polynomial', 'power', 'sin', 'tan', 'sigmoid', 'gaussian', 'circle', 'combined')
    selected_fitting_option = col11.selectbox("Select the approximation formula", fitting_options, index=0)
    if selected_fitting_option == 'exponential':
        function_text = """def fit_function(x, a, b, c):\n    return a * np.exp(b * x) + c"""
    elif selected_fitting_option == 'linear':
        function_text = """def fit_function(x, a, b):\n    return a * x + b"""
    elif selected_fitting_option == 'logarithmic':
        function_text = """def fit_function(x, a, b):\n    return a * np.log(x) + b"""
    elif selected_fitting_option == 'polynomial':
        order = col12.number_input('Order', value=2, min_value=0, max_value=len_data-1)
        function_text = f"def fit_function(x, {', '.join(f'a{i}' for i in range(order + 1))}):\n"
        function_text += f"    return {' + '.join(f'a{i} * x**{i}' for i in range(order + 1))}"
    elif selected_fitting_option == 'power':
        function_text = """def fit_function(x, a, b, c):\n    return a * x**b + c"""
    elif selected_fitting_option == 'sin':
        function_text = """def fit_function(x, a, b, c):\n    return a * np.sin(x*b) + c"""
    elif selected_fitting_option == 'tan':
        function_text = """def fit_function(x, a, b, c):\n    return a * np.tan(x*b) + c"""
    elif selected_fitting_option == 'sigmoid':
        function_text = "def fit_function(x, a, b, c, d, e):\n    return a / (b + np.exp(-c * (x - d))) + e"
    elif selected_fitting_option == 'gaussian':
        function_text = "def fit_function(x, a, b, c, d):\n    return a * np.exp(-((x - b)**2) / (2 * c**2)) + d"
    elif selected_fitting_option == 'circle':
        function_text = circle_function_text
    elif selected_fitting_option == 'combined':
        function_text = combined_function_text
    # 近似式表示 (編集可能)
    edited_function_text = st2.text_area("Approximation formula (editable):", function_text, 
                                        help='The function name is fit_function, and the first argument is x.', on_change=reset_initial_params)
    # 情報クリア
    if st.session_state.prev_fit_function!=edited_function_text:
        st.session_state.prev_fit_function = edited_function_text
        reset_initial_params()
    # 関数チェック
    try:
        namespace = {}
        exec(edited_function_text, globals(), namespace)
        fit_function = namespace.get('fit_function', None)
    except Exception as e:
        st2.error(f"Invalid function format: {e}")
        st.stop()
    return fit_function


def bayesian_optimization(x_data, y_data, fit_function, bounds, optimizer):
    """ベイズ最適化によるcurve_fitの初期値探索
    Args:
        x_data (np.array): xデータ
        y_data (np.array): yデータ
        fit_function (function): フィッティング関数
        bounds (dict): 探索範囲
        optimizer (Optional[BayesianOptimization]): ベイズ最適化インスタンス

    Returns:
        BayesianOptimization: ベイズ最適化インスタンス
    """
    def objective_function(**params):
        """最大化対象関数"""
        try:
            initial_params = list(params.values())
            _params, covariance = curve_fit(fit_function, x_data, y_data, p0=initial_params)
            R2 = calculate_r2(y_data, fit_function(x_data, *_params))
        except:
            R2 = -1
        return R2
    
    if optimizer is None:
        optimizer = BayesianOptimization(f=objective_function, pbounds=bounds, allow_duplicate_points=True)
        argument_names = get_function_arguments(fit_function)
        optimizer.maximize(init_points=len(argument_names), n_iter=0)
    optimizer.maximize(init_points=0, n_iter=1)

    best_r2 = optimizer.max['target']
    optimal_params = optimizer.max['params']
    optimal_params = [optimal_params[key] for key in sorted(optimal_params.keys())]
    return optimal_params, best_r2, optimizer


def optimize_initial_params(st, st2, x_data, y_data, fit_function):
    """初期値の最適化およびGUI更新
    Notes:
        最適化した初期値はst.session_state.initial_paramsで受け渡す
    """
    def run_opt():
        st.session_state.states = Status.PROCESSING
    def stop_opt(notification):
        st.session_state.states = Status.ABORTED

    # 探索範囲、R2閾値
    # with st2.container():
    argument_names = get_function_arguments(fit_function)
    bounds = {}
    cols = st2.columns(len(argument_names[1:]))
    for i, name in enumerate(argument_names[1:]):
        upper = cols[i].number_input(f'{name} upper',value=10.0)
        lower = cols[i].number_input(f'{name} lower',value=-10.0)
        bounds[name] = (lower, upper)
    col11, col12 = st2.columns(2)
    threshold = float(col11.text_input('R2 threshold',value=0.9))
    # 位置固定のためにwidgets定義
    button = st2.empty()
    notification = st2.empty()
    # GUI状態管理
    if st.session_state.states.value==Status.SUCCESS.value:
        st.session_state.states = Status.STOPPED
        notification.success('Exploration completed!')
    if st.session_state.states.value==Status.ABORTED.value:
        st.session_state.states = Status.STOPPED
        notification.warning('Exploration may not be effective...')
    if st.session_state.states.value==Status.STOPPED.value:
        if button.button('Run Optimization', on_click=run_opt):
            st.snow() # 押された瞬間に一番上に戻るためここには入らない
    if st.session_state.states.value==Status.PROCESSING.value:
        if button.button('Stop Optimization', on_click=stop_opt, args=(notification,)):
            st.snow() # 押された瞬間に一番上に戻るためここには入らない
    # 最適化
    if st.session_state.states.value==Status.PROCESSING.value:
        optimizer = st.session_state.optimizer
        n_iter = 0
        while True:
            n_iter += 1
            # ベイズ最適化によるcurve_fitの初期値探索
            optimal_params, best_r2, optimizer = bayesian_optimization(x_data, y_data, fit_function, bounds, optimizer)
            notification.info(f'{n_iter}: Best R2 = {best_r2}')
            # 初期値更新
            st.session_state.optimizer = optimizer
            st.session_state.initial_params = optimal_params
            # R2が閾値を超えたら終了
            if threshold < best_r2:
                st.session_state.states = Status.SUCCESS
                st.rerun()


def main():
    st.set_page_config(
        page_title="Curve Fitting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Curve Fitting")
    if 'prev_fit_function' not in st.session_state:
        st.session_state.prev_fit_function = None
    if 'initial_params' not in st.session_state:
        st.session_state.initial_params = []
    if 'states' not in st.session_state:
        st.session_state.states = Status.STOPPED
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None

    st.subheader("1. Input Data")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        df = pd.DataFrame({'x':[np.nan],'y':[np.nan]})
        # CSVファイルのアップロード
        input_mode = col1.radio('input mode', ('use sample data','upload'), horizontal=True)
        if input_mode=='upload':
            uploaded_file = col1.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                df = read_csv_file(uploaded_file)
        else:
            df = read_csv_file(file_path)
        # データの表示、編集
        df = col1.data_editor(df, height=300, use_container_width=True, num_rows='dynamic')
        # データの選択
        col21, col22 = col2.columns(2)
        selected_column = col21.selectbox("Select X-axis column", df.columns, index=0)
        x_data = df[selected_column]
        selected_column_y = col22.selectbox("Select Y-axis column", df.columns, index=1)
        y_data = df[selected_column_y]
        # データのプロット
        scatter_trace = go.Scatter(x=x_data, y=y_data, mode='markers', name="Data")
        layout_data = go.Layout(xaxis=dict(title=selected_column), yaxis=dict(title=selected_column_y))
        fig_data = go.Figure(data=[scatter_trace], layout=layout_data)
        col2.plotly_chart(fig_data, use_container_width=True)

    st.subheader("2. Curve Fitting")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        # 2-1. フィッティング関数取得
        fit_function = get_fit_function(st, col1, len(x_data))
        # 2-2. 初期値設定
        argument_names = get_function_arguments(fit_function)
        init_type = col1.radio('Initial Values',('Manual','Optimization'), horizontal=True, on_change=reset_initial_params)
        if init_type == 'Optimization':
            # 初期値の最適化
            optimize_initial_params(st, col1, x_data, y_data, fit_function)
        else:
            cols = col1.columns(len(argument_names[1:]))
            vals = []
            for i, name in enumerate(argument_names[1:]):
                val = cols[i].number_input(f'{name}', value=1.0)
                vals.append(val)
            st.session_state.initial_params = vals
        # 2-3. フィッティング & 可視化
        if (1 < len(x_data)) and (len(st.session_state.initial_params) != 0):
            try:
                # フィッティングの実行
                params, covariance = curve_fit(fit_function, x_data, y_data, p0=st.session_state.initial_params)
                # フィッティング結果の表示
                col2.write("Fitting Results:")
                col21, col22 = col2.columns(2)
                for name, param in zip(argument_names[1::2], params[::2]):
                    col21.write(f'{name} = {param}')
                for name, param in zip(argument_names[2::2], params[1::2]):
                    col22.write(f'{name} = {param}')
                r2_value = calculate_r2(y_data, fit_function(x_data, *params))
                col2.write(f'R2 = {r2_value}')
                # フィッティング曲線のプロット
                x_fit = np.linspace(min(x_data), max(x_data), 1000)
                y_fit = fit_function(x_fit, *params)
                fit_trace = go.Scatter(x=x_fit, y=y_fit, mode='lines', name="Fit", line=dict(color="red"))
                layout_fit = go.Layout(xaxis=dict(title=selected_column), yaxis=dict(title=selected_column_y))
                fig_fit = go.Figure(data=[scatter_trace, fit_trace], layout=layout_fit)
                col2.plotly_chart(fig_fit, use_container_width=True)
            except Exception as e:
                col1.error(f'{e}')


if __name__ == '__main__':
    main()
