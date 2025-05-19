# env: scienv

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import tensorflow_decision_forests as tfdf
import tensorflow as tf
import functools

import ydf
import time
import psutil
from threading import Thread, Event


class ResourceMonitor(Thread):
    """后台资源监控线程"""
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = Event()
        self.max_cpu = 0.0
        # self.max_gpu = 0.0

    def run(self):
        while not self.stop_event.wait(self.interval):
            self.max_cpu = max(self.max_cpu, psutil.cpu_percent())

    def stop(self):
        self.stop_event.set()

# 资源跟踪装饰器
def resource_monitor(model_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 启动监控线程
            monitor = ResourceMonitor()
            monitor.start()
            
            process = psutil.Process()
            start_mem = process.memory_info().rss / 1024**2  # MB
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_mem = process.memory_info().rss / 1024**2
            monitor.stop()
            monitor.join()

            mem = end_mem - start_mem
            if mem < 0:
                mem = -mem
            performance_data[model_name]['time'].append(end_time - start_time)
            performance_data[model_name]['memory'].append(mem)
            performance_data[model_name]['cpu'].append(monitor.max_cpu)
            # performance_data[model_name]['gpu'].append(monitor.max_gpu)
            
            print(f"[{model_name.upper()}] Time: {end_time-start_time:.2f}s | "
                  f"Mem: {end_mem-start_mem:.1f}MB | "
                  f"CPU: {monitor.max_cpu:.1f}% "
                  )
            return result
        return wrapper
    return decorator

def prepare_data(df, crop_type: str, target_col='Yield') -> pd.DataFrame:
    if crop_type != 'threecrops':
        filtered = df[df['Crop.type'] == crop_type]
        cols = {
            'Maize': ["Yield", "Slope", "Irrigation", "Cropping.system.in.the.site", 
                     "Rotational.systems", "Name.of.previous.crop", "Growing.days", 
                     "GDD", "Tmax", "Tmin", "PRE", "RAD", "Soil.type", "SOC", "OP", 
                     "AK", "PH", "PK.fert", "N.fert", "SAND", "SILT", "CLAY", 
                     "AWC", "Bulk.density", "Crop.variety", "AI", "PET10"],
            'Wheat': ["Yield", "Slope", "Irrigation", "Cropping.system.in.the.site",
                     "Rotational.systems", "Name.of.previous.crop", "Growing.days",
                     "GDD", "Tmax", "Tmin", "PRE", "RAD", "Soil.type", "SOC", "OP",
                     "AK", "PH", "PK.fert", "N.fert", "SAND", "SILT", "CLAY",
                     "AWC", "Bulk.density", "Crop.variety", "AI", "PET10"],
            'Rice': ["Yield", "Slope", "Cropping.system.in.the.site", 
                    "Rotational.systems", "Growing.days", "GDD", "Tmax", "Tmin", 
                    "PRE", "RAD", "Soil.type", "SOC", "OP", "AK", "PH", "PK.fert", 
                    "N.fert", "SAND", "SILT", "CLAY", "AWC", "Bulk.density", 
                    "Crop.variety", "AI", "PET10"]
        }
        return filtered[cols[crop_type]]
    else:
        cols = ["Yield.normalized", "Slope", "Irrigation", "Cropping.system.in.the.site",
               "Rotational.systems", "Name.of.previous.crop", "Growing.days", "GDD",
               "Tmax", "Tmin", "PRE", "RAD", "Soil.type", "SOC", "OP", "AK", "PH",
               "PK.fert", "N.fert", "SAND", "SILT", "CLAY", "AWC", "Bulk.density",
               "Crop.variety", "AI", "PET10"]
        return df[cols]

# sklearn 模型需要One-hot预处理
def onehot_encoder(crop_type: pd.DataFrame):

    factor = crop_type.select_dtypes(include='object').columns.tolist()
    for _ in factor:
        one_hot = pd.get_dummies(crop_type[_], prefix=_, dtype=int)
        crop_type = pd.concat([crop_type, one_hot], axis=1)

    crop_type = crop_type.drop(columns=factor)
    return crop_type

def onehot_encoder_enhanced(df, feature_columns=None):
    """支持特征列对齐的编码器"""
    df = df.copy()
    
    # 识别分类特征
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    # 执行one-hot编码
    for col in categorical_cols:
        one_hot = pd.get_dummies(df[col], prefix=col, dtype=int)
        df = pd.concat([df, one_hot], axis=1)
        df = df.drop(col, axis=1)

    # 对齐特征列
    if feature_columns is not None:
        # 添加缺失列
        missing_cols = set(feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        # 过滤多余列
        df = df[feature_columns]
        
    return df


# 数据集分割
def split_dataset(dataset: pd.DataFrame, test_ratio: float=0.30):
  
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

# 建模函数
# sklearn库: RandomForestRegressor
@resource_monitor('sklearn')
def rf_sklearn_build_model(train_data, label='Yield', is_test=True):
    if is_test:
        train_data = onehot_encoder(train_data)
    X = train_data.drop(columns=[label])
    y = train_data[label]
    
    rf_model = RandomForestRegressor(n_estimators=500,
                                max_features=5,
                                oob_score=True,
                                random_state=123)
    rf_model.fit(X, y)
    
    return rf_model

# tensorflow库: tfdf
@resource_monitor('tensorflow')
def rf_tensorflow_build_model(train_data, label='Yield'):
    # 将数据集转换为tf_dataset格式
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_data, label=label, task=tfdf.keras.Task.REGRESSION)
    # test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

    rf_model = tfdf.keras.RandomForestModel(verbose=2,
                                            hyperparameter_template="benchmark_rank1",
                                            num_trees=500, 
                                            task=tfdf.keras.Task.REGRESSION,
                                            compute_oob_performances=True,
                                            compute_oob_variable_importances=True,
                                            random_seed=123,
                                            )

    rf_model.fit(train_ds)
    return rf_model

# ydf库
@resource_monitor('ydf')
def rf_ydf_build_model(train_data, label='Yield'):

    rf_model = ydf.RandomForestLearner(label=label,
                                   task=ydf.Task.REGRESSION,
                                   num_trees=500,
                                   
                                ).train(train_data)
    
    return rf_model

def evaluate_sklearn_model(test_data, model, label='Yield') -> float:

    test_data = onehot_encoder(test_data)
    test_X = test_data.drop(columns=[label])
    true_y = test_data[label]

    pred_y = model.predict(test_X)

    rmse = np.sqrt(mean_squared_error(true_y, pred_y))
    performance_data['sklearn']['rmse'].append(rmse)
    return rmse

def evaluate_tfdf_model(test_data, model, label='Yield') -> float:
    true_y = test_data[label]
    test_X = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data.drop('Yield', axis=1), 
        task=tfdf.keras.Task.REGRESSION
    )

    
    y_pred = model.predict(test_X).flatten()
    rmse = np.sqrt(mean_squared_error(true_y, y_pred))
    performance_data['tensorflow']['rmse'].append(rmse)
    return rmse

def evaluate_ydf_model(test_data, model) -> float:
    evaluation = model.evaluate(test_data)
    performance_data['ydf']['rmse'].append(evaluation.rmse)
    return evaluation.rmse


def train_all_model(dataset: pd.DataFrame, times: int=3, is_test: bool = True, onehot_dataset=None) -> None:
        
    if is_test:
        train_ds_pd, _ = split_dataset(dataset=dataset)
        for _ in range(times):
            print(f'Train Model in : {_} times.')
            # 训练sklearn模型
            rf_sklearn_model = rf_sklearn_build_model(train_data=train_ds_pd)
            
            # 训练TensorFlow模型
            rf_tfdf_model = rf_tensorflow_build_model(train_data=train_ds_pd)
            
            # 训练ydf模型
            rf_ydf_model = rf_ydf_build_model(train_data=train_ds_pd)
    else:
        rf_sklearn_model = rf_sklearn_build_model(train_data=onehot_dataset, is_test=False)
        
        # 训练TensorFlow模型
        rf_tfdf_model = rf_tensorflow_build_model(train_data=dataset)
        
        # 训练ydf模型
        rf_ydf_model = rf_ydf_build_model(train_data=dataset)

    return rf_sklearn_model, rf_tfdf_model, rf_ydf_model

def evaluate_all_model(dataset: pd.DataFrame):
    train_ds_pd, test_ds_pd = split_dataset(dataset=dataset)
    train_oh, test_oh = split_dataset(onehot_encoder(dataset))
    rf_sklearn_model, rf_tfdf_model, rf_ydf_model = train_all_model(dataset=train_ds_pd, is_test=False, onehot_dataset=train_oh)

    rf_sklearn_rmse = evaluate_sklearn_model(test_data=test_oh, model=rf_sklearn_model)
    rf_tfdf_rmse = evaluate_tfdf_model(test_data=test_ds_pd, model=rf_tfdf_model)
    rf_ydf_rmse = evaluate_ydf_model(test_data=test_ds_pd, model=rf_ydf_model)

    return [rf_sklearn_rmse, rf_tfdf_rmse, rf_ydf_rmse]


def plot_performance_comparison(times: int=3):
    frameworks = ['sklearn', 'tensorflow', 'ydf']
    metrics = ['time', 'memory', 'cpu', 'rmse']
    titles = ['Training Time', 'Memory Usage', 'Max CPU Occupied', 'RMSE']
    units = ['Seconds', 'MB', '%', '']

    final_results = {}
    for framework in performance_data:
        final_results[framework] = {
            'avg_time': np.mean(performance_data[framework]['time']),
            'avg_memory': np.mean(performance_data[framework]['memory']),
            'avg_cpu': np.mean(performance_data[framework]['cpu']),
            'avg_rmse': np.mean(performance_data[framework]['rmse'])
        }

    # plt.title(f'Run in {times} times.')
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        bar_width = 0.8 / len(frameworks)
        positions = np.arange(len(frameworks))
        for j, fw in enumerate(frameworks):
            data = final_results[fw][f'avg_{metric}']
            plt.bar(positions[j], data, width=bar_width, label=fw)
        plt.title(titles[i])
        plt.ylabel(units[i])
        plt.grid(axis='y')

    print(final_results) 
    plt.legend()
    plt.tight_layout()
    plt.savefig('性能对比.png', dpi=200)
    plt.show()


if __name__ == '__main__':

    performance_data = {
    'sklearn': {'time': [], 'memory': [], 'cpu': [], 'rmse': []},
    'tensorflow': {'time': [], 'memory': [], 'cpu': [], 'rmse': []},
    'ydf': {'time': [], 'memory': [], 'cpu': [], 'rmse': []}
    }

    data = pd.read_csv('data.csv', index_col="Coden")

    # 读取数据
    Maize = prepare_data(data, 'Maize')
    Maize['Yield'] = Maize['Yield'] / 1000

    Wheat = prepare_data(data, 'Wheat')
    Rice = prepare_data(data, 'Rice')
    threecrops = prepare_data(data, 'threecrops')

    times = 3
    # train_all_model(dataset=Maize, times=times, is_test=True)
    evaluate_all_model(dataset=Maize)
    plot_performance_comparison(times=times)