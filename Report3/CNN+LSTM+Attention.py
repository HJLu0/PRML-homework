import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

train = pd.read_csv('LSTM-Multivariate_pollution.csv')
test = pd.read_csv('pollution_test_data1.csv')

# 数据预处理
def preprocess(df):
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
        df.set_index('datetime', inplace=True)
        df.drop('date', axis=1, inplace=True)

    if 'No' in df.columns:
        df.drop(['No'], axis=1, inplace=True)

    if 'cbwd' in df.columns:
        df = pd.get_dummies(df, columns=['cbwd'])

    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)

    return df
train = preprocess(train)
test = preprocess(test)
train, test = train.align(test, join='left', axis=1, fill_value=0)

# 画热力图
plt.figure(figsize=(10,8))
corr = train.select_dtypes(include=[np.number]).corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    cbar=True
)

plt.title("Feature Correlation Heatmap", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# 构造时间序列数据
def create_dataset(data, n_steps=24, n_out=3):
    X, y = [], []
    for i in range(len(data)-n_steps-n_out):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps:i+n_steps+n_out, 0])
    return np.array(X), np.array(y)

n_steps = 24
n_out = 3

X_train, y_train = create_dataset(train_scaled, n_steps, n_out)
X_test, y_test = create_dataset(test_scaled, n_steps, n_out)

# Attention层
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# 模型构建
input_layer = Input(shape=(n_steps, X_train.shape[2]))

x = Conv1D(64, 3, activation='relu')(input_layer)
x = MaxPooling1D(2)(x)
x = LSTM(64, return_sequences=True)(x)
x = AttentionLayer()(x)

output = Dense(n_out)(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse')

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, y_test)
)

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Training Loss Curve")
plt.show()

y_pred = model.predict(X_test)

def invert_scale(y, scaler, n_features):
    temp = np.zeros((y.shape[0], n_features))
    temp[:,0] = y
    return scaler.inverse_transform(temp)[:,0]
# 单步预测
y_test_inv = invert_scale(y_test[:,0], scaler, train.shape[1])
y_pred_inv = invert_scale(y_pred[:,0], scaler, train.shape[1])

plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label='True')
plt.plot(y_pred_inv, label='Predicted')
plt.legend()
plt.title("PM2.5 Prediction (t+1)")
plt.show()

# 多步预测
steps = ['t+1', 't+2', 't+3']

plt.figure(figsize=(12,6))

for i in range(n_out):
    y_test_i = invert_scale(y_test[:,i], scaler, train.shape[1])
    y_pred_i = invert_scale(y_pred[:,i], scaler, train.shape[1])

    plt.subplot(n_out,1,i+1)
    plt.plot(y_test_i, label='True')
    plt.plot(y_pred_i, label='Predicted')
    plt.title(f"Step {steps[i]} Prediction")
    plt.legend()

plt.tight_layout()
plt.show()
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print("RMSE:", rmse)