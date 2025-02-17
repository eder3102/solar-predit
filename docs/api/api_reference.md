# 光伏发电预测系统 API 接口文档

## API 概述
本文档详细说明了光伏发电预测系统的所有API接口。系统采用RESTful风格设计,所有接口都支持JSON格式数据交互。

## 接口规范

### 1. 请求格式
```json
POST /api/v1/predict
Content-Type: application/json
{
    "timestamp": "2024-02-14T10:00:00Z",
    "features": {
        "temperature": [25.3, 25.1, ...],  // 96点温度数据
        "humidity": [60, 62, ...],         // 96点湿度数据
        "irradiance": [0, 0.1, ...],       // 96点辐照度数据
        ...  // 其他特征
    }
}
```

### 2. 响应格式
```json
{
    "code": 200,
    "message": "success",
    "data": {
        "predictions": [0.0, 0.1, ...],  // 96点预测功率
        "confidence": [0.95, 0.94, ...], // 预测置信度
        "model_version": "v1.0.2"        // 模型版本
    }
}
```

### 3. 错误码说明
| 错误码 | 说明 | 处理建议 |
|--------|------|----------|
| 200    | 成功 | - |
| 400    | 请求参数错误 | 检查输入数据格式 |
| 401    | 未授权 | 检查认证信息 |
| 500    | 服务器错误 | 联系运维人员 |
| 503    | 服务不可用 | 检查服务器状态 |

## API 接口列表

### 1. 预测接口

#### 1.1 实时预测
- 接口: `/api/v1/predict/realtime`
- 方法: POST
- 描述: 获取未来96点(24小时)的发电量预测
- 参数:
  ```json
  {
      "timestamp": "2024-02-14T10:00:00Z",
      "features": {
          "temperature": [25.3, 25.1, ...],  // 96点温度数据
          "humidity": [60, 62, ...],         // 96点湿度数据
          "irradiance": [0, 0.1, ...],       // 96点辐照度数据
          ...  // 其他特征
      }
  }
  ```
- 响应:
  ```json
  {
      "code": 200,
      "message": "success",
      "data": {
          "predictions": [0.0, 0.1, ...],  // 96点预测功率
          "confidence": [0.95, 0.94, ...], // 预测置信度
          "model_version": "v1.0.2"        // 模型版本
      }
  }
  ```

#### 1.2 批量预测
- 接口: `/api/v1/predict/batch`
- 方法: POST
- 描述: 批量获取多个时间点的预测结果
- 参数:
  ```json
  {
      "batch_data": [
          {
              "timestamp": "2024-02-14T10:00:00Z",
              "features": {...}
          },
          {
              "timestamp": "2024-02-14T11:00:00Z",
              "features": {...}
          }
      ]
  }
  ```
- 响应:
  ```json
  {
      "code": 200,
      "message": "success",
      "data": [
          {
              "timestamp": "2024-02-14T10:00:00Z",
              "predictions": [...],
              "confidence": [...]
          },
          {
              "timestamp": "2024-02-14T11:00:00Z",
              "predictions": [...],
              "confidence": [...]
          }
      ]
  }
  ```

### 2. 模型管理接口

#### 2.1 模型状态查询
- 接口: `/api/v1/model/status`
- 方法: GET
- 描述: 获取当前模型状态
- 响应:
  ```json
  {
      "code": 200,
      "message": "success",
      "data": {
          "model_version": "v1.0.2",
          "last_updated": "2024-02-14T10:00:00Z",
          "performance_metrics": {
              "mae": 0.15,
              "rmse": 0.2,
              "r2": 0.9
          },
          "memory_usage": "2.8GB",
          "avg_latency": "42ms"
      }
  }
  ```

#### 2.2 模型更新触发
- 接口: `/api/v1/model/update`
- 方法: POST
- 描述: 触发模型更新
- 参数:
  ```json
  {
      "update_type": "incremental",  // 或 "full"
      "training_data_path": "/path/to/data"
  }
  ```
- 响应:
  ```json
  {
      "code": 200,
      "message": "success",
      "data": {
          "task_id": "update_20240214100000",
          "status": "started",
          "estimated_time": "2h"
      }
  }
  ```

### 3. 监控接口

#### 3.1 性能监控
- 接口: `/api/v1/monitor/performance`
- 方法: GET
- 描述: 获取系统性能指标
- 响应:
  ```json
  {
      "code": 200,
      "message": "success",
      "data": {
          "cpu_usage": "45%",
          "memory_usage": "2.8GB",
          "gpu_usage": "0%",
          "request_count": 1000,
          "avg_latency": "42ms",
          "error_rate": "0.1%"
      }
  }
  ```

#### 3.2 预测质量监控
- 接口: `/api/v1/monitor/quality`
- 方法: GET
- 描述: 获取预测质量指标
- 响应:
  ```json
  {
      "code": 200,
      "message": "success",
      "data": {
          "last_24h": {
              "mae": 0.15,
              "rmse": 0.2,
              "r2": 0.9
          },
          "last_7d": {
              "mae": 0.14,
              "rmse": 0.19,
              "r2": 0.91
          }
      }
  }
  ```

## 调用示例

### Python示例
```python
import requests
import json

def predict_power():
    url = "http://api.example.com/api/v1/predict/realtime"
    data = {
        "timestamp": "2024-02-14T10:00:00Z",
        "features": {
            "temperature": [25.3, 25.1, ...],
            "humidity": [60, 62, ...],
            "irradiance": [0, 0.1, ...]
        }
    }
    
    response = requests.post(url, json=data)
    return response.json()

# 调用示例
result = predict_power()
print(json.dumps(result, indent=2))
```

### curl示例
```bash
curl -X POST "http://api.example.com/api/v1/predict/realtime" \
     -H "Content-Type: application/json" \
     -d '{
         "timestamp": "2024-02-14T10:00:00Z",
         "features": {
             "temperature": [25.3, 25.1],
             "humidity": [60, 62],
             "irradiance": [0, 0.1]
         }
     }'
```

## 注意事项
1. 所有时间戳使用UTC时间
2. 特征数据必须是96点完整数据
3. 接口调用频率限制为100次/分钟
4. 批量接口单次最多支持10个预测点
5. 建议在请求中附带API版本号 