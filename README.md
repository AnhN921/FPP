# DynamicFL
Dynamic Federated Learning

Code thực hiện chạy Federated Learning trên môi trường mạng nhiều thiết bị khác nhau. Sử dụng với framework `PyTorch` và giao thức IoT `MQTT`.

## Mô hình thực hiện

Trong ví dụ, mô hình LSTM được sử dụng để detect DGA (Domain Generation Algorithms).

Các thư viện python đảm bảo được cài đúng yêu cầu trong file `requirements.txt`.

Trong file `server.py` cần chỉnh sửa biến `NUM_ROUND` và `NUM_DEVICE` để phù hợp với số lượng round và thiết bị cần Train.

Trong file `server.py` và `glob_inc\client_fl.py` cần chỉnh sửa biến `broke_name` là địa chỉ ip của Mosquitto MQTT Broker trung gian.

### Cách cài đặt Mosquitto MQTT Broker trung gian trên linux:

Các bước chạy lệnh:
```commandline
sudo apt-add-repository ppa:mosquitto-dev/mosquitto-ppa

sudo apt-get update

sudo apt-get install mosquitto

sudo apt-get install mosquitto-clients

sudo apt clean
```

Cấu hình lại file config trong `/etc/mosquitto/mosquitto.conf` bằng cách thêm 2 trường sau:

```
listener 1883

allow_anonymous true
```
 #### Chú ý: check port kết nối (mặc định 1183) và đã cài Mosquitto MQTT Broker

### Câu lệnh chay:

Run sevrer:
```commandline
python server.py
```

Run client:
```commandline
python client.py id
```

Trong đó:

    id: id của client
# Anhne
# FPP
