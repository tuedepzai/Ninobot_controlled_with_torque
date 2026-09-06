# RL điều khiển mô-men hai bánh cho Nino

Package `nino_rl` cung cấp ba chương trình hoàn chỉnh:

- `train`: huấn luyện PPO trong Gazebo trên dãy gờ cáp với curriculum;
- `evaluate`: chạy test xác định và ghi chỉ số ra CSV/JSON;
- `policy_node`: chạy policy đã học, nhận đường `/plan` từ Nav2 và xuất mô-men trái/phải.

Mục tiêu là đi qua địa hình gồ ghề mà vẫn bám đường, hạn chế lật/kẹt, giảm thời gian và không đi ra ngoài hành lang. Policy sử dụng đồng thời IMU, encoder, LiDAR, odometry và các waypoint nhìn trước.

> **An toàn:** model mới khởi tạo hoặc chưa được đánh giá có thể phát mô-men bất ngờ. Chỉ chạy trong Gazebo cho đến khi đạt tiêu chí test. Khi thử trên robot thật phải kê bánh/giới hạn mô-men, có nút dừng khẩn cấp và người giám sát.

## Phần lấy từ hai bài báo và phần đã điều chỉnh

### Từ *Reinforcement Learning for Wheeled Mobility on Vertically Challenging Terrain*

- dùng PPO cho action liên tục;
- curriculum từ đoạn dễ đến đoạn có nhiều gờ khó hơn;
- reward tiến về đích, phạt đứng yên dưới 1 cm/0,1 s, phạt roll/pitch quá 30 độ và phạt timeout theo quãng đường còn lại;
- đánh giá success, thời gian và roll/pitch.

Bài báo xuất vận tốc/góc lái và dùng SWAE cho bản đồ cao độ. Nino là robot vi sai 2WD, vì vậy implementation này **đổi action thành mô-men bánh trái/phải** và không giả vờ có bản đồ cao độ khi robot chỉ có LiDAR 2D.

### Từ *Reinforcement Learning-Based Control of a 4-Wheel Independent Steering Mobile Robot for Robust Path Tracking in Outdoor Environments*

- 9 điểm nhìn trước ở `0.5, 1, 2, 3.5, 5, 7.5, 10, 12.5, 15 m`;
- reward progress/alignment/lateral/heading/smoothness và các hệ số trong paper;
- mạng policy/value `[256, 256]`, ELU, `gamma=0.99`, learning rate `5e-4`;
- domain randomization cho độ bám, trễ motor 10–40 ms, noise/bias/dropout quan sát.

Bài báo dùng robot 4WIS, Isaac Lab và 4096 môi trường song song. Nino dùng Gazebo Harmonic với một mô phỏng vật lý, nên action 4WIS được đổi thành 2 mô-men bánh. RTX chạy phần cập nhật neural network; Gazebo/ROS vẫn chủ yếu chạy CPU. Con số 59 triệu bước/18 phút của paper **không phải** tốc độ có thể kỳ vọng từ một Gazebo đơn.

Trong implementation hiện tại, độ bám `[0.3, 1.0]` được mô phỏng bằng hệ số truyền mô-men ngẫu nhiên theo episode; không thay đổi hệ số contact của SDF khi simulator đang chạy. Dãy gờ thật trong `long_hall.sdf` cao khoảng 1,6–4,4 cm và curriculum tăng dần quãng đường chứa gờ. Paper dùng đường Catmull–Rom kín bán kính khoảng 15 m; hành lang Nino không đủ rộng, nên code sinh đường Catmull–Rom mở ngẫu nhiên với biên ngang tối đa 0,6 m và tăng biên theo curriculum.

### Observation và action của Nino

Observation có 36 giá trị hữu hạn, được chuẩn hóa:

| Thành phần | Số chiều |
|---|---:|
| 9 waypoint `(x,y)` trong hệ robot | 18 |
| `cos/sin` sai số hướng | 2 |
| vận tốc thẳng và yaw rate | 2 |
| encoder velocity trái/phải | 2 |
| IMU roll, pitch, gyro-z, accel-x, accel-z | 5 |
| khoảng cách nhỏ nhất trong 5 vùng LiDAR | 5 |
| action trước đó | 2 |

Action là `Box([-1,-1], [1,1])`, nhân với giới hạn mặc định `4 N.m`, rồi phát lên `/wheel_torque_commands` theo thứ tự `[trái, phải]`. `effort_drive` vẫn giới hạn cứng tối đa `12 N.m`, slew-rate và timeout 0,25 s.

## 1. Cài đặt Ubuntu 24.04 + ROS 2 Jazzy + RTX 5060

Các dependency ROS:

```bash
sudo apt update
sudo apt install python3-venv python3-colcon-common-extensions \
  ros-jazzy-ros-gz ros-jazzy-gz-ros2-control \
  ros-jazzy-ros-gz-interfaces ros-jazzy-tf2-ros
```

Tạo môi trường Python nhưng giữ quyền nhìn thấy `rclpy` của ROS:

```bash
cd /home/tue/ninorobot
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/nino_rl/requirements.txt
```

Driver CUDA 13.0 có khả năng chạy wheel PyTorch CUDA tương thích ngược. Không cài PyTorch CPU-only. Kiểm tra thật sự bằng phép nhân tensor trên GPU:

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install
source install/setup.bash
ros2 run nino_rl check_cuda
```

Kết quả đúng phải có `CUDA khả dụng: True`, tên `NVIDIA GeForce RTX 5060` và dòng `Phép nhân CUDA kiểm tra: OK`.

Luôn build bằng `python -m colcon` khi `.venv` đang active. Nhờ vậy executable Python trong `install/` dùng đúng interpreter có PyTorch/SB3, đồng thời vẫn import được ROS Jazzy.

## 2. Huấn luyện

Đóng mọi phiên `sim.launch.py` cũ trước, sau đó mở hai terminal.

Terminal 1 — mô phỏng headless và service reset episode:

```bash
cd /home/tue/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 launch nino_rl training_sim.launch.py headless:=true
```

Terminal 2 — train 500.000 bước bằng CUDA:

```bash
cd /home/tue/ninorobot
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
source install/setup.bash
ros2 run nino_rl train --timesteps 500000 --output rl_runs
```

Chạy thử pipeline ngắn trước khi train dài:

```bash
ros2 run nino_rl train --timesteps 4096 --checkpoint-every 2048 --check-env
```

Mỗi run tạo thư mục theo thời gian gồm:

```text
rl_runs/YYYYMMDD-HHMMSS/
├── checkpoints/nino_ppo_*.zip
├── monitor.csv
├── nino_ppo_final.zip
├── ppo.yaml
└── tensorboard/
```

Theo dõi reward/loss/GPU:

```bash
tensorboard --logdir rl_runs
watch -n 1 nvidia-smi
```

Train tiếp từ checkpoint:

```bash
ros2 run nino_rl train \
  --resume rl_runs/YYYYMMDD-HHMMSS/checkpoints/nino_ppo_250000_steps.zip \
  --timesteps 250000
```

Các stage curriculum mặc định kết thúc ở `x=4, 10, 18, 30 m`. Chỉnh torque, curriculum, reward, PPO và randomization trong `config/ppo.yaml`. Sau khi chỉnh phải build lại hoặc truyền đường dẫn source:

```bash
ros2 run nino_rl train --config /home/tue/ninorobot/src/nino_rl/config/ppo.yaml
```

## 3. Test model

Giữ Terminal 1 đang chạy `training_sim.launch.py`, rồi chạy:

```bash
ros2 run nino_rl evaluate \
  --model rl_runs/YYYYMMDD-HHMMSS/nino_ppo_final.zip \
  --episodes 10
```

Test độ bền với noise/trễ/độ bám ngẫu nhiên:

```bash
ros2 run nino_rl evaluate \
  --model rl_runs/YYYYMMDD-HHMMSS/nino_ppo_final.zip \
  --episodes 25 --randomized
```

Output trong `rl_runs/evaluation/` gồm từng episode ở CSV và summary JSON: success rate, thời gian, sai số ngang trung bình, roll/pitch trung bình. Nên chỉ chuyển sang robot thật khi test deterministic và randomized đều ổn định, không rollover/collision và sai số phù hợp giới hạn cơ khí của bạn.

## 4. Chạy policy trong Gazebo

Mở simulation có GUI/RViz:

```bash
ros2 launch nino_rl training_sim.launch.py headless:=false rviz:=true
```

Chạy policy với đường thẳng mặc định trong `config/path.yaml`:

```bash
ros2 run nino_rl policy_node \
  --model rl_runs/YYYYMMDD-HHMMSS/nino_ppo_final.zip \
  --use-sim-time
```

Dừng bằng `Ctrl-C`; node luôn phát torque `0,0` khi shutdown, và `effort_drive` tự timeout nếu policy chết.

## 5. Kết nối Nav2/Linorobot2

`policy_node` nghe `nav_msgs/msg/Path` trên `/plan`. Nếu Nav2 phát path trong frame `map`, node dùng TF để đổi sang `odom`. Khi chưa có `/plan`, nó dùng `config/path.yaml`.

Sau khi Linorobot2 + Nav2 đã chạy và có `/odom`, `/imu/data`, `/joint_states`, `/scan`, TF `map -> odom`, chạy:

```bash
source /opt/ros/jazzy/setup.bash
source /home/tue/ninorobot/.venv/bin/activate
source /home/tue/ninorobot/install/setup.bash
ros2 run nino_rl policy_node \
  --model /duong/dan/nino_ppo_final.zip \
  --plan-topic /plan --device cuda
```

Không thêm `--use-sim-time` trên robot thật. Có thể remap nếu Nav2 của bạn phát global plan ở tên khác, ví dụ `--plan-topic /plan_smoothed`.

Nav2 controller server vẫn có thể phát `/cmd_vel`; lệnh torque từ policy có ưu tiên trong adapter khi được phát đều ở 10 Hz. Progress checker và goal checker của Nav2 vẫn theo `/odom`. Trước khi chạy thật, kiểm tra:

```bash
ros2 topic hz /plan
ros2 topic hz /odom
ros2 topic hz /imu/data
ros2 topic hz /joint_states
ros2 topic hz /scan
ros2 topic echo /wheel_torque_applied
```

## Xử lý lỗi nhanh

- `Missing /world/long_hall/control`: phải launch bằng `training_sim.launch.py`, không chỉ `sim.launch.py`.
- thiếu `/reset_wheel_odometry`: build/source lại workspace sau thay đổi `nino_control`.
- `torch.cuda.is_available() = False`: active đúng `.venv`, chạy `check_cuda`; không cho script âm thầm train CPU.
- thiếu sensor khi reset: kiểm tra simulation chỉ chạy một phiên và bốn topic `/odom`, `/imu/data`, `/joint_states`, `/scan` đang có dữ liệu.
- model báo shape khác `(36,)`: model đó được train bằng observation version khác, không được dùng trực tiếp.
- robot rung mạnh: giảm `max_wheel_torque_nm`, tăng phạt `smoothness_weight`, rồi train/evaluate lại; không chỉnh model trong lúc đang chạy thật.

## Nguồn

- Tong Xu, Chenhui Pan, Xuesu Xiao, *Reinforcement Learning for Wheeled Mobility on Vertically Challenging Terrain*, arXiv:2409.02383.
- Hyoseok Lee, Hyun-Min Joe, *Reinforcement Learning-Based Control of a 4-Wheel Independent Steering Mobile Robot for Robust Path Tracking in Outdoor Environments*, Sensors 2026, 26, 1761.
- [Cài PyTorch và kiểm tra CUDA](https://docs.pytorch.org/get-started/locally/)
- [Stable-Baselines3: custom environment](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [Gazebo Harmonic và ROS 2](https://gazebosim.org/docs/harmonic/ros2_integration/)
