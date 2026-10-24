# Tạo bản đồ và điều hướng Nino (ROS 2 Jazzy)

## Hiểu luồng dữ liệu

- **SLAM Toolbox** dùng lidar `/scan` và chuyển động đo từ encoder để vừa dựng
  bản đồ vừa ước lượng vị trí robot. Bạn lái robot bằng bàn phím khi tạo bản đồ.
- **Map saver** lưu bản đồ thành ảnh `.pgm` và file `.yaml`. YAML chứa độ phân
  giải, gốc tọa độ và đường dẫn tới ảnh; cần giữ cả hai file cùng nhau.
- **AMCL** dùng lidar để xác định robot đang ở đâu trên bản đồ đã lưu.
- **Nav2** tìm đường, tránh vật cản và gửi `/cmd_vel` tới `effort_drive`.
  Bộ điều khiển hiện tại tiếp tục chuyển vận tốc thành mô-men hai bánh.

Chuỗi hệ tọa độ cần có là `map → odom → base_footprint → laser`.
SLAM hoặc AMCL cung cấp `map → odom`; chỉ chạy **một trong hai**.
`effort_drive` cung cấp `odom → base_footprint`. Không bật
`linorobot2_mode:=true` trong hướng dẫn này vì chưa chạy bộ lọc EKF thay thế nó.

## Chuẩn bị

Các lệnh ROS dưới đây chạy **trong container**. Mỗi terminal WSL mới, vào bằng:

```bash
docker exec -it nino-dev bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Container hiện tại đã được cài dependencies. Dockerfile cũng đã bổ sung chúng
để lần build image tiếp theo vẫn có sẵn. Nếu dùng container khác chưa có:

```bash
apt-get update
apt-get install -y ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox
```

Build sau khi cập nhật mã nguồn (rồi source lại ở mỗi terminal):

```bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 1. Chạy robot và tạo bản đồ

Terminal 1 — khởi động một mô phỏng duy nhất:

```bash
ros2 launch nino_description sim.launch.py
```

Terminal 2 — khởi động SLAM và RViz:

```bash
ros2 launch nino_navigation mapping.launch.py
```

Terminal 3 — lái bằng bàn phím; giữ terminal này được chọn để nhận phím:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```

Giảm tốc bằng phím `z` trước khi lái, nhấn `k` để dừng. Đi chậm, quay từ từ,
đi qua các vùng cần lập bản đồ, rồi quay lại vùng đã đi qua để SLAM có cơ hội
hiệu chỉnh sai lệch. Trong RViz, bản đồ phải dần hiện ra và lidar khớp với tường.
Bản đồ chỉ bao gồm vùng đã quan sát; khởi động SLAM không tự lái robot khám phá.

World mặc định là hành lang dài có gờ trên sàn. Hành lang ít đặc điểm có thể
khiến ước lượng vị trí dọc hành lang kém chính xác. Lidar 2D ở cao cũng không
nhìn thấy mọi gờ thấp; Nav2 không tự đánh giá khả năng vượt gờ của robot.

## 2. Lưu bản đồ

Dừng robot bằng `k`, giữ SLAM chạy. Trong terminal khác:

```bash
mkdir -p /ros2_ws/maps
ros2 run nav2_map_server map_saver_cli -f /ros2_ws/maps/hall_01 \
  --ros-args -p use_sim_time:=true -p save_map_timeout:=15.0
ls -lh /ros2_ws/maps/hall_01.*
```

Chờ thông báo `Map saved successfully`. Kết quả là `hall_01.yaml` và
`hall_01.pgm`. Chọn tên mới cho lần lưu tiếp theo để giữ bản cũ.
Thư mục `/ros2_ws` được mount từ workspace nên bản đồ vẫn còn sau khi tắt container.

## 3. Điều hướng trên bản đồ đã lưu

Nhấn `Ctrl-C` ở terminal SLAM/RViz và terminal teleop. Giữ mô phỏng chạy;
cũng có thể khởi động lại mô phỏng với cùng world rồi thực hiện bước dưới.
Không chạy SLAM đồng thời với AMCL.

```bash
ros2 launch nino_navigation navigation.launch.py map:=/ros2_ws/maps/hall_01.yaml
```

Trong RViz:

1. Chọn **2D Pose Estimate**, nhấn và kéo tại vị trí robot thực tế trên bản đồ;
   hướng kéo là hướng đầu robot. Khi vừa khởi động lại mô phỏng, robot ở
   `(0, 0)` và hướng `+X`; nếu giữ mô phỏng từ lúc mapping, dùng vị trí cuối cùng.
2. Đợi lidar khớp với tường trên bản đồ. Đây là bước giúp AMCL biết vị trí ban đầu.
3. Chọn **2D Goal Pose**, nhấn và kéo tại vùng trống đã được lập bản đồ.
   Nav2 sẽ tạo đường và điều khiển robot đến đích.

Khởi đầu bằng một đích cách khoảng 1 m. Chỉ dùng một nguồn lệnh chuyển động
khi Nav2 đang chạy; không điều khiển bàn phím hoặc gửi mô-men trực tiếp cùng lúc.
`Ctrl-C` launch navigation để dừng Nav2; watchdog của `effort_drive` đưa vận tốc
đích về 0 khi không còn lệnh.

## Kiểm tra khi chưa chạy được

```bash
ros2 topic hz /scan
ros2 topic hz /odom
ros2 run tf2_ros tf2_echo odom base_footprint
ros2 run tf2_ros tf2_echo map base_footprint
ros2 lifecycle get /amcl
ros2 lifecycle get /bt_navigator
ros2 topic info /cmd_vel --verbose
```

`amcl` và `bt_navigator` cần ở trạng thái `active`. Nếu thiếu TF `map`, kiểm tra
SLAM đã nhận scan hay AMCL đã nhận **2D Pose Estimate**. Nếu bản đồ không nạp,
kiểm tra YAML và ảnh đi kèm có tồn tại trong container.

Để chạy không mở RViz, thêm `rviz:=false` vào launch mapping/navigation.
Các launch này không tự khởi động Gazebo; mô phỏng phải chạy trước.

## Cấu hình để bạn tự tìm hiểu

- `src/nino_navigation/config/slam.yaml`: độ phân giải bản đồ 0,05 m/ô,
  các frame và topic lidar.
- `src/nino_navigation/config/nav2.yaml`: tốc độ tối đa 0,25 m/s,
  tốc độ quay 0,6 rad/s, bán kính tránh va chạm 0,38 m. Bán kính này bao quanh
  góc chassis đo từ CAD: `sqrt(0,31² + 0,19²) ≈ 0,364 m`, cộng khoảng dư.
- **Costmap** là lưới đánh dấu chỗ trống, vật cản và vùng đệm quanh vật cản.
  `inflation_radius` mở rộng vùng có chi phí cao quanh tường để robot ưu tiên
  giữ khoảng cách. Nó khác với bán kính vật lý của robot.
- `enable_stamped_cmd_vel: false` giữ kiểu `geometry_msgs/msg/Twist`
  tương thích với `effort_drive` hiện tại.

Bài thực hành: tự lái và lưu `hall_01`, sau đó khởi động lại mô phỏng, đặt pose
ban đầu và thử ba đích khác nhau. Quan sát lidar có còn khớp tường khi robot quay
hay vượt gờ không trước khi tăng tốc độ.

Tham khảo: [Nav2 Jazzy](https://api.nav2.org/nav2-jazzy/html/index.html),
[SLAM Toolbox](https://docs.ros.org/en/jazzy/p/slam_toolbox/).

## Kiểm chứng tích hợp ngày 24/10/2026

Đã build cả ba package trong container `nino-dev`, chạy Gazebo không mở GUI,
nhận `/scan` và `/map`, lưu bản đồ thử 265 × 83 ô rồi tắt SLAM. Sau đó khởi
động AMCL/Nav2 bằng chính bản đồ vừa lưu, gửi pose ban đầu và đích `(1, 0)`.
`controller_server` và `bt_navigator` đều `active`; action trả về `SUCCEEDED`
với `error_code: 0`. Odometry cuối khoảng `(0,777; 0)` m, nằm trong dung sai
đích 0,25 m. Đã xác nhận `/cmd_vel` là `Twist` và `effort_drive` nhận topic này.
8 test hiện có đều qua. Trường hợp đường dẫn bản đồ không tồn tại báo lỗi rõ ràng.

Log và bản đồ **thử nghiệm ngắn**, chưa phải bản đồ đầy đủ hành lang, nằm ở
`log/nav2_smoke/`. Kiểm chứng này chưa bao gồm thao tác RViz bằng tay, quét toàn
bộ hành lang hoặc điều hướng qua mọi gờ sàn.
