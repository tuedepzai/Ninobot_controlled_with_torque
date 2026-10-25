# Tạo bản đồ và điều hướng Nino (ROS 2 Jazzy)

## Mặc định mới: giữ nguyên đường trong mỗi episode

`navigation.launch.py` hiện mặc định `fixed_path:=true`. Khi nhận một đích:

1. Gọi `ComputePathToPose` **một lần**, dùng planner `GridBased` hiện có.
2. Lưu nguyên đường trả về và gửi một goal `FollowPath` cho DWB.
3. Giữ đường gốc cho tới khi hoàn thành, thất bại hoặc bị hủy. Không retry,
   recovery, tạo đường vòng hoặc lập đường lại trong episode.
4. Nếu khoảng cách robot tới đường vượt `0.20 m`, hủy action và chặn vận tốc
   về 0. Giữ trạng thái thất bại cho tới khi reset.

**Giữ đường gốc không có nghĩa giữ nguyên hướng bánh.** Nếu robot bị lệch nhẹ,
DWB vẫn điều khiển robot quay về đường cũ. Đường tham chiếu không bị bẻ cong
để nối từ vị trí lệch đến đích. Đường tính ban đầu có thể cong do map và vật cản;
chế độ này không ép planner tạo đường thẳng qua vật cản.

Trong RViz, xem **Fixed reference path** (`/episode/reference_path`, màu xanh).
Không dùng đường cục bộ của DWB để đánh giá xem đường gốc có bị thay đổi hay không.
Đường gốc giữ nguyên tọa độ trong `map`; vị trí robot vẫn được AMCL cập nhật.

Chạy mô phỏng như hướng dẫn bên dưới, sau đó trong terminal navigation:

```bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select nino_navigation
source install/setup.bash
ros2 launch nino_navigation navigation.launch.py \
  map:=/ros2_ws/maps/hall_01.yaml max_path_deviation:=0.20
```

Đặt **2D Pose Estimate**, chờ lidar khớp tường rồi gửi **2D Goal Pose**.
Mỗi đích được chấp nhận là một episode. Đích mới khi đang chạy sẽ bị từ chối,
thay vì tự đổi đường. Sau mỗi episode, reset trước khi gửi đích tiếp:

```bash
ros2 service call /episode/reset std_srvs/srv/Trigger '{}'
```

Reset chỉ xóa trạng thái episode; không dịch chuyển robot hay reset Gazebo.
Reset bị từ chối nếu action cũ chưa kết thúc hoặc chưa xác nhận robot dừng.
Hủy episode đang chạy:

```bash
ros2 service call /episode/cancel std_srvs/srv/Trigger '{}'
```

Xem trạng thái và độ lệch:

```bash
ros2 topic echo /episode/state --qos-durability transient_local
ros2 topic echo /episode/path_deviation
```

Kiểm tra node đã sẵn sàng nhận đích (kể cả TF trong buffer của chính nó):

```bash
ros2 service call /episode/ready std_srvs/srv/Trigger '{}'
```

`/episode/state` chứa trạng thái, nguyên nhân, số yêu cầu lập đường và mã SHA256
của đường gốc. Các tham số trong `config/episode.yaml` được đọc khi khởi động;
khởi động lại launch để áp dụng thay đổi. `max_path_deviation` cũng có launch
argument như ví dụ trên. Độ lệch là khoảng cách ngắn nhất từ vị trí robot tới
các đoạn thẳng nối các điểm trên đường, tính trong `map`, kiểm tra ở 20 Hz.

Chế độ này giữ AMCL, Navfn, DWB, velocity smoother và collision monitor.
Không chạy BT navigator, behavior server, route server, docking server hay
waypoint follower, nên không có nhánh tự tìm đường thay thế. Node
`episode_manager` là nguồn `/cmd_vel` duy nhất; nó nhận lệnh đã qua collision
monitor trên `/episode/cmd_vel_candidate`. Không chạy teleop, lệnh mô-men hoặc
launch navigation khác đồng thời. Node từ chối bắt đầu nếu phát hiện nguồn
`/cmd_vel` khác, nhưng đây không phải cơ chế khóa topic trước publisher bên ngoài.

Khi thất bại/hủy: chặn lệnh khác 0 ngay, yêu cầu hủy action, tiếp tục phát vận
tốc 0 và dùng `/odom` kiểm tra dừng. Nếu không xác nhận được dừng/action kết
thúc trong 5 giây, báo thất bại và tiếp tục giữ lệnh 0. Watchdog và bộ điều
khiển mô-men nền được giữ nguyên. Launch mô phỏng mặc định không bật phần bù
RL; tích hợp SAC thử nghiệm và hướng dẫn riêng ở
[`rl_training/README.md`](rl_training/README.md).

Nếu cần kiểm tra lại Nav2 tiêu chuẩn có replanning, dùng riêng:

```bash
ros2 launch nino_navigation navigation.launch.py \
  map:=/ros2_ws/maps/hall_01.yaml fixed_path:=false
```

### Giới hạn của phép đo

- Bộ giám sát dùng vị trí **ước lượng từ AMCL/TF**, không dùng vị trí thật của
  Gazebo. Nếu robot trượt nhưng AMCL chưa nhận ra, lỗi thật có thể lớn hơn lỗi
  đang đo. AMCL cũng có thể làm lỗi nhảy khi định vị lại.
- Xác nhận dừng dùng vận tốc encoder trong `/odom`, chưa chứng minh robot không
  còn trượt ngang. Thử nghiệm robot thật cần bổ sung đánh giá độc lập.
- Đường tự cắt/chạy gần nhau có thể làm phép tìm đoạn gần nhất nhận nhầm nhánh.
- Chế độ này tính một đường mới ở đầu mỗi episode, không đảm bảo đường giữa các
  episode giống nhau. Chưa hỗ trợ nhập quỹ đạo lưu sẵn.
- `NavigateToPose.behavior_tree` phải để trống; không có BT chạy trong chế độ này.
  Feedback `distance_remaining` là khoảng cách thẳng tới điểm cuối; thời gian
  còn lại chưa được ước lượng.

### Kiểm chứng chế độ đường cố định

```bash
colcon test --packages-select nino_control nino_navigation
colcon test-result --verbose

# Chỉ chạy khi không có Gazebo/Nav2 khác: script tự mở rồi đóng mô phỏng.
python3 src/nino_navigation/test/verify_fixed_path.py
python3 src/nino_navigation/test/verify_fixed_path.py --standard
```

Đã chạy 18 kiểm thử mới về khoảng cách, số yêu cầu action, giữ nguyên đường,
thành công, lỗi controller, quá ngưỡng, hủy, nhận goal muộn, timeout và reset.
`colcon test-result` báo 28 test entries, 0 lỗi, 0 thất bại (bao gồm test của
`nino_control` và các mục tổng hợp của CTest).

Kiểm chứng Gazebo/Nav2 ngày 24/10/2026: đích 1 m thành công với đường không đổi.
Episode thứ hai dịch robot ngang và đặt lại AMCL tương ứng để tạo lỗi định vị
đã biết; đo lệch `0.305 m > 0.200 m`, action thất bại, giữ nguyên đường và phát
lệnh 0. Vận tốc odometry cuối khoảng `0.00101 m/s`, `-0.00050 rad/s`.
Bằng chứng nằm ở `log/fixed_path_20261024_143945_228540/result.json` và log đi kèm.
Thử chỉ dịch robot mà không đặt lại AMCL chưa làm bộ giám sát nhận ra đầy đủ độ
lệch; do đó không coi thử nghiệm này là xác nhận phát hiện mọi kiểu trượt.
Chế độ Nav2 cũ (`fixed_path:=false`) cũng đã chạy lại đích 1 m thành công;
kết quả ở `log/fixed_path_20261024_144148_281026/result.json`.

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
