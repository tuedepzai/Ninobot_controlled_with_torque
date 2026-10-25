# Residual SAC cho Nino trong Gazebo

Đây là phần tích hợp và huấn luyện thử, **chưa phải policy đã chứng minh cải
thiện khả năng vượt dây**. Mục tiêu đã chọn: ưu tiên bám đường/hoàn thành episode,
sau đó giảm rung. Phần Nav2 tính một đường duy nhất mỗi episode vẫn được giữ.

## Điều khiển nền và phần RL

Robot trong repo có hai bánh chủ động và hai caster thụ động. RL có hai đầu ra,
theo thứ tự `[trái, phải]`:

```text
Nav2 → vận tốc → PI → mô-men nền ─────────┐
                                        ├→ giới hạn → bánh xe
SAC → action [-1,1]² → nhân 0,3 N·m ─────┘
```

`τ = clip(τ_PI + Δτ_RL, -2, +2)` N·m trong chế độ bám vận tốc hiện tại.
Sau đó lệnh vẫn qua giới hạn độ thay đổi mô-men, bảo vệ tốc độ và giới hạn
phần cứng như trước. Không tăng giới hạn PI hoặc mô-men phần cứng.

`enable_residual` mặc định **false** trong launch mô phỏng. Môi trường RL tự bật
khi train/evaluate và tắt khi baseline. Phần bù chỉ có hiệu lực khi:

- Episode đang `FOLLOWING`.
- Lệnh vận tốc còn mới và khác 0.
- Lệnh RL là hai số hữu hạn, chưa quá 0,2 giây theo đồng hồ thực.
- Không ở chế độ gửi mô-men trực tiếp.

Thất bại/hủy/dừng làm phần bù về 0. Collision monitor và EpisodeManager vẫn giữ
quyền dừng. `/wheel_torque_commands` là lệnh **ghi đè** mô-men, không phải đầu
vào cho RL; RL dùng `/wheel_torque_residual`.

## SAC học gì?

**Observation** là dữ liệu policy nhận vào. Có 23 giá trị được chia tỷ lệ và
giới hạn trong `[-10, 10]`: lỗi ngang và góc so với đường, khoảng đường còn lại,
roll/pitch, gia tốc IMU đã trừ trọng lực, vận tốc góc IMU, vận tốc robot, vận tốc
yêu cầu, tốc độ hai bánh, mô-men PI, action trước và phần bù được chấp nhận.
Thứ tự chính xác ở `ros_env.py: OBSERVATIONS` và được lưu trong metadata mỗi run.

**Action** là hai số trong `[-1, 1]`, tương ứng mô-men bù tối đa ±0,3 N·m.
SAC không thay đích, đường hay lệnh vận tốc Nav2.

**Reward** là điểm để policy học. Trong `features.py`:

```text
r = 20 × tiến triển dọc đường
    - dt × (8 × lỗi ngang² + 0,5 × lỗi góc²
            + 0,02 × ||gia tốc đã trừ trọng lực||²
            + 0,01 × ||action||² + 0,05)
    - 0,01 × ||action - action_trước||²
    + 30 nếu thành công, -30 nếu thất bại hoặc hết thời gian
```

Tiến triển được tính từ vị trí chiếu lên đường trong `map`, không phải chỉ
quãng đường bánh quay. Mức thưởng/phạt này là điểm khởi đầu cần kiểm chứng,
không phải bộ trọng số tối ưu. Bài thực hành cho bạn: xem từng thành phần trong
`steps.jsonl`, giải thích vì sao policy đứng yên không thể nhận thưởng dương.

Mỗi step khoảng 0,1 giây mô phỏng. Phần bù được phát lại trong lúc chờ step.
Nếu việc cập nhật mạng lâu hơn watchdog, hệ thống tự quay về điều khiển nền.
Môi trường theo API [Gymnasium](https://gymnasium.farama.org/introduction/create_custom_env/),
SAC dùng [Stable-Baselines3 2.7.1](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/sac.html).

## Cài đặt trong container

Môi trường Python riêng nằm ở `rl_training/.venv` trong container hiện tại.
Để cài lại khi cần:

```bash
cd /ros2_ws
python3 -m venv --system-site-packages rl_training/.venv
rl_training/.venv/bin/python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
rl_training/.venv/bin/python -m pip install -r rl_training/requirements.txt
```

Train/evaluate mặc định dùng GPU (`--device cuda`), kiểm tra cả CUDA forward và
backward trước khi mở Gazebo. Nếu GPU không dùng được, báo lỗi rõ ràng thay vì
âm thầm chuyển CPU. Chọn `--device cpu` khi muốn chạy CPU, hoặc `--device auto`
để tự chọn. Baseline không chạy mạng nên vẫn dùng CPU. Không thay thiết lập
GPU/WSLg/Docker; chỉ thay bản Torch trong venv. RTX 5060 cần bản Torch hỗ trợ
kiến trúc Blackwell; bản 2.7.1 CUDA 12.8 được chọn theo
[hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/).

GPU thực hiện tính toán mạng SAC; ROS, bộ PI và phần lớn mô phỏng vật lý vẫn
chạy trên CPU, nên tốc độ tổng có thể còn bị giới hạn bởi Gazebo.

Build phần ROS sau khi cập nhật mã:

```bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
source rl_training/.venv/bin/activate
```

## Chạy theo thứ tự

**Tắt các phiên Gazebo/Nav2/teleop khác trước.** Môi trường tự khởi động Gazebo
và Nav2, đặt AMCL ở vị trí xuất phát, gửi đích, thu dữ liệu, hủy và đóng các tiến
trình nó tạo. Nó từ chối chạy nếu phát hiện mô phỏng/nguồn `/cmd_vel` khác.

### 1. Mốc không bù RL

```bash
python rl_training/train.py --mode baseline --map /ros2_ws/maps/hall_01.yaml \
  --episodes 3
```

Mặc định đích `(11, 0)` nằm sau dây cuối ở khoảng x=9,5 m, cách tường cuối ở
x=12 m một mét để thân robot và vùng đệm tránh vật cản còn chỗ. Không có tác
động mô-men bù trong baseline.
Thêm `--gui` nếu muốn mở Gazebo. Mặc định chỉ chạy server mô phỏng.

### 2. Kiểm tra SAC ngắn

```bash
python rl_training/train.py --mode smoke --map /ros2_ws/maps/hall_01.yaml
```

Chạy 32 bước, bắt đầu cập nhật mạng sau 8 bước. Episode giới hạn 2 giây để
kiểm tra reset nhiều lượt. **Model này chỉ dùng kiểm tra đường dữ liệu**, không
dùng để kết luận RL tốt hơn PI. Bước này kiểm tra thật SAC, không phải mô phỏng
một thuật toán học bằng các số ngẫu nhiên.

### 3. Huấn luyện

```bash
python rl_training/train.py --mode train --map /ros2_ws/maps/hall_01.yaml \
  --steps 10000 --seed 1 --device cuda
```

SAC mặc định thu 1000 bước trước khi cập nhật mạng; sau đó mỗi step cập nhật một
lần. Cấu hình nằm ở `config/sac.yaml`. Thời gian chạy phụ thuộc tốc độ Gazebo;
bản đầu reset bằng khởi động lại mô phỏng nên chậm. Không tự chạy huấn luyện
dài trong quá trình tích hợp. `Ctrl-C` yêu cầu hủy, bỏ mô-men bù, lưu model đang
có và đóng mô phỏng của run đó.

### 4. Đánh giá model

### Train dài và tự so sánh sau khi đủ bước

`long_run.py` tiếp tục model cùng replay buffer từ một run có sẵn. `--steps`
là **tổng số bước đích**, không phải số bước cộng thêm. Nạp model 10.000 bước
với `--steps 50000000` sẽ học thêm 49.990.000 bước. Model gốc được giữ nguyên.
Không sửa reward, giới hạn mô-men hay watchdog để chạy lâu hơn.

Job đang cấu hình cho 50M:

```bash
python -u rl_training/long_run.py \
  --resume-train-run /ros2_ws/rl_training/runs/train_20261025_181046_262901 \
  --map /ros2_ws/maps/hall_01.yaml --steps 50000000 \
  --checkpoint-every 100000 --episodes 10 --eval-every 20000 --eval-episodes 3 \
  --import-evaluation /ros2_ws/rl_training/runs/comparison_20261025_174125_962971 \
  --output /ros2_ws/rl_training/runs/long_50m_20261026_v5
```

Job trên đã được khởi chạy nền bằng `docker exec -d`; **không chạy lại lệnh
đó khi phiên đang chạy**. Đóng terminal không dừng job, nhưng Docker/WSL/máy
vẫn phải hoạt động. Không tự thay cài đặt nguồn, Docker hay GPU.

Theo tốc độ run 10.000 bước trước (~27 phút), 50 triệu bước có thể mất khoảng
3 tháng, chưa tính gián đoạn. Đây không phải cam kết thời gian hoàn tất hay
cam kết policy tốt hơn. Checkpoint mặc định hiện là mỗi 100.000 bước thay vì
1.000 để tránh lưu hàng chục nghìn bản replay lớn. Không tự xóa checkpoint.
Ngoài ra mỗi step vẫn ghi JSON nên dữ liệu dài hạn có thể chiếm hàng chục đến
hàng trăm GB; cần theo dõi dung lượng.

Theo dõi trong container:

```bash
cat /ros2_ws/rl_training/runs/long_50m_20261026_v5/status.json
tail -n 40 /ros2_ws/rl_training/runs/long_50m_20261026_v5/train_000000020000.log
df -h /ros2_ws
```

`status.json` lưu PID worker, PID train và đường dẫn run từng phase. `train_log`
là tên log của đoạn train hiện tại; ví dụ trên là đoạn đến mốc 20.000. Mỗi đoạn
có log riêng, không ghi đè log đoạn trước.
`TRAIN` là đang huấn luyện; sau đó lần lượt `BASELINE`, `EVALUATE`, `REPORT`,
`COMPLETE`. Khi thành công, `phases.report` là thư mục báo cáo HTML/PNG/CSV.
PI và SAC dùng cùng config/đích/map, 10 lượt mỗi chế độ, seed đánh giá bắt đầu
101. Không so sánh model chưa đủ 50M như thể đã đạt 50M.

Dừng an toàn từ container (chỉ worker của job này):

```bash
python - <<'PY'
import json, os, signal
from pathlib import Path
job = Path('/ros2_ws/rl_training/runs/long_50m_20261026_v5')
status = json.loads((job/'status.json').read_text())
if status['state'] not in ('COMPLETE', 'FAILED', 'INTERRUPTED'):
    command = Path(f'/proc/{status["pid"]}/cmdline').read_bytes().split(b'\0')
    if str(job).encode() not in command or not any(c.endswith(b'/long_run.py') for c in command):
        raise RuntimeError('PID không còn trỏ tới worker của job này; không gửi signal')
    os.kill(status['pid'], signal.SIGINT)
PY
```

Worker chuyển SIGINT cho trainer và đợi nó hủy goal, bỏ residual, lưu model
và replay rồi đóng các tiến trình sở hữu. Không dùng `kill -9` khi muốn lưu.
Timeout cảm biến/lifecycle vẫn dừng job và lưu fault; **không tự retry một
episode đang chạy**, không thêm reward giả. `FAILED` ghi lý do trong status
và log. Nếu máy/Docker bị tắt đột ngột, status có thể còn trạng thái cũ; cần
kiểm tra tiến trình thực tế và dùng checkpoint còn đọc được.

Để tiếp tục sau gián đoạn, lấy `phases.train` làm `--resume-train-run`, giữ
`--steps 50000000`, và chọn **thư mục output mới**. Worker chọn model/replay
hoàn tất, snapshot fault/interrupted hoặc checkpoint có cặp tương ứng. Không
ghi đè run/job cũ. Nếu cần chọn checkpoint cụ thể, gọi trực tiếp:

```bash
python rl_training/train.py --mode train --map /ros2_ws/maps/hall_01.yaml \
  --config /ros2_ws/rl_training/runs/TEN_RUN/config.yaml \
  --resume /ros2_ws/rl_training/runs/TEN_RUN/checkpoints/sac_100000_steps.zip \
  --replay-buffer /ros2_ws/rl_training/runs/TEN_RUN/checkpoints/sac_replay_buffer_100000_steps.pkl \
  --steps 50000000 --checkpoint-every 100000 --device cuda
```

Kiểm chứng ngày 25/10/2026: 40 test RL qua. Run
`train_20261025_154155_737387` nạp thật model/replay 10.000 bước, tiếp tục trên
CUDA đến 10.032 bước/9.032 cập nhật, lưu cặp checkpoint ở 10.016 và 10.032.
Lần kiểm thử trước đó `train_20261025_153922_088278` gặp timeout dữ liệu và đã
lưu fault; vẫn giữ nguyên để điều tra. Chưa hoàn tất 50M và chưa có kết quả
so sánh sau 50M tại thời điểm khởi chạy job.

Job đầu `long_50m_20261025` dừng vì timeout dữ liệu khi luồng nhận ROS dùng
chung với learner; model/replay vẫn được lưu. Bridge RL hiện chạy executor
trên luồng riêng và chụp bản sao dữ liệu dưới lock trước khi tính observation.
Không nới timeout 0,75 s hay watchdog. Callback lỗi được truyền lên learner;
luồng nhận được shutdown/join trước khi hủy node. Test giữ luồng chính bận
hơn 0,75 s xác nhận ROS vẫn cập nhật dữ liệu. Run
`train_20261025_154830_605229` đã tiếp tục thật từ 10.000 lên 10.064 bước,
9.064 cập nhật trên CUDA với cơ chế này. Job `_v2` là job chạy lại sau sửa;
job lỗi ban đầu vẫn được giữ để kiểm tra.

Sửa bổ sung: `_v2` dừng ở reset thứ 9 vì publisher cũ còn trong ROS graph,
dù log cho thấy các tiến trình đã tắt. Mỗi reset giờ shutdown context cũ và
tạo context/DDS participant mới, giữ nguyên model/replay Python. Vẫn chờ
discovery và từ chối khởi chạy khi còn publisher `/clock` hoặc `/cmd_vel`.
Timeout ở bước này là StartupFault có tối đa ba lần thử trước goal; nếu hết
lượt thì lưu infrastructure snapshot và dừng. Không kill phiên bên ngoài.

Trainer còn lưu `exception_model.zip` + `exception_replay.pkl` cho các ngoại
lệ thông thường rồi ném lại lỗi; worker có thể chọn cặp snapshot này để resume.
45 test RL đã qua, gồm tạo context mới, chặn publisher còn sống, retry có
giới hạn và thực sự lưu/nạp model/replay khi reset ném RuntimeError.

Run `_v2` dùng code cũ không lưu được model mới trước ngoại lệ graph; log cuối
ở khoảng 10.883 bước, thư mục checkpoints chưa có cặp mới. Các bước chưa lưu
không khôi phục được từ JSON telemetry. `_v3` tiếp tục từ bản hoàn tất
`train_20261025_154830_605229` ở 10.064 bước, không tuyên bố giữ được toàn bộ
tiến triển của `_v2`. Mục tiêu tổng vẫn là 50 triệu; so sánh chỉ sau khi đạt đủ.

Kiểm thử reset thực tế sau sửa: run `baseline_20261025_162331_541146` hoàn tất
12/12 episode, mỗi episode cố ý giới hạn 0,2 giây và kết thúc TIME_LIMIT. Không
có startup recovery hay timeout graph trong run này. Đây là kiểm chứng reset
liên tiếp, không phải đánh giá điều hướng thành công hay độ ổn định 50M bước.
Job `long_50m_20261025_v3` được chạy nền sau khi kiểm thử đóng hết tiến trình.

Đánh giá giữa chừng ngày 26/10/2026: đã dừng an toàn `_v3`, lưu/nạp kiểm tra
snapshot 16.206 bước, 15.206 cập nhật và replay 16.206 mẫu. Ba lượt evaluate
SAC thành công 3/3, PI 2/3; RMSE trung bình theo N mẫu 4,220 cm và 5,156 cm
(giảm 18,16%). RMS gia tốc có trọng số thời gian mẫu tăng từ 2,247 lên
7,889 m/s². Mẫu nhỏ chưa chứng minh hiệu quả tổng quát; chưa thấy giảm mức
gia tốc IMU. Xem `runs/comparison_20261025_174125_962971/report.html` và
`assessment.md` cho nguồn, lỗi hạ tầng đã tách riêng và giới hạn. Job `_v4`
đã tiếp tục từ snapshot 16.206 bước. Chưa hoàn tất 50M.

### Evaluate mỗi 20.000 bước và dashboard realtime

Web cục bộ: **http://localhost:8766**. Chọn job `long_50m_20261026_v5`.
Dashboard cập nhật khoảng 2 giây/lần: số bước xấp xỉ theo log, reward, critic
loss, episode, lỗi ngang, mô-men bù, trạng thái tiến trình thật và bảng evaluate.
Nút Dừng cập nhật chỉ dừng tải dữ liệu của trang, không dừng train. Khi job
dừng hoặc dữ liệu bị chậm, trang báo rõ; không suy luận tiến trình còn sống
chỉ từ trạng thái đã lưu. Các biểu đồ chỉ dùng dữ liệu có thật.

Worker train đến đúng các mốc toàn cục 20k, 40k, 60k… rồi lưu model + replay,
đóng mô phỏng train, evaluate action xác định và tiếp tục từ chính bản lưu đó.
Không chạy hai mô phỏng chồng nhau, không đưa dữ liệu evaluate vào replay.
Mỗi mốc giữa chừng chạy 3 lượt SAC; PI chạy 3 lượt ở mốc đầu rồi được dùng
lại làm tham chiếu cùng cấu hình/seed. Mốc cuối 50M chạy 10 lượt mỗi chế độ.
Các lần evaluate tạo báo cáo độc lập và thêm hàng vào `evaluations.json` và
`evaluations.csv` trong thư mục job. Dashboard có đường RMSE qua các mốc,
liên kết báo cáo và nút tải bảng CSV. Mốc 16.206 bước đã đánh giá thủ công được
nhập với nhãn manual; các mốc kiểm thử ngắn nằm trong job test riêng.

Snapshot cuối mỗi đoạn 20k có model, optimizer và replay; checkpoint nội bộ
100k vẫn cấu hình nhưng đoạn thường kết thúc trước đó. Không xóa bản cũ.
Mỗi lần tiếp tục sẽ khởi tạo lại episode/ROS và RNG theo cơ chế resume SB3;
không cam kết tương đương từng bit với một lần learn không ngắt. Nếu evaluate
lỗi, worker dừng và giữ model tại mốc. Resume đúng mốc sẽ evaluate mốc đó lại
trước khi học tiếp, tránh bỏ qua đánh giá chưa hoàn tất.

SB3 dùng GPU được kiểm tra từ chính model: `device_verification.json` lưu
device của policy/actor/critic, device của kết quả actor forward và các tensor
trạng thái optimizer (trừ scalar step của Adam). Yêu cầu CUDA mà model nằm
CPU sẽ dừng rõ ràng. Model test thực tế cho actor, critic, forward và optimizer
đều `cuda:0`, RTX 5060, CUDA 12.8. GPU utilization thấp khi chờ Gazebo không
đồng nghĩa mạng đang chạy CPU. Dashboard đọc file bằng chứng này, không import
Torch hay can thiệp model. Kiểm tra thêm tiến trình bằng `docker exec nino-dev
nvidia-smi` từ host.

Trang dashboard train chỉ đọc file, bind loopback, không có nút dừng job train.
Server dữ liệu chạy trong Docker ở port 8765. Relay WSL ở port 8766 giúp Windows
truy cập localhost mà không thay cấu hình Docker/GPU/WSLg. Sau khi máy/Docker
khởi động lại, chạy từ PowerShell (không mở bản thứ hai khi port đã có server):

```powershell
docker exec -d nino-dev bash -lc 'cd /ros2_ws && exec rl_training/.venv/bin/python -u rl_training/dashboard.py --port 8765 > /tmp/nino-dashboard.log 2>&1'
wsl.exe -d Ubuntu -- python3 /home/tue/Ninobot_controlled_with_torque/rl_training/dashboard.py --port 8766 --backend-container nino-dev
```

Job `_v4` đã dừng an toàn ở 18.602 bước để chuyển lịch. Kiểm thử riêng
`long_periodic_test_20261026` hoàn tất hai mốc 18.603 và 18.604, một lượt eval
mỗi mốc chỉ để kiểm chứng luồng; baseline được tái sử dụng, bảng có đủ hai
hàng mới. `_v5` tiếp tục model/replay 18.604 bước, mục tiêu 50M, mốc tới 20.000.
64 test RL qua. API live, route báo cáo và logic JS đã được kiểm tra, gồm
trạng thái lỗi/rỗng, bảng và bằng chứng GPU; chưa xác nhận giao diện bằng
trình duyệt thật vì môi trường không có browser automation khả dụng.

### Demo Gazebo: Test PI và Test PI + SAC

Trang điều khiển demo: **http://localhost:8766/demo**. Chọn:

- **Test PI · không SAC**: robot về đầu hành lang và chạy với
  `enable_residual=false` trong effort_drive; không gọi mạng SAC để lấy action.
- **Test PI + SAC**: cùng vị trí xuất phát, bật residual và chạy model 20.000
  bước với action xác định. Không học thêm, không đổi PI gains hay giới hạn.
- **Dừng robot**: hủy goal Nav2, bỏ residual, giữ Gazebo mở.
- **Chạy lại**: khởi động lại mô phỏng để reset odom/PI/AMCL, rồi tự chạy chế
  độ đang chọn. **Camera theo robot** dùng dịch vụ GUI Gazebo để bám model nino.
- **Đóng demo**: đóng các tiến trình mô phỏng/Nav2 do demo tạo. Train vẫn tạm
  dừng cho tới khi được tiếp tục, tránh hai phiên điều khiển chồng nhau.

Hai biểu đồ cho hai bánh chủ động: PI nền, residual được chấp nhận và tổng lệnh
sau các giới hạn; đơn vị N·m. Đây không phải cảm biến đo mô-men thực. Do giới
hạn tốc độ/thay đổi mô-men, tổng lệnh có thể khác tổng hai đường kia tại cùng
mẫu. Mẫu hiển thị ở bước RL và lúc đứng yên, chưa phải toàn bộ tín hiệu 50 Hz.

Demo có server riêng, không nhận tùy ý mô-men, đường dẫn model hay topic từ web.
Chỉ các lệnh start/stop/restart/chọn PI/chọn SAC/focus/close được chấp nhận;
endpoint không điều khiển job train. Mọi thay đổi chế độ đều bắt đầu episode
mới và vẫn giữ EpisodeManager, collision monitor và watchdog. Robot có hai
bánh chủ động cùng hai caster thụ động, không phải bốn action torque độc lập.

Chỉ khởi động khi đã lưu và dừng job train/Gazebo khác:

```bash
python rl_training/demo_server.py \
  --model /ros2_ws/rl_training/runs/train_20261025_181509_419600/model.zip \
  --config /ros2_ws/rl_training/runs/train_20261025_181509_419600/config.yaml \
  --map /ros2_ws/maps/hall_01.yaml
```

Server demo bind loopback 8770; web 8766 relay tới nó. Cần giữ hai server của
dashboard như hướng dẫn trên. Không đổi Docker GPU hay WSLg. Model giữ nguyên;
dữ liệu demo ghi vào `runs/live_demo_.../`, tách khỏi các phép evaluate định kỳ.

Đã kiểm chứng ngày 26/10/2026: 70 test qua. Test PI chạy thật với
`enable_residual=False`, telemetry residual hai bánh bằng 0. Nút dừng hủy goal
và chuyển STOPPED, giữ GUI mở. Test PI + SAC chạy thật với
`enable_residual=True` và residual khác 0; GUI move_to/follow trả lời thành công.
Đã kiểm tra cú pháp/logic JS gửi đúng lệnh hai nút; capture cửa sổ WSLg bị
timeout nên không tuyên bố đã kiểm tra giao diện bằng screenshot. Job train
được lưu và tạm dừng trong khi demo hoạt động; không tự chạy chồng mô phỏng.

### Đánh giá thủ công model

Thay đường dẫn bên dưới bằng model thực tế của run train:

```bash
python rl_training/train.py --mode evaluate --map /ros2_ws/maps/hall_01.yaml \
  --model /ros2_ws/rl_training/runs/train_TEN_RUN/model.zip --episodes 3
```

Đánh giá dùng action xác định (`deterministic=True`), không cập nhật policy.
So sánh với baseline bằng cùng map, đích, giới hạn lệch, bộ trọng số và thời gian.
Chạy nhiều lượt và nhiều seed trước khi kết luận có cải thiện.

## Dữ liệu và đồ thị

### Phase tiếp theo: so sánh PI với model SAC đã train

Baseline là robot chỉ dùng bộ PI; evaluate là PI cộng mô-men bù từ model, không
cập nhật mạng. Ưu tiên đọc tỷ lệ thành công, rồi độ lệch đường, cuối cùng mới
đọc rung IMU. Reward dương hoặc lỗi/rung thấp trong episode ngắn chưa đủ để
kết luận policy tốt.

Chọn model train đã hoàn tất; không dùng model smoke để đánh giá hiệu quả.
Ví dụ model hiện có đã hoàn tất 10.000 bước:

```bash
TRAIN=/ros2_ws/rl_training/runs/train_20261025_135101_536012
python rl_training/train.py --mode baseline --config "$TRAIN/config.yaml" \
  --map /ros2_ws/maps/hall_01.yaml --episodes 3 --seed 101
python rl_training/train.py --mode evaluate --config "$TRAIN/config.yaml" \
  --map /ros2_ws/maps/hall_01.yaml --model "$TRAIN/model.zip" \
  --episodes 3 --seed 101 --device cuda
```

Chạy tuần tự, chờ lệnh trước kết thúc. Dùng cùng config, seed bắt đầu và số
episode. Mỗi episode dùng seed bắt đầu cộng số thứ tự. Đây không đảm bảo Gazebo
và AMCL lặp lại giống hệt; đường giữa các episode vẫn có thể khác nhẹ.

Thay hai đường dẫn bằng các thư mục `Run:` vừa in ra:

```bash
python rl_training/compare_runs.py \
  --baseline /ros2_ws/rl_training/runs/baseline_TEN_RUN \
  --policy /ros2_ws/rl_training/runs/evaluate_TEN_RUN
```

Script tạo thư mục `runs/comparison_.../` mới chứa `report.json`, `episodes.csv`
và `comparison.png`. Nó từ chối so sánh config/seed khác nhau, số episode hoàn
tất khác nhau, đường đổi trong episode hoặc nhiều hơn một lần planning.
Baseline có mô-men bù khác 0 cũng bị từ chối. Các episode chưa hoàn tất được
liệt kê riêng, không tính vào tỷ lệ thành công; phiên không có step được ghi
riêng để kiểm tra log khởi động. Không ghi đè kết quả cũ.

Lệnh này còn tạo báo cáo RMSE chi tiết: `report.html` (tiếng Việt, chọn từng
lượt và rê chuột đọc lỗi), `report.md`, `overview.png/.svg`,
`path_traces.png/.svg` và `rmse_analysis.json`. Mở `report.html` bằng trình
duyệt; giữ nguyên thư mục để ảnh và các liên kết CSV/JSON hoạt động. Nút In
trong báo cáo dùng hộp thoại in của trình duyệt để lưu PDF khi cần.

RMSE theo N mẫu được tính riêng: `sqrt(sum(error²)/N)`. Số ở đầu báo cáo là
trung bình RMSE từng lượt, mỗi lượt trọng số bằng nhau; RMSE gộp mọi mẫu và
RMSE trong đoạn tiến triển chung cũng được ghi riêng. Chỉ số cũ có trọng số
thời gian vẫn giữ dưới tên `cross_track_rms_m`, chỉ số mới là
`cross_track_rmse_m`. RMS gia tốc IMU không phải sai số RMSE vì chưa có tín
hiệu gia tốc mục tiêu. Nền của robot là PI, không có D, nên không gọi là PID.

Báo cáo ngày 25/10/2026 ở `runs/comparison_20261025_150658_049393/`, dùng lại
sáu episode Gazebo đã ghi (không chạy mô phỏng mới để tạo báo cáo). RMSE trung
bình PI 4,838 cm, PI + SAC 4,955 cm; đến đích 3/3 và 2/3. Đã chạy 35 test RL;
đã mở kiểm tra PNG. JavaScript được kiểm tra cú pháp và sáu trường hợp chọn
lượt/trục, đọc tooltip bằng DOM giả; chưa kiểm chứng HTML trong trình duyệt
thật do công cụ không có trình duyệt khả dụng.

Baseline/evaluate lưu `result.json` sau mỗi episode hoàn tất; metadata evaluate
ghi model sử dụng và chế độ action xác định. Timeout dữ liệu/đồng hồ trong
episode là infrastructure fault: run dừng và ghi `infrastructure_fault.json`,
không tự chạy lại episode và không tạo reward thất bại giả.

Sau khi đóng phiên cũ, reset chờ tối đa `startup_progress_timeout` (15 giây mặc
định) để ROS xóa thông tin publisher `/clock` và `/cmd_vel`. Nếu vẫn còn, run
dừng để tránh chạy chồng phiên; không tự đóng bất kỳ phiên bên ngoài nào.

RMS là căn của trung bình bình phương: ví dụ lỗi ngang
`sqrt(sum(error² × dt) / sum(dt))`. RMS phản ánh độ lớn lỗi qua thời gian và
không triệt tiêu lỗi trái/phải. Rung dùng RMS độ lớn vector gia tốc đã trừ trọng
lực, đơn vị m/s². Dữ liệu vẫn lấy tại các ranh giới step, chưa phải toàn bộ IMU
50 Hz. Log mới ghi `step_duration`; log cũ suy ra khoảng thời gian từ
`simulation_time`. Báo cáo không tự tuyên bố policy đạt yêu cầu; cần nhiều lượt,
nhiều seed và kiểm chứng định vị trước khi kết luận.

#### Kết quả đánh giá thực tế ngày 25/10/2026

Model `train_20261025_135101_536012/model.zip` đã học 10.000 bước, 9.000 cập
nhật. Đánh giá ba lượt mỗi chế độ, seed bắt đầu 101, cùng config đã lưu và map
`hall_01`, đích `(11, 0)`, giới hạn lệch 0,20 m:

| Chỉ số | PI | PI + SAC |
|---|---:|---:|
| Thành công | 3/3 | 2/3 |
| Trung bình RMS lệch đường | 0,04836 m | 0,04942 m |
| Trung bình RMS gia tốc đã trừ trọng lực | 5,491 m/s² | 6,181 m/s² |

Các trung bình dùng mọi episode hoàn tất, gồm cả thất bại. Lượt SAC thất bại
do monitor phát hiện lệch 0,202 m > 0,200 m; log sau khi dừng có residual bằng
0. Cả sáu episode có đúng một yêu cầu planning và reference không đổi trong
episode. Model hiện tại chưa chứng minh cải thiện ưu tiên bám đường; ba lượt
mỗi chế độ chưa đủ cho kết luận thống kê.

Nguồn: `runs/baseline_20261025_143141_617503/` và
`runs/evaluate_20261025_143817_026264/`. Báo cáo/CSV/ảnh ở
`runs/comparison_20261025_144251_308900/`. 30 test RL đã qua, ảnh báo cáo đã mở
kiểm tra. Các tiến trình mô phỏng/đánh giá đã đóng sau kiểm thử.

Các run bị gián đoạn vẫn được giữ: baseline `20261025_142915_231010` bị timeout
thông tin publisher cũ sau lượt đầu; evaluate `20261025_143620_644170` mất
odometry/clock và heartbeat giữa lượt đầu, không có terminal sample. Chúng
không thuộc phép so sánh sáu episode hoàn tất ở trên. Không nới watchdog hay
giới hạn an toàn để hoàn thành phép đánh giá. Sau sửa phân loại timeout lúc
chạy, lỗi này được ghi thành infrastructure fault thay vì reward giả.

Mỗi run có thư mục mới trong `rl_training/runs/`, không ghi đè kết quả cũ:

- `config.yaml`, `metadata.json`: cấu hình, thứ tự observation, seed, phiên bản.
- `episode_.../simulation.log`, `navigation.log`: log ROS/Gazebo.
- `episode_.../steps.jsonl`: lỗi đường, IMU, torque, action, reward từng step.
- `model.zip`, `replay_buffer.pkl`: mạng SAC và dữ liệu học khi train hoàn tất.
- `checkpoints/`: bản lưu định kỳ trong run dài.

Tạo ảnh đồ thị cho một episode:

```bash
python rl_training/plot_run.py \
  /ros2_ws/rl_training/runs/TEN_RUN/episode_TEN_EPISODE
```

Kết quả `episode.png` gồm độ lệch, gia tốc ba trục và mô-men. Dữ liệu này được
lấy tại ranh giới step RL, không phải bản ghi IMU đầy đủ 50 Hz. Có thể chạy
rosbag song song nếu cần toàn bộ mẫu IMU.

`/wheel_torque_base` là mô-men PI; `/wheel_residual_applied` là phần bù được
chấp nhận sau chặn trạng thái và giới hạn tổng mô-men, **trước** giới hạn độ
thay đổi/tốc độ. `/wheel_torque_applied` là tổng lệnh sau các bộ giới hạn đó.

## Reset và giới hạn nghiên cứu

Reset chỉ dịch robot về gốc sẽ không reset bộ tích phân odometry/PI, TF và AMCL.
Vì vậy mỗi episode ở bản đầu tạo phiên mô phỏng/Nav2 mới. Không xóa map, bag,
kết quả train hay Git. Chỉ dừng các tiến trình con của môi trường.

- Reward/độ lệch dùng vị trí AMCL: trượt ngang chưa được định vị nhận ra có thể
  không bị phạt đúng. Chưa dùng ground truth để huấn luyện.
- Quan sát tức thời chưa chứa lịch sử IMU, trạng thái tích phân PI hay thông tin
  dây phía trước; đây là bài toán quan sát chưa đầy đủ.
- Reward rung dùng mẫu tại mỗi step, chưa tích phân mọi mẫu IMU trong khoảng đó.
- Seed đặt cho NumPy/SAC; Gazebo và AMCL vẫn có nhiễu, không đảm bảo lặp lại bit.
- Đường được tính một lần mỗi episode; đường giữa các episode có thể khác nhẹ.
- Chỉ một môi trường tại một thời điểm; chưa hỗ trợ nhiều Gazebo song song.
- Trước khi gửi đích, môi trường đợi `/episode/ready` trả lời sẵn sàng ba lần
  liên tiếp. Dịch vụ này kiểm tra **TF trong chính EpisodeManager**, odometry,
  trạng thái dừng và các action server. Không gửi lại yêu cầu planning để xử lý
  lỗi TF khởi động; các lần hỏi sẵn sàng không tạo đường.
- Khi lỗi cảm biến/tiến trình, dừng run và đóng phiên thay vì học trên dữ liệu giả.
- Không dùng bản tích hợp này lên robot thật khi chưa có kiểm chứng riêng.

Cảnh báo `Unable to import Axes3D` do Matplotlib hệ thống và venv dùng nhầm
`mpl_toolkits` được xử lý bằng `plotting_compat.py`: chọn toolkit đi cùng bản
Matplotlib trong venv, không che cảnh báo. Các entrypoint train/plot đã gọi nó.

## Kiểm thử

```bash
colcon test --packages-select nino_control nino_navigation
colcon test-result --verbose
python -m pytest -q rl_training/tests
```

## Kết quả đã thực sự chạy ngày 24/10/2026

- Build ROS thành công; `colcon test-result` báo 33 test entries, 0 lỗi/thất bại
  (gồm các mục tổng hợp CTest). Ba test feature/reward trong venv cũng qua.
- Kiểm chứng Nav2 không RL: đích 1 m thành công; tạo độ lệch định vị có kiểm
  soát gây thất bại, giữ nguyên đường và dừng. Log ở
  `log/fixed_path_20261024_153100_879737/`.
- Môi trường RL ở chế độ baseline: đích 1 m thành công sau 42 bước, một yêu cầu
  planning. Run `runs/baseline_20261024_153512_257270/`.
- SAC smoke: 32 bước, 24 cập nhật gradient, hai episode/reset; mô-men bù quan
  sát lớn nhất `0.290453 N·m`, tổng mô-men trong ±2 N·m. Đường không đổi trong
  mỗi episode. `runs/smoke_20261024_153628_343282/verification.json` chứa bằng chứng.
- Đã nạp lại model, kiểm tra action hữu hạn và chạy evaluation trong Gazebo
  với giới hạn 1 giây; kết thúc `TIME_LIMIT` như cấu hình. Run
  `runs/evaluate_20261024_153821_383527/`. Đây không phải phép đánh giá hiệu quả.
- Đã xuất và xem ảnh `episode.png` của episode smoke đầu tiên.

Kiểm tra lại một run smoke (thay đường dẫn nếu tạo run mới):

```bash
python rl_training/tests/verify_run.py \
  rl_training/runs/smoke_20261024_153628_343282
```

Trong quá trình tích hợp đã gặp pose ban đầu gửi quá sớm và một lần timeout
dịch vụ lifecycle. Môi trường hiện đợi control/bridge ổn định và AMCL active
trước khi gửi pose; nếu vẫn timeout, run dừng và giữ log để kiểm tra. Chưa coi
startup Gazebo/Nav2 là hoàn toàn ổn định trên mọi máy.

## Cập nhật GPU và đích cuối hành lang — 25/10/2026 (Bangkok)

- Đã cài `torch 2.7.1+cu128`, kiểm tra forward/backward thật trên RTX 5060.
  `pip check` không báo dependency hỏng. Không thay Docker/WSLg hoặc driver.
- Mặc định train/evaluate là CUDA; đích mặc định đổi thành `(11, 0)`.
- Sửa lỗi từ chối episode do TF trong EpisodeManager chưa sẵn sàng bằng dịch
  vụ `/episode/ready`; không tăng ngưỡng tuổi TF và không gọi lại planner.
- Sửa việc observation cũ sau cập nhật CUDA: xử lý tiếp callback cảm biến/TF
  tới khi có dữ liệu hợp lệ trong timeout hiện có. Không che NaN hoặc cho học
  bằng dữ liệu cũ. Lỗi dữ liệu thực sự vẫn dừng run.
- Đã kiểm tra import Axes3D đúng bản Matplotlib, không còn cảnh báo trong train.
- Tám test RL đều qua; kết quả colcon hiện 34 test entries, 0 lỗi/thất bại.
- Run GPU `runs/smoke_20261024_182447_635855/`: 32 bước, 24 cập nhật gradient,
  4 episode/reset. Model train và nạp lại trên CUDA; mô-men bù cao nhất
  `0.293661 N·m`, tổng trong ±2 N·m; đường giữ nguyên trong mỗi episode.
  Đích yêu cầu và điểm cuối đường đều `(11, 0)`. Xem `verification.json`.

Run này dùng episode ngắn 2 giây để kiểm tra luồng GPU/reset, **chưa phải xác
nhận robot đã đi hết hành lang**. Chạy train thông thường để học rồi đánh giá
đủ episode với cùng cấu hình như baseline.

```bash
python rl_training/tests/verify_run.py \
  rl_training/runs/smoke_20261024_182447_635855 --device cuda
```

## Sửa timeout odometry đầu episode — 25/10/2026

EpisodeManager hiện nhận `/odom` bằng một luồng chỉ làm nhiệm vụ nhận dữ liệu
và ghi thời điểm tới. Luồng chính vẫn quản lý trạng thái/action và lệnh dừng.
Nhờ đó, xử lý đường hoặc gửi action lâu không làm thời điểm nhận odometry bị
nhầm với thời điểm callback ở luồng chính được chạy.

Trước khi bắt đầu, `/episode/ready` yêu cầu odometry có timestamp tăng và tới
đều trong ít nhất `odom_warmup_duration: 1.0` giây. Sau khi chạy, ngưỡng mất
odometry **vẫn là 0,5 giây**, không tăng để che lỗi. Timestamp lặp lại không làm
mới dữ liệu; đồng hồ quay ngược cũng được xử lý như lỗi dữ liệu.

Mất odometry/TF thật vẫn hủy action và giữ vận tốc 0. Result có mã `65001` để
môi trường RL phân biệt lỗi dữ liệu với lỗi robot lệch đường. Lỗi này **không
trả về transition phạt −30** cho SAC: run dừng, lưu
`infrastructure_fault_model.zip`, `infrastructure_fault_replay.pkl` và
`infrastructure_fault.json` khi có model. Lệch đường vẫn là thất bại nhiệm vụ
và bị phạt như trước. Không tự lập lại đường hoặc tự tiếp tục khi mất cảm biến.

Build lại trước lần train tiếp theo:

```bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select nino_navigation
source install/setup.bash
source rl_training/.venv/bin/activate
python rl_training/train.py --mode train --map /ros2_ws/maps/hall_01.yaml \
  --device cuda --steps 10000
```

Mặc định không mở GUI để giảm tải lúc train. Có thể thêm `--gui` khi cần quan
sát. Không chạy thêm một phiên Gazebo/Nav2 khác cùng lúc.

Sau khi dừng phiên mô phỏng do môi trường tạo, reset chờ tối đa 5 giây cho ROS
graph bỏ thông tin publisher cũ. Việc DDS thông báo chậm không còn bị kết luận
ngay là một phiên Gazebo khác. Hết thời gian vẫn từ chối, không tắt node ngoài.

Kiểm chứng sau sửa: 42 test entries ROS, 0 lỗi/thất bại; 11 test RL đều qua.
Có test giữ luồng chính bận 0,7 giây trong khi odometry tiếp tục phát, xác nhận
không báo mất odometry giả; test ngắt odometry thật vẫn hủy/dừng và trả mã lỗi
dữ liệu. Run `runs/smoke_20261024_192000_306354/` dùng CUDA, không GUI: 32 bước,
24 cập nhật mạng qua 3 episode/reset, không gặp timeout odometry. Đã kiểm tra
model lưu/nạp trên CUDA, đường không đổi và mô-men vẫn trong giới hạn. Xem
`verification.json` trong run đó. Đây vẫn là kiểm chứng tích hợp ngắn.

Kiểm tra với GUI đã qua đầu episode sau sửa nhưng lần mở GUI tiếp theo gặp
`Failed to create OpenGL context` / `D3D12: Removing Device`. Đó là lỗi hiển thị
riêng, chưa sửa thiết lập GPU/WSLg. Vì vậy hãy chạy train mặc định **không thêm
`--gui`** trước; CUDA vẫn được dùng cho mạng SAC.

## Phục hồi lỗi kích hoạt Nav2 khi reset

Nếu service chuyển trạng thái bị mất phản hồi, một node có thể ở `inactive`
trong khi Lifecycle Manager đang chờ. Trước khi gửi đích, môi trường hiện kiểm
tra tên và trạng thái từng node; nếu không tiến triển trong
`startup_progress_timeout` (mặc định 15 giây), nó đóng phiên do nó tạo và thử
khởi động lại. `startup_attempts: 3` là tối đa ba lần thử, gồm lần đầu.

Việc thử lại chỉ áp dụng lúc khởi động, **trước khi gửi goal navigation**.
Policy và replay được giữ trong bộ nhớ; không tạo transition/reward cho lần
khởi động lỗi và không gọi planner trong lần lỗi. Lỗi lúc robot đang chạy vẫn
dừng an toàn, không tự khởi động lại episode đang thực hiện. Lỗi cấu hình hoặc
một tiến trình tự thoát không bị che bằng vòng thử lại này.

`startup_recovery.jsonl` lưu từng lần lỗi, tên node/trạng thái và thư mục log.
Nếu hết ba lần thử, run dừng và lưu model/replay qua cơ chế infrastructure
fault. Không thay GPU/WSLg, thuật toán SAC hoặc ngưỡng lệch đường.

Kiểm tra khả năng phục hồi với một lỗi khởi động cố ý trước goal:

```bash
python rl_training/tests/verify_startup_recovery.py --mode smoke \
  --map /ros2_ws/maps/hall_01.yaml --device cuda --steps 64
```

Để kiểm tra riêng timeout chờ AMCL nhận `/initialpose`:

```bash
python rl_training/tests/verify_startup_recovery.py --fault-stage initialpose \
  --mode smoke --map /ros2_ws/maps/hall_01.yaml --device cuda --steps 32
```

Reset kiểm tra `map_server` và AMCL active trước khi chờ subscriber `/initialpose`.
Timeout ROS trước khi gửi goal được ghi rõ bước lỗi và thử lại có giới hạn như
lỗi lifecycle. Ngay trước khi gửi goal, quyền thử lại khởi động kết thúc, kể cả
khi phản hồi chấp nhận goal bị thất lạc. Lỗi cấu hình và tiến trình tự thoát vẫn
dừng run. Đây là phục hồi khởi động, không sửa bản thân lỗi truyền thông ROS.

Kiểm chứng bổ sung ngày 25/10/2026: 18 test RL qua. Run
`runs/smoke_20261025_134806_306146/` gây timeout `/initialpose` ở lần đầu,
phục hồi trước goal rồi hoàn tất 32 bước, 24 cập nhật SAC trên CUDA qua 3 episode.
`verification.json` xác nhận model nạp được, một lần planning mỗi episode,
reference không đổi và tổng mô-men trong ±2 N·m. Đây chưa phải kiểm chứng policy
học tốt hoặc độ ổn định của một run dài.

Đây là script kiểm thử; train bình thường vẫn dùng `rl_training/train.py`.

Đã kiểm chứng ngày 25/10/2026: 17 test RL qua. Run
`runs/smoke_20261024_200421_648186/` cố ý cho lần khởi động đầu thiếu một service
lifecycle; môi trường ghi lỗi rồi phục hồi trước goal, sau đó chạy 5 episode,
64 bước và 56 cập nhật mạng trên CUDA. Model lưu/nạp được, mỗi episode có một
yêu cầu planning và đường không đổi; tổng mô-men trong ±2 N·m. Xem
`startup_recovery.jsonl` và `verification.json`. Đây là kiểm chứng phục hồi
khởi động và tích hợp, chưa phải đánh giá chất lượng policy.

Log lỗi thực tế trước sửa ghi `controller_server/change_state` không gửi được
phản hồi; đây là dấu hiệu tương tự
[race condition dịch vụ Fast DDS được ROS 2 ghi nhận](https://github.com/ros2/rmw_fastrtps/issues/842).
Phục hồi có giới hạn không khẳng định đã sửa bản thân middleware; nếu lỗi vẫn
lặp lại cả ba lần, chương trình dừng và giữ bằng chứng thay vì lặp vô hạn.
