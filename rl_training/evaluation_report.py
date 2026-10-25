"""Standalone Vietnamese RMSE report using only recorded evaluation samples."""

from html import escape
import json
from pathlib import Path

import numpy as np


COLORS = ('#176bba', '#e17a18')
LABELS = ('PI', 'PI + SAC')


def sample_rmse(errors):
    errors = np.asarray(errors, dtype=float)
    if not errors.size or not np.all(np.isfinite(errors)):
        raise ValueError('RMSE requires finite, nonempty errors')
    return float(np.sqrt(np.mean(errors**2)))


def comparison_data(report):
    groups = []
    for key in ('baseline', 'policy'):
        group = report[key]
        episodes = []
        for e in group['episodes']:
            if e['outcome'] == 'INCOMPLETE':
                continue
            path = Path(group['run'])/e['episode']/'steps.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            episodes.append(dict(name=e['episode'], outcome=e['outcome'], reason=e['reason'],
                                 t=[r['simulation_time'] for r in rows],
                                 progress=[r['progress'] for r in rows],
                                 error_cm=[100*r['cross_track'] for r in rows],
                                 acceleration=[float(np.linalg.norm(r['acceleration'])) for r in rows],
                                 torque=[r['total_torque'] for r in rows]))
        groups.append(episodes)
    pairs = []
    for index, (base, policy) in enumerate(zip(*groups)):
        end = min(max(base['progress']), max(policy['progress']))
        start = max(min(base['progress']), min(policy['progress']))
        if end <= start:
            raise ValueError('Paired episodes have no common progress interval')
        rmses = []
        for episode in (base, policy):
            errors = [e for p, e in zip(episode['progress'], episode['error_cm']) if start <= p <= end]
            rmses.append(sample_rmse(errors))
        pairs.append(dict(pair=index+1, start_m=start, end_m=end,
                          baseline_rmse_cm=rmses[0], policy_rmse_cm=rmses[1]))
    return groups, pairs


def figures(report, output, groups):
    from plotting_compat import prepare_matplotlib_namespace
    prepare_matplotlib_namespace()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, key in enumerate(('baseline', 'policy')):
        aggregate = report[key]['aggregate']
        axes[0, 0].bar(i, aggregate['success_rate']*100, color=COLORS[i], width=0.48)
        axes[0, 0].text(i, aggregate['success_rate']*100+3,
                        f"{aggregate['outcome_counts']['SUCCEEDED']}/{aggregate['completed_episodes']}",
                        ha='center', fontweight='bold')
    axes[0, 0].set(title='1. Đến đích an toàn', ylabel='Tỷ lệ thành công (%)', ylim=(0, 115),
                   xticks=[0, 1], xticklabels=LABELS)
    for i, episodes in enumerate(groups):
        x = np.arange(len(episodes)) + (i-0.5)*0.34
        values = [sample_rmse(e['error_cm']) for e in episodes]
        bars = axes[0, 1].bar(x, values, width=0.32, color=COLORS[i], label=LABELS[i])
        for bar, episode, value in zip(bars, episodes, values):
            if episode['outcome'] != 'SUCCEEDED':
                bar.set_hatch('///')
                bar.set_edgecolor('#9b1c1c')
            axes[0, 1].text(bar.get_x()+bar.get_width()/2, value+0.06, f'{value:.2f}',
                            ha='center', fontsize=9)
        for j, episode in enumerate(episodes):
            marker = 'o' if episode['outcome'] == 'SUCCEEDED' else 'X'
            axes[1, 0].scatter(j+(i-0.5)*0.18, sample_rmse(episode['acceleration']),
                               color=COLORS[i], marker=marker, s=80,
                               label=LABELS[i] if j == 0 else None)
            axes[1, 1].plot(episode['progress'], np.abs(episode['error_cm']),
                            color=COLORS[i], alpha=0.65, lw=1,
                            label=LABELS[i] if j == 0 else None)
    n = len(groups[0])
    axes[0, 1].set(title='2. RMSE lệch đường từng lượt', ylabel='RMSE (cm)',
                   xticks=np.arange(n), xticklabels=[f'Lượt {i+1}' for i in range(n)])
    axes[1, 0].set(title='3. Mức gia tốc IMU đã trừ trọng lực', ylabel='RMS gia tốc (m/s²)',
                   xticks=np.arange(n), xticklabels=[f'Lượt {i+1}' for i in range(n)])
    threshold = 100*report['baseline']['config']['environment']['max_path_deviation']
    axes[1, 1].axhline(threshold, color='#ab2630', ls='--', label=f'Ngưỡng {threshold:g} cm')
    axes[1, 1].set(title='4. Lỗi xuất hiện ở đâu?', xlabel='Tiến triển dọc đường (m)', ylabel='|Lỗi ngang| (cm)')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=0.17)
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.legend(fontsize=9)
    fig.suptitle('NINO • PI so với PI + Residual SAC\nDữ liệu chạy thật trong Gazebo · gạch chéo / X: thất bại',
                 fontsize=17, fontweight='bold')
    for ext in ('png', 'svg'):
        fig.savefig(output/f'overview.{ext}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(n, 2, figsize=(13, 3*n), squeeze=False,
                             constrained_layout=True, sharex=True, sharey=True)
    for i, episodes in enumerate(groups):
        for j, episode in enumerate(episodes):
            ax = axes[j, i]
            ax.plot(episode['progress'], episode['error_cm'], color=COLORS[i], lw=1.1)
            ax.axhline(threshold, color='#ab2630', ls='--', lw=0.8)
            ax.axhline(-threshold, color='#ab2630', ls='--', lw=0.8)
            ax.axhline(0, color='#8b98a8', lw=0.6)
            ax.set(title=f"{LABELS[i]} · lượt {j+1} · {episode['outcome']} · RMSE {sample_rmse(episode['error_cm']):.2f} cm",
                   ylabel='Lỗi ngang (cm)', ylim=(-threshold*1.3, threshold*1.3))
            ax.grid(alpha=0.15)
            if episode['outcome'] != 'SUCCEEDED':
                ax.scatter(episode['progress'][-1], episode['error_cm'][-1], color='#ab2630', marker='X', zorder=5)
    for ax in axes[-1]:
        ax.set_xlabel('Tiến triển dọc đường (m)')
    fig.suptitle(f'Giữ cả lượt thất bại trong báo cáo · cùng thang đo cho {2*n} lượt', fontsize=16)
    for ext in ('png', 'svg'):
        fig.savefig(output/f'path_traces.{ext}', dpi=180)
    plt.close(fig)


def write_detailed_report(report, output):
    groups, pairs = comparison_data(report)
    figures(report, output, groups)
    summary = []
    for group in groups:
        per_episode = [sample_rmse(e['error_cm']) for e in group]
        summary.append(dict(mean_episode_rmse_cm=float(np.mean(per_episode)),
                            pooled_rmse_cm=sample_rmse([v for e in group for v in e['error_cm']]),
                            episode_rmse_cm=per_episode,
                            mean_acceleration_rms=float(np.mean([sample_rmse(e['acceleration']) for e in group]))))
    difference = (summary[1]['mean_episode_rmse_cm']/summary[0]['mean_episode_rmse_cm']-1)*100
    data = dict(groups=groups, common_progress=pairs, summary=summary,
                source_runs=[report[k]['run'] for k in ('baseline', 'policy')])
    (output/'rmse_analysis.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    rows = []
    for i, group in enumerate(groups):
        for j, episode in enumerate(group):
            rows.append(f'<tr><td>{LABELS[i]}</td><td>{j+1}</td><td>{escape(episode["outcome"])}</td>'
                        f'<td>{summary[i]["episode_rmse_cm"][j]:.3f}</td>'
                        f'<td>{max(abs(v) for v in episode["error_cm"]):.2f}</td>'
                        f'<td>{episode["t"][-1]:.2f}</td><td>{max(episode["progress"]):.2f}</td>'
                        f'<td>{escape(episode["reason"]) or "—"}</td></tr>')
    common_rows = ''.join(f'<tr><td>{p["pair"]}</td><td>{p["start_m"]:.2f}–{p["end_m"]:.2f}</td>'
                          f'<td>{p["baseline_rmse_cm"]:.3f}</td><td>{p["policy_rmse_cm"]:.3f}</td></tr>' for p in pairs)
    environment = report['baseline']['config']['environment']
    success = [report[k]['aggregate']['outcome_counts']['SUCCEEDED'] for k in ('baseline', 'policy')]
    n = len(groups[0])
    conclusion = (f'PI đến đích {success[0]}/{n} lượt; PI + SAC đến đích {success[1]}/{n} lượt. '
                  f'RMSE trung bình từng lượt: {summary[0]["mean_episode_rmse_cm"]:.3f} cm và '
                  f'{summary[1]["mean_episode_rmse_cm"]:.3f} cm (SAC thay đổi {difference:+.1f}%). '
                  'Không suy ra ý nghĩa thống kê từ mẫu thử nhỏ này; chưa chứng minh được mức cải thiện tổng quát của RL.')
    payload = json.dumps(dict(groups=groups, threshold_cm=100*environment['max_path_deviation']),
                         ensure_ascii=False).replace('<', '\\u003c')
    html = HTML.replace('@@CONCLUSION@@', escape(conclusion)).replace('@@ROWS@@', ''.join(rows))
    html = html.replace('@@COMMON@@', common_rows).replace('@@DATA@@', payload)
    html = html.replace('@@MODEL@@', escape(str(report['policy']['metadata'].get('model', 'Không ghi nhận'))))
    html = html.replace('@@SOURCES@@', '<br>'.join(escape(v) for v in data['source_runs']))
    html = html.replace('@@CONFIG@@', escape(f"Đích ({environment['goal_x']}, {environment['goal_y']}) m · "
                                           f"Ngưỡng {environment['max_path_deviation']} m · Seed bắt đầu "
                                           f"{report['baseline']['metadata']['seed']} · {n} lượt / chế độ"))
    html = html.replace('@@COUNT@@', str(2*n)).replace('@@N@@', str(n))
    html = html.replace('@@THRESHOLD@@', f"{100*environment['max_path_deviation']:g}")
    for i in (0, 1):
        html = html.replace(f'@@RMSE{i}@@', f'{summary[i]["mean_episode_rmse_cm"]:.2f}')
        html = html.replace(f'@@SUCCESS{i}@@', f'{success[i]}/{n}')
        html = html.replace(f'@@POOLED{i}@@', f'{summary[i]["pooled_rmse_cm"]:.3f}')
    (output/'report.html').write_text(html, encoding='utf-8')
    markdown = '# Kiểm thử PI và PI + Residual SAC bằng RMSE\n\n'+conclusion+'\n\n'
    markdown += 'Bộ điều khiển nền là PI (P + I, không có D). SAC bổ sung mô-men, không thay PI.\n\n'
    markdown += 'RMSE = sqrt(sum(e_i²) / N), e_i là lỗi ngang so với đường tham chiếu.\n'
    markdown += 'Báo cáo dùng mẫu tại ranh giới step RL; đây là dữ liệu chạy Gazebo đã ghi, không chạy mô phỏng mới.\n\n'
    markdown += '| Bộ điều khiển | Lượt | Kết quả | RMSE (cm) |\n|---|---:|---|---:|\n'
    for i, group in enumerate(groups):
        for j, episode in enumerate(group):
            markdown += f'| {LABELS[i]} | {j+1} | {episode["outcome"]} | {summary[i]["episode_rmse_cm"][j]:.3f} |\n'
    markdown += '\n![Tổng quan](overview.png)\n\n![Lỗi từng lượt](path_traces.png)\n\n'
    markdown += 'Báo cáo tương tác và phương pháp đầy đủ: [report.html](report.html).\n\nNguồn:\n\n'
    markdown += '\n'.join('- '+v for v in data['source_runs'])+'\n'
    (output/'report.md').write_text(markdown, encoding='utf-8')


HTML = r'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nino — kiểm thử RMSE PI và SAC</title>
<style>
:root{font-family:system-ui,Arial;color:#18293e;background:#edf2f7;font-size:16px}body{margin:0}
main{max-width:1150px;margin:auto;padding:40px 24px}header{background:#142c48;color:white;padding:40px;border-radius:18px}
h1{font-size:36px;line-height:1.15;margin:12px 0}h2{font-size:24px;margin-top:0}.eyebrow{color:#a8d8ff;letter-spacing:2px;font-size:12px}
p,li{line-height:1.65}.muted{color:#526579}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}
.card,section{background:white;border:1px solid #d9e3ed;border-radius:14px;padding:26px;margin-top:20px}
.cards .card{border-top:5px solid #176bba;margin:0}.cards .card:last-child{border-top-color:#e17a18}
.value{font-size:36px;font-weight:750}.unit{font-size:16px;color:#526579}.tag{background:#eef4fa;padding:5px 10px;border-radius:8px;font-size:13px}
.callout{border-left:5px solid #c44b25;background:#fff7ed;padding:14px 20px;margin-top:20px}
img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dfe7ef;padding:12px 8px}
th{background:#f3f6fa}.scroll{overflow:auto}a{color:#176bba}code{overflow-wrap:anywhere;font-size:12px}
select,button{font:inherit;padding:8px 12px;border:1px solid #cbd5e1;border-radius:7px;background:white}
.tools{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:16px 0}svg{width:100%;display:block}#tip{min-height:28px;color:#364c65}
@media(max-width:650px){main{padding:16px}.cards{grid-template-columns:1fr}header,section{padding:20px}h1{font-size:28px}}
@media print{body{background:white}main{padding:0;max-width:none}section{break-inside:avoid}button,.tools{display:none}header{color:#142c48;background:white;padding:12px}a{color:inherit}.eyebrow{color:#142c48}}
</style><main>
<header><div class="eyebrow">NINO ROBOTICS / ĐÁNH GIÁ SAU HUẤN LUYỆN</div>
<h1>RL có giúp robot bám đường tốt hơn?</h1><p>Kiểm thử bằng RMSE • @@COUNT@@ lượt chạy thực tế trong Gazebo • đánh giá policy SAC đã lưu</p>
<span class="tag" style="color:#142c48">PI so với PI + Residual SAC</span></header>
<div class="cards"><div class="card"><b>ĐIỀU KHIỂN NỀN · PI</b><div class="value">@@RMSE0@@ <span class="unit">cm RMSE</span></div><p>Đến đích <b>@@SUCCESS0@@ lượt</b> · không có mô-men bù RL</p></div>
<div class="card"><b>PI + RESIDUAL SAC</b><div class="value">@@RMSE1@@ <span class="unit">cm RMSE</span></div><p>Đến đích <b>@@SUCCESS1@@ lượt</b> · mô-men bù tối đa ±0,3 N·m</p></div></div>
<div class="callout">@@CONCLUSION@@</div>
<section><h2>01 / Tổng quan kết quả</h2><p class="muted">RMSE thấp hơn là tốt hơn, nhưng phải đọc cùng tỷ lệ đến đích. Lượt thất bại được giữ trong mọi trung bình.</p>
<img src="overview.svg" alt="Tỷ lệ thành công, RMSE từng lượt, gia tốc IMU và lỗi theo tiến triển dọc đường"></section>
<section><h2>02 / Xem trực tiếp từng lượt</h2><p>Chọn lượt, rồi đưa chuột lên biểu đồ để xem lỗi ngang tại mẫu gần nhất. Hai lượt cùng số thứ tự dùng cùng seed yêu cầu; nhiễu Gazebo/AMCL vẫn có thể khác nhau.</p>
<div class="tools"><label>Lượt <select id="pair"></select></label><label>Trục ngang <select id="axis"><option value="progress">Tiến triển dọc đường (m)</option><option value="t">Thời gian mô phỏng (s)</option></select></label><button onclick="window.print()">In / lưu PDF từ trình duyệt</button></div>
<p><b style="color:#176bba">━━ PI</b> &nbsp; <b style="color:#e17a18">━━ PI + SAC</b> &nbsp; <span style="color:#ab2630">┄ Ngưỡng thất bại ±@@THRESHOLD@@ cm</span></p>
<svg id="chart" viewBox="0 0 1000 320" role="img" aria-label="Biểu đồ lỗi ngang tương tác"></svg><div id="tip">Đưa chuột lên biểu đồ để đọc giá trị.</div><p id="outcomes" class="muted"></p></section>
<section><h2>03 / @@COUNT@@ lượt, cùng thang đo</h2><img src="path_traces.svg" alt="Lỗi ngang từng lượt và ngưỡng thất bại">
<div class="scroll"><table><thead><tr><th>Bộ điều khiển</th><th>Lượt</th><th>Kết quả</th><th>RMSE (cm)</th><th>Đỉnh |e| (cm)</th><th>Thời gian (s)</th><th>Tiến triển tối đa (m)</th><th>Lý do</th></tr></thead><tbody>@@ROWS@@</tbody></table></div></section>
<section><h2>04 / Đối chiếu trên đoạn đường chung</h2><p>Với mỗi cặp, chỉ lấy mẫu trong khoảng tiến triển mà cả hai lượt đã đi qua. Không kéo dài lượt thất bại, không nội suy lỗi. Số mẫu và thời gian lưu tại mỗi vị trí có thể khác nhau; đây chưa phải thử nghiệm kiểm soát hoàn toàn.</p>
<table><thead><tr><th>Cặp</th><th>Đoạn chung (m)</th><th>PI RMSE (cm)</th><th>PI + SAC RMSE (cm)</th></tr></thead><tbody>@@COMMON@@</tbody></table></section>
<section><h2>05 / RMSE được tính thế nào?</h2><p><b>RMSE = √[(e₁² + e₂² + … + eₙ²) / N]</b>, với e là lỗi ngang so với đường tham chiếu. Bình phương giúp lỗi trái/phải không triệt tiêu nhau; căn bậc hai đưa kết quả về đơn vị mét hoặc centimet.</p>
<p>Con số lớn ở đầu báo cáo là <b>trung bình RMSE của từng episode</b>: mỗi lượt có trọng số bằng nhau. Nếu gộp tất cả mẫu rồi tính một RMSE, PI là <b>@@POOLED0@@ cm</b>, SAC là <b>@@POOLED1@@ cm</b>; cách này cho lượt dài nhiều trọng số hơn. Cả hai được lưu trong JSON.</p>
<p><b>RMSE bám đường</b> khác với <b>RMS gia tốc</b>: gia tốc được trừ trọng lực nhưng không có tín hiệu gia tốc mục tiêu, nên chỉ gọi là RMS mức gia tốc, không gọi là sai số RMSE của IMU. Gia tốc này còn chứa tăng/giảm tốc chủ động.</p>
<p>File report.json vẫn giữ RMS lỗi có trọng số thời gian: √[Σ(e²·Δt)/ΣΔt]. Đây là chỉ số bổ sung, không thay thế công thức RMSE theo N mẫu ở báo cáo này. Mẫu terminal trùng timestamp được giữ trong RMSE theo N mẫu, có thời lượng 0 khi tính RMS theo thời gian.</p></section>
<section><h2>06 / Điều kiện, nguồn dữ liệu và giới hạn</h2><p>@@CONFIG@@</p>
<p>Code nền hiện chỉ có P + I, không có thành phần D; vì vậy gọi là PI, chưa phải PID. RL cộng mô-men vào PI, không thay bộ điều khiển nền. Mỗi lượt chỉ tính đường một lần; reference được kiểm tra không đổi trong lượt.</p>
<p><b>Dùng dữ liệu đã ghi từ các phép chạy Gazebo trong hai run nguồn; không chạy thêm mô phỏng khi tạo báo cáo.</b> So sánh giữ nguyên cấu hình đã lưu và map. Model evaluate dùng action xác định và không học thêm.</p>
<ul><li>Có @@N@@ lượt mỗi chế độ: cần nhiều seed và số lượt lớn hơn trước khi kết luận thống kê hay chất lượng tổng quát của policy.</li><li>Lỗi vị trí dùng AMCL, chưa phải ground truth; có thể bỏ sót trượt thực.</li><li>Dữ liệu IMU lấy ở ranh giới step RL khoảng 10 Hz, không phải toàn bộ 50 Hz; có thể bỏ sót đỉnh rung.</li><li>Đường giữa các lượt có thể khác nhẹ dù không replanning trong lượt. Cùng map path không chứng minh nội dung map lịch sử giống hệt.</li><li>Run gián đoạn do hạ tầng được giữ riêng và không tính là thất bại policy. Báo cáo này gồm mọi episode hoàn tất trong hai run nguồn, kể cả episode lệch quá ngưỡng.</li></ul>
<p>Model: <code>@@MODEL@@</code></p><p>Nguồn: <code>@@SOURCES@@</code></p>
<p><a href="episodes.csv">CSV từng lượt</a> · <a href="report.json">Chỉ số có trọng số thời gian</a> · <a href="rmse_analysis.json">RMSE và dữ liệu đồ thị</a> · <a href="overview.png">Ảnh tổng quan PNG</a> · <a href="path_traces.png">Ảnh lỗi từng lượt PNG</a> · <a href="report.md">Báo cáo Markdown</a></p></section>
</main><script>
const D=@@DATA@@, NS='http://www.w3.org/2000/svg';const pair=document.querySelector('#pair'),axis=document.querySelector('#axis'),chart=document.querySelector('#chart');
D.groups[0].forEach((_,i)=>pair.add(new Option('Lượt '+(i+1),i)));
function el(name,attrs,text){let e=document.createElementNS(NS,name);for(let k in attrs)e.setAttribute(k,attrs[k]);if(text!==undefined)e.textContent=text;chart.appendChild(e);return e}
function draw(){chart.replaceChildren();let j=+pair.value,key=axis.value,A=D.groups.map(g=>g[j]),M=Math.max(...A.flatMap(e=>e[key])),Y=Math.max(D.threshold_cm*1.3,...A.flatMap(e=>e.error_cm.map(Math.abs)))*1.03;
let X=v=>65+v/M*900,F=v=>145-v/Y*120;
for(let i=0;i<=5;i++){let v=M*i/5;el('line',{x1:X(v),y1:20,x2:X(v),y2:270,stroke:'#edf1f5'});el('text',{x:X(v),y:290,'text-anchor':'middle',fill:'#526579','font-size':12},v.toFixed(1))}
[-20,-10,0,10,20].forEach(v=>{el('line',{x1:65,x2:965,y1:F(v),y2:F(v),stroke:'#e1e7ee'});el('text',{x:52,y:F(v)+4,'text-anchor':'end',fill:'#526579','font-size':12},v)});
[D.threshold_cm,-D.threshold_cm].forEach(v=>el('line',{x1:65,x2:965,y1:F(v),y2:F(v),stroke:'#ab2630','stroke-dasharray':'6 5'}));
A.forEach((e,i)=>{el('path',{d:e[key].map((x,k)=>(k?'L':'M')+X(x).toFixed(2)+' '+F(e.error_cm[k]).toFixed(2)).join(' '),fill:'none',stroke:['#176bba','#e17a18'][i],'stroke-width':1.7});if(e.outcome!=='SUCCEEDED'){let k=e.t.length-1;el('circle',{cx:X(e[key][k]),cy:F(e.error_cm[k]),r:5,fill:'#ab2630'})}});
el('text',{x:65,y:14,fill:'#526579','font-size':12},'Lỗi ngang (cm)');el('text',{x:510,y:315,'text-anchor':'middle',fill:'#526579','font-size':12},key==='t'?'Thời gian mô phỏng (s)':'Tiến triển dọc đường (m)');
document.querySelector('#outcomes').textContent=A.map((e,i)=>(i?'PI + SAC':'PI')+': '+e.outcome+(e.reason?' — '+e.reason:'')).join(' | ');
chart.onmousemove=event=>{let rect=chart.getBoundingClientRect(),x=((event.clientX-rect.left)/rect.width*1000-65)/900*M;x=Math.max(0,Math.min(M,x));let text=A.map((e,i)=>{if(x<Math.min(...e[key])||x>Math.max(...e[key]))return(i?'SAC':'PI')+': ngoài đoạn đã ghi';let nearest=0;for(let k=1;k<e[key].length;k++)if(Math.abs(e[key][k]-x)<Math.abs(e[key][nearest]-x))nearest=k;return(i?'SAC':'PI')+': e = '+e.error_cm[nearest].toFixed(3)+' cm, t = '+e.t[nearest].toFixed(2)+' s'});document.querySelector('#tip').textContent=(key==='t'?'t = ':'s = ')+x.toFixed(2)+(key==='t'?' s':' m')+' | '+text.join(' | ')};
}pair.onchange=draw;axis.onchange=draw;draw();
</script></html>'''
